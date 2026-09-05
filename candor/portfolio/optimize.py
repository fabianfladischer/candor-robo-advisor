"""Restringierte Mean-Variance-Optimierung mit Nachhaltigkeitsfilter.

Das Problem
-----------
Gesucht sind Gewichte ``w``, die bei gegebener Mindestrendite die
Portfoliovarianz minimieren::

    min  w' Σ w
    u.d.N.
         w' µ  >=  r_ziel          (Zielrendite)
         Σ w   =   1               (Vollinvestition)
         0 <= w <= w_max           (keine Leerverkäufe, Positionsobergrenze)
         l_i <= w_i <= u_i         (Bandbreiten je Assetklasse)
         L_g <= Σ_{i∈g} w_i <= U_g (Bandbreiten je Assetgruppe)

Das ist ein konvexes quadratisches Programm; ein lokales Optimum ist damit
global. Die Restriktionen stammen vollständig aus ``policies/classes_v1.yaml``
und nicht aus dem Code (Prinzip P3).

Warum die Restriktionen keine Beiwerk sind
------------------------------------------
Die unrestringierte MVO ist als Verfahren instabil: Sie liest Schätzfehler in
den erwarteten Renditen als Signal und konzentriert auf wenige Positionen
(Michaud, 1989). Die Bänder sind die wirksamste bekannte Gegenmaßnahme — sie
begrenzen den Schaden, den ein Schätzfehler anrichten kann, unabhängig davon,
wie groß er ist. Zusammen mit der Schrumpfung beider Momente (siehe
:mod:`candor.portfolio.estimate`) und der Pflicht-Benchmark 1/N (siehe
:mod:`candor.portfolio.benchmark`) bilden sie die drei Sicherungen gegen die
Fehlerverstärkung der MVO.

Reihenfolge der Schritte
------------------------
Der Nachhaltigkeitsfilter wirkt **vor** der Optimierung, nicht danach. Ein
nachträgliches Streichen von Positionen würde die Restriktionen verletzen und
das Ergebnis stillschweigend unzulässig machen. Unterschreitet das gefilterte
Universum die Mindestgröße, bricht die Funktion ab (Invariante I8, Prinzip P5):
Eine Nachhaltigkeitspräferenz still aufzuweichen wäre ein Verstoß gegen
Art. 54 Abs. 9 f. DelVO (EU) 2017/565 in der Fassung der DelVO (EU) 2021/1253.

Solver
------
Zuerst die in der Policy genannten cvxpy-Solver in ihrer Reihenfolge. Meldet
keiner einen zulässigen Status, folgt scipy SLSQP als Rückfallverfahren. Der
Wechsel ist kein Detail, sondern wird im Trace vermerkt — ein Ergebnis aus dem
Rückfallverfahren ist nicht dasselbe wie eines aus dem Innere-Punkte-Verfahren.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Final

import cvxpy as cp
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from candor import format_de
from candor.domain.client import SustainabilityPreferences
from candor.domain.errors import DomainError
from candor.domain.policy import ClassesPolicy, PortfolioPolicy, RiskClassDefinition
from candor.portfolio.estimate import MomentEstimate

FloatArray = npt.NDArray[np.float64]

#: Rechtsgrundlage des Nachhaltigkeitsfilters.
SUSTAINABILITY_LEGAL_BASIS: Final = (
    "Art. 54 Abs. 9 und 10 DelVO (EU) 2017/565 i. d. F. der DelVO (EU) 2021/1253"
)

#: Rundung der ausgewiesenen Nachhaltigkeitsanteile. Vier Nachkommastellen
#: genügen für eine Prozentangabe mit zwei Nachkommastellen und halten den
#: Trace über Läufe hinweg bitgleich (Prinzip P1).
SHARE_QUANT: Final = Decimal("0.0001")

#: Solverstatus, die cvxpy als brauchbar meldet.
ACCEPTABLE_STATUS: Final = frozenset({cp.OPTIMAL, cp.OPTIMAL_INACCURATE})

#: Höchstzahl der Iterationen des Rückfallverfahrens.
SLSQP_MAX_ITERATIONS: Final = 500


class OptimizationError(DomainError):
    """Die Optimierung hat kein zulässiges Portfolio geliefert."""


class SustainabilityQuotaError(DomainError):
    """Das empfohlene Portfolio unterschreitet einen verlangten Mindestanteil.

    Der Fehler schließt die Lücke zwischen erhobener Kundenvorgabe und
    tatsächlichem Ergebnis. Ohne diese Prüfung würde der Mindestanteil in der
    Geeignetheitserklärung zwar ausgewiesen, aber nie gegen das Portfolio
    geprüft – das Dokument behauptete eine eingehaltene Vorgabe, ohne sie zu
    belegen.

    Attributes:
        shortfalls: Ausprägung auf das Paar aus verlangtem und erreichtem Anteil.
        legal_basis: Rechtsgrundlage der Präferenz.
    """

    def __init__(self, shortfalls: dict[str, tuple[Decimal, Decimal]]) -> None:
        """Erzeugt den Fehler.

        Args:
            shortfalls: Ausprägung auf verlangten und erreichten Anteil.
        """
        self.shortfalls = shortfalls
        self.legal_basis = SUSTAINABILITY_LEGAL_BASIS
        befunde = ", ".join(
            f"{name}: verlangt {format_de.percent(verlangt)}, "
            f"erreicht {format_de.percent(erreicht)}"
            for name, (verlangt, erreicht) in sorted(shortfalls.items())
        )
        super().__init__(
            f"Das empfohlene Portfolio hält einen verlangten Mindestanteil nicht ein "
            f"({befunde}). Eine Empfehlung, die eine erhobene Nachhaltigkeitsvorgabe "
            f"unterschreitet, ergeht nicht. — Rechtsgrundlage: {SUSTAINABILITY_LEGAL_BASIS}"
        )


class SustainabilityUniverseError(DomainError):
    """Der Nachhaltigkeitsfilter lässt zu wenige Assetklassen übrig.

    Der Fehler ist die Umsetzung von Invariante I8. Er wird ausdrücklich
    geworfen, statt die Präferenz aufzuweichen oder das Universum aufzufüllen.

    Attributes:
        remaining: Verbliebene Assetklassen nach dem Filter.
        required: Geforderte Mindestgröße.
        excluded: Ausgeschlossene Assetklassen mit Begründung.
        legal_basis: Rechtsgrundlage der Präferenz.
    """

    def __init__(
        self,
        remaining: tuple[str, ...],
        required: int,
        excluded: dict[str, str],
    ) -> None:
        """Erzeugt den Fehler.

        Args:
            remaining: Verbliebene Assetklassen.
            required: Geforderte Mindestgröße.
            excluded: Ausgeschlossene Assetklassen mit Begründung.
        """
        self.remaining = remaining
        self.required = required
        self.excluded = excluded
        self.legal_basis = SUSTAINABILITY_LEGAL_BASIS
        super().__init__(
            f"Der Nachhaltigkeitsfilter lässt nur {len(remaining)} von "
            f"{len(remaining) + len(excluded)} Assetklassen übrig; gefordert sind "
            f"mindestens {required}. Verblieben: {', '.join(remaining) or '—'}. "
            "Eine Optimierung auf diesem Universum wäre nicht ausreichend gestreut. "
            "Die Nachhaltigkeitspräferenz wird nicht stillschweigend aufgeweicht — "
            f"Rechtsgrundlage: {SUSTAINABILITY_LEGAL_BASIS}."
        )


class SustainabilityConflictError(DomainError):
    """Nachhaltigkeitspräferenz und Risikoklasse sind unvereinbar.

    Der Fall entsteht, wenn der Filter eine Assetklasse entfernt, für die das
    Regelwerk der Risikoklasse eine **positive Untergrenze** vorsieht. Beide
    Anforderungen sind für sich rechtmäßig — die Streuungsvorgabe der Klasse und
    die Nachhaltigkeitspräferenz des Kunden —, zusammen aber nicht erfüllbar.

    Das System löst den Widerspruch **nicht** auf. Es weicht weder die Präferenz
    auf noch verletzt es das Band, sondern legt den Konflikt offen. Das ist
    dieselbe Haltung wie beim Zielkonflikt zwischen Anlageziel und zulässiger
    Risikoklasse: Eine Empfehlung, die eine der beiden Vorgaben stillschweigend
    übergeht, wäre schlechter als keine Empfehlung.

    Attributes:
        risk_class: Betroffene Risikoklasse.
        blocking: Assetklassen mit positiver Untergrenze, die der Filter
            entfernt hat, samt dieser Untergrenze.
        legal_basis: Rechtsgrundlage der Präferenz.
    """

    def __init__(self, risk_class: int, blocking: dict[str, float]) -> None:
        """Erzeugt den Fehler.

        Args:
            risk_class: Betroffene Risikoklasse.
            blocking: Assetklassen mit ihrer Untergrenze.
        """
        self.risk_class = risk_class
        self.blocking = blocking
        self.legal_basis = SUSTAINABILITY_LEGAL_BASIS
        aufzaehlung = ", ".join(
            f"{name} (mindestens {anteil:.0%})" for name, anteil in sorted(blocking.items())
        )
        super().__init__(
            f"Die Nachhaltigkeitspräferenz und Risikoklasse {risk_class} sind nicht "
            f"gleichzeitig erfüllbar: Das Regelwerk schreibt {aufzaehlung} vor, der "
            "Nachhaltigkeitsfilter schließt diese Position(en) jedoch aus. Weder die "
            "Präferenz noch die Streuungsvorgabe wird stillschweigend aufgeweicht. "
            "Mögliche Auswege: eine andere Ausprägung der Nachhaltigkeitspräferenz, "
            "ein erweitertes Anlageuniversum oder eine angepasste Bandbreite der Klasse. "
            f"Rechtsgrundlage der Präferenz: {SUSTAINABILITY_LEGAL_BASIS}."
        )


@dataclass(frozen=True, slots=True)
class UniverseFilter:
    """Ergebnis des Nachhaltigkeitsfilters.

    Attributes:
        included: Zugelassene Assetklassen in ursprünglicher Reihenfolge.
        excluded: Ausgeschlossene Assetklassen mit Begründung.
        applied: Ob überhaupt gefiltert wurde.
    """

    included: tuple[str, ...]
    excluded: dict[str, str]
    applied: bool

    @property
    def is_reduced(self) -> bool:
        """Gibt an, ob der Filter das Universum tatsächlich verkleinert hat."""
        return bool(self.excluded)


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Ergebnis der Optimierung.

    Attributes:
        weights: Gewichte je Assetklasse des **vollen** Universums; gefilterte
            Positionen tragen exakt null.
        expected_return: Erwartete Monatsrendite des Portfolios.
        volatility: Monatliche Volatilität des Portfolios.
        risk_class: Risikoklasse, deren Restriktionen galten.
        target_return: Tatsächlich verwendete Zielrendite.
        target_return_lowered: Ob die Zielrendite abgesenkt werden musste.
        solver: Name des Verfahrens, das die Lösung geliefert hat.
        fallback_used: Ob das Rückfallverfahren zum Einsatz kam.
        attempts: Solver mit ihrem gemeldeten Status, in Reihenfolge.
        universe: Ergebnis des Nachhaltigkeitsfilters.
        asset_keys: Assetklassen in Spaltenreihenfolge.
    """

    weights: FloatArray
    expected_return: float
    volatility: float
    risk_class: int
    target_return: float
    target_return_lowered: bool
    solver: str
    fallback_used: bool
    attempts: tuple[tuple[str, str], ...]
    universe: UniverseFilter
    asset_keys: tuple[str, ...]

    def as_summary(self) -> dict[str, object]:
        """Verdichtet das Ergebnis für Trace und Bericht.

        Returns:
            Zuordnung der wesentlichen Kennzahlen.
        """
        return {
            "risikoklasse": self.risk_class,
            "erwartete_monatsrendite": round(self.expected_return, 8),
            "monatsvolatilitaet": round(self.volatility, 8),
            "zielrendite": round(self.target_return, 8),
            "zielrendite_abgesenkt": self.target_return_lowered,
            "solver": self.solver,
            "rueckfallverfahren": self.fallback_used,
            "versuche": [f"{name}={status}" for name, status in self.attempts],
            "universum_gefiltert": self.universe.is_reduced,
            "ausgeschlossen": sorted(self.universe.excluded),
        }


def filter_universe(
    preferences: SustainabilityPreferences,
    policy: PortfolioPolicy,
    asset_keys: tuple[str, ...],
) -> UniverseFilter:
    """Reduziert das Anlageuniversum nach den Nachhaltigkeitspräferenzen.

    Eine Assetklasse bleibt nur, wenn sie **jede** vom Kunden verlangte
    Ausprägung erfüllt. Die Ausprägungen sind kumulativ zu lesen: Wer
    Taxonomiekonformität *und* Berücksichtigung der nachteiligen Auswirkungen
    verlangt, bekommt nicht die Vereinigung, sondern den Schnitt.

    Args:
        preferences: Nachhaltigkeitspräferenzen des Kunden.
        policy: Regelwerk der Portfoliokonstruktion.
        asset_keys: Assetklassen in Spaltenreihenfolge.

    Returns:
        Das Filterergebnis.

    Raises:
        SustainabilityUniverseError: Wenn das gefilterte Universum die
            Mindestgröße unterschreitet (Invariante I8).
    """
    if not preferences.has_preferences:
        return UniverseFilter(included=asset_keys, excluded={}, applied=False)

    verlangt = {
        "taxonomie": preferences.taxonomy_aligned.requested,
        "sfdr": preferences.sfdr_sustainable.requested,
        "pai": preferences.pai_considered.requested,
    }
    einstufung = policy.nachhaltigkeit.einstufung

    zugelassen: list[str] = []
    ausgeschlossen: dict[str, str] = {}
    for name in asset_keys:
        bewertung = einstufung[name]
        erfuellt = {
            "taxonomie": bewertung.taxonomie,
            "sfdr": bewertung.sfdr,
            "pai": bewertung.pai,
        }
        fehlend = [
            merkmal
            for merkmal, gefordert in verlangt.items()
            if gefordert and not erfuellt[merkmal]
        ]
        if fehlend:
            ausgeschlossen[name] = (
                f"erfüllt nicht: {', '.join(fehlend)} — {bewertung.hinweis.strip()}"
            )
        else:
            zugelassen.append(name)

    mindestgroesse = policy.nachhaltigkeit.mindestuniversum
    if len(zugelassen) < mindestgroesse:
        raise SustainabilityUniverseError(tuple(zugelassen), mindestgroesse, ausgeschlossen)

    return UniverseFilter(included=tuple(zugelassen), excluded=ausgeschlossen, applied=True)


def achieved_shares(
    weights: FloatArray,
    preferences: SustainabilityPreferences,
    policy: PortfolioPolicy,
    asset_keys: tuple[str, ...],
) -> dict[str, Decimal]:
    """Ermittelt je verlangter Ausprägung den erreichten Anteil am Portfolio.

    Gegenstück zu :func:`filter_universe`: Dieses prüft, was **vor** der
    Optimierung zugelassen wurde, jenes, was **danach** tatsächlich im Portfolio
    steht. Der Vergleich beider Größen ist der Beleg, den die
    Geeignetheitserklärung ohne diese Messung schuldig bliebe.

    Der harte Ausschluss in :func:`filter_universe` führt beim gegenwärtigen
    Stand stets auf einen Anteil von eins. Die Funktion misst dennoch, statt den
    Wert zu setzen: Eine spätere Fassung, die den Ausschluss durch eine Quote
    ersetzt, wäre sonst still falsch.

    Args:
        weights: Gewichte je Assetklasse des vollen Universums.
        preferences: Nachhaltigkeitspräferenzen des Kunden.
        policy: Regelwerk der Portfoliokonstruktion.
        asset_keys: Assetklassen in Spaltenreihenfolge.

    Returns:
        Zuordnung der verlangten Ausprägungen auf ihren erreichten Anteil.
        Leer, wenn keine Ausprägung verlangt wird.
    """
    verlangt = {
        "taxonomie": preferences.taxonomy_aligned.requested,
        "sfdr": preferences.sfdr_sustainable.requested,
        "pai": preferences.pai_considered.requested,
    }
    einstufung = policy.nachhaltigkeit.einstufung
    gesamt = float(weights.sum())

    anteile: dict[str, Decimal] = {}
    for merkmal, gefordert in verlangt.items():
        if not gefordert:
            continue
        summe = sum(
            float(weights[i])
            for i, name in enumerate(asset_keys)
            if getattr(einstufung[name], merkmal)
        )
        roh = summe / gesamt if gesamt > 0.0 else 0.0
        gerundet = Decimal(str(round(roh, 8)))
        anteile[merkmal] = gerundet.quantize(SHARE_QUANT, rounding=ROUND_HALF_UP)
    return anteile


def _bounds_for_class(
    definition: RiskClassDefinition,
    asset_keys: tuple[str, ...],
    allowed: frozenset[str],
) -> tuple[FloatArray, FloatArray]:
    """Bildet Unter- und Obergrenzen je Assetklasse.

    Gefilterte Positionen erhalten das Band ``[0, 0]``. Damit bleibt die
    Dimension des Problems unverändert, und die Zuordnung Gewicht-zu-Assetklasse
    ist über alle Läufe hinweg dieselbe — das erleichtert Vergleich und Trace.

    Args:
        definition: Definition der Risikoklasse.
        asset_keys: Assetklassen in Spaltenreihenfolge.
        allowed: Nach dem Filter zugelassene Assetklassen.

    Returns:
        Untergrenzen und Obergrenzen.
    """
    untergrenzen = np.zeros(len(asset_keys))
    obergrenzen = np.zeros(len(asset_keys))
    deckel = float(definition.positionsobergrenze)
    for i, name in enumerate(asset_keys):
        if name not in allowed:
            continue
        band = definition.einzelwerte[name]
        untergrenzen[i] = float(band.min)
        obergrenzen[i] = min(float(band.max), deckel)
    return untergrenzen, obergrenzen


def _group_matrix(
    classes: ClassesPolicy, asset_keys: tuple[str, ...]
) -> tuple[tuple[str, ...], FloatArray]:
    """Bildet die Zuordnungsmatrix von Assetklassen auf Assetgruppen.

    Args:
        classes: Regelwerk der Risikoklassen.
        asset_keys: Assetklassen in Spaltenreihenfolge.

    Returns:
        Die Gruppennamen und die Matrix der Form ``(gruppen, assetklassen)``.
    """
    gruppen = tuple(classes.gruppenzuordnung)
    matrix = np.zeros((len(gruppen), len(asset_keys)))
    for zeile, gruppe in enumerate(gruppen):
        for name in classes.gruppenzuordnung[gruppe]:
            if name in asset_keys:
                matrix[zeile, asset_keys.index(name)] = 1.0
    return gruppen, matrix


def _feasible_bounds(untergrenzen: FloatArray, obergrenzen: FloatArray) -> None:
    """Prüft, ob die Einzelbänder überhaupt ein Portfolio zulassen.

    Args:
        untergrenzen: Untergrenzen je Assetklasse.
        obergrenzen: Obergrenzen je Assetklasse.

    Raises:
        OptimizationError: Wenn Vollinvestition unmöglich ist.
    """
    if float(untergrenzen.sum()) > 1.0 + 1e-9:
        msg = (
            f"Die Untergrenzen summieren sich auf {untergrenzen.sum():.4f}; eine "
            "Gewichtssumme von eins ist nicht erreichbar. Der Nachhaltigkeitsfilter "
            "hat vermutlich Positionen entfernt, deren Untergrenze im Regelwerk "
            "positiv ist."
        )
        raise OptimizationError(msg)
    if float(obergrenzen.sum()) < 1.0 - 1e-9:
        msg = (
            f"Die Obergrenzen summieren sich auf {obergrenzen.sum():.4f}; eine "
            "Gewichtssumme von eins ist nicht erreichbar. Das verbliebene Universum "
            "ist für diese Risikoklasse zu klein."
        )
        raise OptimizationError(msg)


def _solve_cvxpy(  # noqa: PLR0913, PLR0917 — jede Groesse ist eine eigene Nebenbedingung
    moments: MomentEstimate,
    untergrenzen: FloatArray,
    obergrenzen: FloatArray,
    gruppenmatrix: FloatArray,
    gruppen_min: FloatArray,
    gruppen_max: FloatArray,
    target_return: float,
    solver_name: str,
) -> tuple[FloatArray | None, str]:
    """Löst das quadratische Programm mit einem cvxpy-Solver.

    Args:
        moments: Geschätzte Momente.
        untergrenzen: Untergrenzen je Assetklasse.
        obergrenzen: Obergrenzen je Assetklasse.
        gruppenmatrix: Zuordnung Assetklassen auf Gruppen.
        gruppen_min: Untergrenzen je Gruppe.
        gruppen_max: Obergrenzen je Gruppe.
        target_return: Geforderte Mindestrendite.
        solver_name: Name des Solvers.

    Returns:
        Die Gewichte und der gemeldete Status; die Gewichte sind ``None``,
        wenn der Status nicht brauchbar ist.
    """
    gewichte = cp.Variable(len(untergrenzen))
    # psd_wrap: Die geschrumpfte Kovarianz ist mathematisch positiv definit,
    # kann aber durch Rundung minimal asymmetrisch sein. Ohne den Hinweis
    # verweigert cvxpy die Annahme des Problems als konvex.
    kovarianz = cp.psd_wrap(moments.covariance)
    ziel = cp.Minimize(cp.quad_form(gewichte, kovarianz))
    nebenbedingungen = [
        cp.sum(gewichte) == 1,
        gewichte >= untergrenzen,
        gewichte <= obergrenzen,
        moments.mean @ gewichte >= target_return,
        gruppenmatrix @ gewichte >= gruppen_min,
        gruppenmatrix @ gewichte <= gruppen_max,
    ]
    problem = cp.Problem(ziel, nebenbedingungen)
    try:
        problem.solve(solver=solver_name)
    except (cp.error.SolverError, cp.error.DCPError) as fehler:
        return None, f"Ausnahme: {type(fehler).__name__}"

    status = str(problem.status)
    if status not in ACCEPTABLE_STATUS or gewichte.value is None:
        return None, status
    return np.asarray(gewichte.value, dtype=np.float64), status


def _solve_slsqp(  # noqa: PLR0913, PLR0917 — spiegelbildlich zu _solve_cvxpy
    moments: MomentEstimate,
    untergrenzen: FloatArray,
    obergrenzen: FloatArray,
    gruppenmatrix: FloatArray,
    gruppen_min: FloatArray,
    gruppen_max: FloatArray,
    target_return: float,
) -> tuple[FloatArray | None, str]:
    """Löst dasselbe Problem mit scipy SLSQP als Rückfallverfahren.

    SLSQP ist ein allgemeines Verfahren für nichtlineare Optimierung und nutzt
    die Konvexität des Problems nicht aus. Es ist deshalb langsamer und findet
    nur ein lokales Optimum — das bei konvexem Problem zwar global ist, aber
    nur bis auf die Genauigkeit des Verfahrens erreicht wird. Als
    Rückfallverfahren ist es gerechtfertigt, als Standardweg nicht.

    Args:
        moments: Geschätzte Momente.
        untergrenzen: Untergrenzen je Assetklasse.
        obergrenzen: Obergrenzen je Assetklasse.
        gruppenmatrix: Zuordnung Assetklassen auf Gruppen.
        gruppen_min: Untergrenzen je Gruppe.
        gruppen_max: Obergrenzen je Gruppe.
        target_return: Geforderte Mindestrendite.

    Returns:
        Die Gewichte und der gemeldete Status.
    """

    def varianz(gewichte: FloatArray) -> float:
        """Portfoliovarianz als Zielfunktion.

        Args:
            gewichte: Gewichte je Assetklasse.

        Returns:
            Die Varianz.
        """
        return float(gewichte @ moments.covariance @ gewichte)

    nebenbedingungen = [
        {"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
        {"type": "ineq", "fun": lambda w: float(w @ moments.mean - target_return)},
        {"type": "ineq", "fun": lambda w: gruppenmatrix @ w - gruppen_min},
        {"type": "ineq", "fun": lambda w: gruppen_max - gruppenmatrix @ w},
    ]
    start = untergrenzen + (1.0 - untergrenzen.sum()) * _spielraum_anteil(untergrenzen, obergrenzen)
    ergebnis = minimize(
        varianz,
        start,
        method="SLSQP",
        bounds=list(zip(untergrenzen, obergrenzen, strict=True)),
        constraints=nebenbedingungen,
        options={"maxiter": SLSQP_MAX_ITERATIONS, "ftol": 1e-12},
    )
    if not ergebnis.success:
        return None, f"SLSQP: {ergebnis.message}"
    return np.asarray(ergebnis.x, dtype=np.float64), "SLSQP: optimal"


def _spielraum_anteil(untergrenzen: FloatArray, obergrenzen: FloatArray) -> FloatArray:
    """Verteilt den Rest proportional zum Spielraum ``max − min``.

    Dient allein als Startpunkt des Rückfallverfahrens; die Wahl entspricht der
    Referenzallokation aus :mod:`candor.portfolio.universe`.

    Args:
        untergrenzen: Untergrenzen je Assetklasse.
        obergrenzen: Obergrenzen je Assetklasse.

    Returns:
        Anteile mit Summe eins, oder Gleichverteilung ohne Spielraum.
    """
    spielraum = obergrenzen - untergrenzen
    gesamt = float(spielraum.sum())
    if gesamt <= 0.0:
        return np.full(len(untergrenzen), 1.0 / len(untergrenzen))
    return np.asarray(spielraum / gesamt, dtype=np.float64)


def check_constraints(
    weights: FloatArray,
    definition: RiskClassDefinition,
    classes: ClassesPolicy,
    asset_keys: tuple[str, ...],
    tolerance: float,
) -> tuple[str, ...]:
    """Prüft ein Ergebnis gegen alle Restriktionen (Invariante I5).

    Die Prüfung wiederholt bewusst, was der Solver zugesichert hat. Ein Solver
    kann einen Status ``optimal`` melden und dennoch Restriktionen leicht
    verletzen; bei ``OPTIMAL_INACCURATE`` ist das sogar zu erwarten.

    Args:
        weights: Gewichte je Assetklasse.
        definition: Definition der Risikoklasse.
        classes: Regelwerk der Risikoklassen.
        asset_keys: Assetklassen in Spaltenreihenfolge.
        tolerance: Zulässige Überschreitung.

    Returns:
        Die Verstöße im Klartext; leer bei einem zulässigen Portfolio.
    """
    verstoesse: list[str] = []
    summe = float(weights.sum())
    if abs(summe - 1.0) > tolerance:
        verstoesse.append(f"Gewichtssumme {format_de.decimal(summe, 6)} weicht von eins ab.")
    if float(weights.min()) < -tolerance:
        verstoesse.append(
            f"Negatives Gewicht {format_de.decimal(float(weights.min()), 6)} (Leerverkauf)."
        )

    deckel = float(definition.positionsobergrenze)
    for i, name in enumerate(asset_keys):
        gewicht = float(weights[i])
        band = definition.einzelwerte[name]
        if gewicht > deckel + tolerance:
            verstoesse.append(
                f"{name}: {format_de.percent(gewicht)} über der Positionsobergrenze "
                f"{format_de.percent(deckel)}."
            )
        if gewicht < float(band.min) - tolerance:
            verstoesse.append(
                f"{name}: {format_de.percent(gewicht)} unter der Untergrenze "
                f"{format_de.percent(float(band.min))}."
            )
        if gewicht > float(band.max) + tolerance:
            verstoesse.append(
                f"{name}: {format_de.percent(gewicht)} über der Obergrenze "
                f"{format_de.percent(float(band.max))}."
            )

    for gruppe, mitglieder in classes.gruppenzuordnung.items():
        anteil = float(
            sum(weights[asset_keys.index(name)] for name in mitglieder if name in asset_keys)
        )
        band = definition.gruppen[gruppe]
        if anteil < float(band.min) - tolerance:
            verstoesse.append(
                f"Gruppe {gruppe}: {format_de.percent(anteil)} unter "
                f"{format_de.percent(float(band.min))}."
            )
        if anteil > float(band.max) + tolerance:
            verstoesse.append(
                f"Gruppe {gruppe}: {format_de.percent(anteil)} über "
                f"{format_de.percent(float(band.max))}."
            )

    return tuple(verstoesse)


def optimize(  # noqa: PLR0913 — jede Angabe ist eine eigene fachliche Groesse
    moments: MomentEstimate,
    risk_class: int,
    classes: ClassesPolicy,
    policy: PortfolioPolicy,
    asset_keys: tuple[str, ...],
    *,
    target_return: float,
    universe: UniverseFilter | None = None,
) -> OptimizationResult:
    """Löst die restringierte Mean-Variance-Optimierung für eine Risikoklasse.

    Args:
        moments: Geschätzte Momente.
        risk_class: Risikoklasse von 1 bis 5.
        classes: Regelwerk der Risikoklassen.
        policy: Regelwerk der Portfoliokonstruktion.
        asset_keys: Assetklassen in Spaltenreihenfolge.
        target_return: Geforderte monatliche Mindestrendite.
        universe: Ergebnis des Nachhaltigkeitsfilters. Ohne Angabe gilt das
            volle Universum.

    Returns:
        Das Optimierungsergebnis.

    Raises:
        OptimizationError: Wenn die Risikoklasse unbekannt ist, die
            Restriktionen kein Portfolio zulassen oder kein Verfahren eine
            zulässige Lösung liefert.
    """
    definition = classes.risikoklassen.get(risk_class)
    if definition is None:
        msg = f"Unbekannte Risikoklasse {risk_class}. Bekannt: {sorted(classes.risikoklassen)}."
        raise OptimizationError(msg)

    gefiltert = universe if universe is not None else UniverseFilter(asset_keys, {}, applied=False)
    zugelassen = frozenset(gefiltert.included)

    # Der Konflikt zwischen Filter und Pflichtuntergrenze wird VOR dem Solver
    # erkannt. Andernfalls käme er als „Solver meldet Erfolg, Ergebnis verletzt
    # die Restriktionen" heraus — eine Meldung, die den Solver beschuldigt und
    # die eigentliche, fachliche Ursache verdeckt.
    blockierend = {
        name: float(definition.einzelwerte[name].min)
        for name in gefiltert.excluded
        if definition.einzelwerte[name].min > 0
    }
    if blockierend:
        raise SustainabilityConflictError(risk_class, blockierend)

    untergrenzen, obergrenzen = _bounds_for_class(definition, asset_keys, zugelassen)
    _feasible_bounds(untergrenzen, obergrenzen)

    gruppen, gruppenmatrix = _group_matrix(classes, asset_keys)
    gruppen_min = np.array([float(definition.gruppen[name].min) for name in gruppen])
    gruppen_max = np.array([float(definition.gruppen[name].max) for name in gruppen])

    einstellungen = policy.optimierung
    schritt = einstellungen.zielrendite_absenkung_schritt
    aktuelles_ziel = target_return
    versuche: list[tuple[str, str]] = []

    for absenkung in range(einstellungen.zielrendite_absenkung_max_schritte + 1):
        aktuelles_ziel = target_return - absenkung * schritt
        for solver_name in einstellungen.solver_reihenfolge:
            gewichte, status = _solve_cvxpy(
                moments,
                untergrenzen,
                obergrenzen,
                gruppenmatrix,
                gruppen_min,
                gruppen_max,
                aktuelles_ziel,
                solver_name,
            )
            versuche.append((f"{solver_name}@{absenkung}", status))
            if gewichte is not None:
                return _ergebnis(
                    gewichte,
                    moments,
                    definition,
                    classes,
                    policy,
                    asset_keys,
                    risk_class,
                    aktuelles_ziel,
                    lowered=absenkung > 0,
                    solver=solver_name,
                    fallback=False,
                    versuche=versuche,
                    universe=gefiltert,
                )

        gewichte, status = _solve_slsqp(
            moments,
            untergrenzen,
            obergrenzen,
            gruppenmatrix,
            gruppen_min,
            gruppen_max,
            aktuelles_ziel,
        )
        versuche.append((f"{einstellungen.fallback}@{absenkung}", status))
        if gewichte is not None:
            return _ergebnis(
                gewichte,
                moments,
                definition,
                classes,
                policy,
                asset_keys,
                risk_class,
                aktuelles_ziel,
                lowered=absenkung > 0,
                solver=einstellungen.fallback,
                fallback=True,
                versuche=versuche,
                universe=gefiltert,
            )

    protokoll = "; ".join(f"{name}: {status}" for name, status in versuche[-6:])
    msg = (
        f"Kein Verfahren hat für Risikoklasse {risk_class} ein zulässiges Portfolio "
        f"geliefert. Die Zielrendite wurde in {einstellungen.zielrendite_absenkung_max_schritte} "
        f"Schritten von {target_return:.6f} auf {aktuelles_ziel:.6f} gesenkt, ohne dass das "
        f"Problem lösbar wurde. Letzte Versuche: {protokoll}."
    )
    raise OptimizationError(msg)


def _ergebnis(  # noqa: PLR0913, PLR0917 — Bündelung der Ergebnisgroessen
    weights: FloatArray,
    moments: MomentEstimate,
    definition: RiskClassDefinition,
    classes: ClassesPolicy,
    policy: PortfolioPolicy,
    asset_keys: tuple[str, ...],
    risk_class: int,
    target_return: float,
    *,
    lowered: bool,
    solver: str,
    fallback: bool,
    versuche: list[tuple[str, str]],
    universe: UniverseFilter,
) -> OptimizationResult:
    """Bereinigt die Gewichte, prüft sie und baut das Ergebnisobjekt.

    Args:
        weights: Rohgewichte des Solvers.
        moments: Geschätzte Momente.
        definition: Definition der Risikoklasse.
        classes: Regelwerk der Risikoklassen.
        policy: Regelwerk der Portfoliokonstruktion.
        asset_keys: Assetklassen in Spaltenreihenfolge.
        risk_class: Risikoklasse.
        target_return: Verwendete Zielrendite.
        lowered: Ob die Zielrendite abgesenkt wurde.
        solver: Name des Verfahrens.
        fallback: Ob das Rückfallverfahren zum Einsatz kam.
        versuche: Protokoll der Solverversuche.
        universe: Ergebnis des Nachhaltigkeitsfilters.

    Returns:
        Das geprüfte Ergebnis.

    Raises:
        OptimizationError: Wenn die Lösung Restriktionen verletzt.
    """
    toleranz = policy.optimierung.restriktionstoleranz
    # Numerisches Rauschen um null abschneiden, dann normieren. Ein Gewicht von
    # -1e-17 ist kein Leerverkauf, sondern Gleitkommaarithmetik.
    bereinigt = np.clip(weights, 0.0, None)
    # Ausgefilterte Positionen werden **exakt** genullt, nicht bloß auf einen
    # verschwindend kleinen Wert gedrückt. Der Solver hinterlässt dort Reste in
    # der Größenordnung 1e-15; die wären zwar wirtschaftlich bedeutungslos, aber
    # der Ausschluss ist eine Zusage an den Kunden und keine Näherung. Wer im
    # Ergebnis nach einer ausgeschlossenen Assetklasse sucht, muss null finden.
    for name in universe.excluded:
        bereinigt[asset_keys.index(name)] = 0.0
    summe = float(bereinigt.sum())
    if summe <= 0.0:
        msg = "Die Lösung besteht ausschließlich aus Nullgewichten."
        raise OptimizationError(msg)
    bereinigt = bereinigt / summe

    verstoesse = check_constraints(bereinigt, definition, classes, asset_keys, toleranz)
    if verstoesse:
        msg = (
            f"Das Verfahren {solver} meldet Erfolg, das Ergebnis verletzt aber die "
            f"Restriktionen der Klasse {risk_class}: {' '.join(verstoesse)}"
        )
        raise OptimizationError(msg)

    return OptimizationResult(
        weights=bereinigt,
        expected_return=moments.portfolio_return(bereinigt),
        volatility=moments.portfolio_volatility(bereinigt),
        risk_class=risk_class,
        target_return=target_return,
        target_return_lowered=lowered,
        solver=solver,
        fallback_used=fallback,
        attempts=tuple(versuche),
        universe=universe,
        asset_keys=asset_keys,
    )


def minimum_variance(
    moments: MomentEstimate,
    risk_class: int,
    classes: ClassesPolicy,
    policy: PortfolioPolicy,
    asset_keys: tuple[str, ...],
) -> OptimizationResult:
    """Löst das Minimum-Varianz-Portfolio ohne bindende Zielrendite.

    Der linke Randpunkt der Effizienzlinie. Er kommt ohne die Schätzung
    erwarteter Renditen aus und ist deshalb der robusteste Punkt der Kurve.

    Args:
        moments: Geschätzte Momente.
        risk_class: Risikoklasse von 1 bis 5.
        classes: Regelwerk der Risikoklassen.
        policy: Regelwerk der Portfoliokonstruktion.
        asset_keys: Assetklassen in Spaltenreihenfolge.

    Returns:
        Das Minimum-Varianz-Portfolio der Klasse.
    """
    return optimize(
        moments,
        risk_class,
        classes,
        policy,
        asset_keys,
        target_return=float(moments.mean.min()) - 1.0,
    )
