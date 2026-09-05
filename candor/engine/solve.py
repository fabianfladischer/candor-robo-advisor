"""Die drei Stellschrauben des Konfliktberichts.

Wird eine Empfehlung verweigert, tritt an ihre Stelle die Frage: **Was müsste
sich ändern?** Drei Größen kommen in Betracht, und jede wird numerisch auf die
geforderte Erfolgswahrscheinlichkeit gelöst:

* **S1 — Sparrate.** Welche monatliche Rate wäre nötig, wenn Ziel und Termin
  bleiben?
* **S2 — Horizont.** Um wie viele Monate müsste der Termin verschoben werden,
  wenn die Rate bleibt?
* **S3 — Zielbetrag.** Welcher Betrag ist erreichbar, wenn Rate und Termin
  bleiben?

Bisektion auf einer Monte-Carlo-Zielfunktion
--------------------------------------------
Die Erfolgswahrscheinlichkeit ist eine Schätzung aus 10 000 Pfaden. Würde man
sie in jeder Iteration mit einem neuen Seed berechnen, schwankte sie um rund
einen halben Prozentpunkt — **die Bisektion konvergierte nicht**, sondern liefe
gegen das Rauschen. Sie schlösse ein Intervall ein, in dem sich das Vorzeichen
bei jeder erneuten Auswertung ändern kann.

Deshalb ist der Seed über **alle** Iterationen und über **alle drei
Stellschrauben** fixiert; er ist Bestandteil von :class:`SolverContext` und
nicht Parameter der einzelnen Suche. Die Zielfunktion ist damit deterministisch:
dieselbe Eingabe liefert exakt dieselbe Wahrscheinlichkeit. Sie ist eine
Treppenfunktion mit bis zu 10 000 Stufen, aber monoton, und die Bisektion
erreicht die geforderte Toleranz in wenigen Dutzend Schritten.

Der Preis: Das Ergebnis hängt am gewählten Seed. Bei 10 000 Pfaden entspricht
das einer Unsicherheit in der Größenordnung des Standardfehlers — derselben,
die auch die Ausgangsdiagnose trägt. Die Bisektion erzeugt sie nicht, sie reicht
sie weiter.

Verfahren je Stellschraube
--------------------------
Nicht alle drei brauchen dasselbe Verfahren, und es wäre unredlich, so zu tun:

* **S1** verändert die Zahlungsströme und damit jeden Pfad. Hier ist die
  Bisektion tatsächlich nötig; je Auswertung läuft eine vollständige Simulation.
* **S2** verändert nur, **wann** abgelesen wird. Eine einzige Simulation bis zum
  längsten geprüften Horizont genügt; die Bisektion läuft anschließend über den
  Monatsindex derselben Wertmatrix. Das ist erheblich schneller und zugleich
  konsistenter, weil alle geprüften Horizonte auf denselben Pfaden beruhen.
* **S3** verändert die Simulation überhaupt nicht — der Zielbetrag ist nur die
  Schwelle, gegen die gezählt wird. Der gesuchte Wert ist damit **exakt** das
  empirische Quantil der Endwertverteilung. Eine Bisektion darauf wäre langsamer
  und ungenauer als die geschlossene Antwort. Sie wird trotzdem bereitgestellt
  (:func:`solve_target_by_bisection`) und in den Tests gegen das Quantil
  geprüft.

Nichtkonvergenz
---------------
Erreicht die Bisektion die Toleranz nicht innerhalb der Iterationsobergrenze,
wird ein :class:`SolverError` ausgelöst. Es gibt keinen stillen Rückfall auf den
letzten Zwischenwert: Eine Zahl, von der niemand weiß, ob sie konvergiert ist,
wäre in einem Konfliktbericht schlimmer als gar keine.

Davon zu unterscheiden ist der Fall, dass eine Stellschraube die geforderte
Wahrscheinlichkeit **auch am Rand des Suchbereichs nicht erreicht**. Das ist
kein Fehler, sondern ein Befund: Die Stellschraube löst den Konflikt nicht. Sie
wird mit :attr:`~candor.domain.result.Feasibility.NICHT_ERREICHBAR` ausgewiesen.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Final

import numpy as np

from candor import format_de
from candor.domain.client import Client
from candor.domain.goal import Goal
from candor.domain.policy import PolicySet
from candor.domain.result import Feasibility, Lever, LeverId
from candor.engine.simulate import FloatArray, ReturnSampler, simulate
from candor.goals.model import template_for

#: Toleranz der Bisektion auf der Sparrate, in Euro je Monat.
CONTRIBUTION_TOLERANCE: Final = 1.0

#: Toleranz der Bisektion auf dem Zielbetrag, in Euro.
TARGET_TOLERANCE: Final = 1.0

#: Obergrenze der Suche nach der erforderlichen Sparrate, in Euro je Monat.
MAX_CONTRIBUTION: Final = 100_000.0

#: Obergrenze der Suche nach dem erforderlichen Horizont, in Monaten.
MAX_HORIZON_MONTHS: Final = 720

#: Iterationsobergrenze jeder Bisektion.
MAX_ITERATIONS: Final = 60

#: Genauigkeit der ausgewiesenen Geldbeträge.
EURO_QUANT: Final = Decimal("0.01")

#: Monate je Jahr.
MONTHS_PER_YEAR: Final = 12

#: Deutsche Anführungszeichen, als Unicode-Escape geschrieben. Als Literal
#: gesetzt sind sie im Editor nicht vom ASCII-Anführungszeichen zu unterscheiden
#: und beenden dann versehentlich die Zeichenkette. tests/test_bericht.py fängt
#: diesen Fehler ab.
QUOTE_OPEN: Final = "\u201e"
QUOTE_CLOSE: Final = "\u201c"


class SolverError(RuntimeError):
    """Ein Suchverfahren hat die geforderte Genauigkeit nicht erreicht."""


@dataclass(frozen=True, slots=True)
class SolverContext:
    """Unveränderlicher Rahmen aller drei Stellschraubensuchen.

    Der **Seed gehört hierher und nicht in die einzelne Suche**: Alle drei
    Stellschrauben und alle Iterationen darin arbeiten auf derselben Pfadschar.
    Nur so sind die drei Zahlen eines Konfliktberichts untereinander vergleichbar
    und beim erneuten Lauf reproduzierbar.

    Attributes:
        sampler: Ziehungsverfahren der Monatsrenditen.
        weights: Zielgewichte je Assetklasse.
        policies: Vollständiges Regelwerk.
        paths: Zahl der Pfade je Auswertung.
        seed: Über alles fixierter Seed.
        asset_keys: Assetklassen in Spaltenreihenfolge.
    """

    sampler: ReturnSampler
    weights: FloatArray
    policies: PolicySet
    paths: int
    seed: int
    asset_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BisectionResult:
    """Ergebnis einer Suche.

    Attributes:
        value: Gefundener Wert.
        probability: Erfolgswahrscheinlichkeit an diesem Wert.
        iterations: Zahl der Auswertungen der Zielfunktion.
        reachable: Ob die geforderte Wahrscheinlichkeit im Suchbereich
            überhaupt erreicht wird.
    """

    value: float
    probability: float
    iterations: int
    reachable: bool


def bisect_increasing(  # noqa: PLR0913 — Intervall, Schwelle und Abbruch sind je eigene Groessen
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    target: float,
    *,
    tolerance: float,
    max_iterations: int = MAX_ITERATIONS,
) -> BisectionResult:
    """Sucht den kleinsten Wert, an dem eine steigende Funktion die Schwelle erreicht.

    Args:
        objective: Zielfunktion, monoton steigend und **deterministisch**.
        lower: Untere Intervallgrenze.
        upper: Obere Intervallgrenze.
        target: Zu erreichende Schwelle.
        tolerance: Zulässige Restbreite des Intervalls.
        max_iterations: Iterationsobergrenze.

    Returns:
        Das Ergebnis der Suche. Erreicht die Funktion die Schwelle auch am oberen
        Rand nicht, ist ``reachable`` falsch und ``value`` der obere Rand.

    Raises:
        SolverError: Wenn die Toleranz innerhalb der Iterationsobergrenze nicht
            erreicht wird oder das Intervall unzulässig ist.
    """
    if upper <= lower:
        msg = f"Unzulässiges Suchintervall [{lower}, {upper}]."
        raise SolverError(msg)

    oberer_wert = objective(upper)
    if oberer_wert < target:
        return BisectionResult(upper, oberer_wert, 1, reachable=False)

    unterer_wert = objective(lower)
    if unterer_wert >= target:
        return BisectionResult(lower, unterer_wert, 2, reachable=True)

    links, rechts = lower, upper
    auswertungen = 2
    for _ in range(max_iterations):
        mitte = (links + rechts) / 2.0
        auswertungen += 1
        if objective(mitte) >= target:
            rechts = mitte
        else:
            links = mitte
        if rechts - links <= tolerance:
            return BisectionResult(rechts, objective(rechts), auswertungen + 1, reachable=True)

    msg = (
        f"Die Bisektion hat nach {max_iterations} Iterationen die Toleranz {tolerance} "
        f"nicht erreicht; das Restintervall ist [{links}, {rechts}]. Es wird bewusst "
        "kein Zwischenwert zurückgegeben."
    )
    raise SolverError(msg)


def bisect_decreasing(  # noqa: PLR0913 — spiegelbildlich zu bisect_increasing
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    target: float,
    *,
    tolerance: float,
    max_iterations: int = MAX_ITERATIONS,
) -> BisectionResult:
    """Sucht den größten Wert, an dem eine fallende Funktion die Schwelle noch hält.

    Umgesetzt als Spiegelung auf :func:`bisect_increasing`, damit es nur eine
    Stelle mit Abbruch- und Fehlerlogik gibt.

    Args:
        objective: Zielfunktion, monoton fallend und **deterministisch**.
        lower: Untere Intervallgrenze.
        upper: Obere Intervallgrenze.
        target: Zu haltende Schwelle.
        tolerance: Zulässige Restbreite des Intervalls.
        max_iterations: Iterationsobergrenze.

    Returns:
        Das Ergebnis der Suche.

    Raises:
        SolverError: Wenn die Toleranz nicht erreicht wird oder das Intervall
            unzulässig ist.
    """

    def gespiegelt(wert: float) -> float:
        """Wertet die fallende Funktion an der gespiegelten Stelle aus.

        Args:
            wert: Gespiegelte Eingabe.

        Returns:
            Der Funktionswert.
        """
        return objective(lower + upper - wert)

    ergebnis = bisect_increasing(
        gespiegelt, lower, upper, target, tolerance=tolerance, max_iterations=max_iterations
    )
    return BisectionResult(
        value=lower + upper - ergebnis.value,
        probability=ergebnis.probability,
        iterations=ergebnis.iterations,
        reachable=ergebnis.reachable,
    )


def _probability(final_values: FloatArray, target: float) -> float:
    """Anteil der Pfade, die einen Zielbetrag erreichen.

    Args:
        final_values: Endwerte je Pfad.
        target: Zielbetrag.

    Returns:
        Die Erfolgswahrscheinlichkeit.
    """
    return float(np.mean(final_values >= target))


def final_values(
    context: SolverContext, goal: Goal, contribution: float | None = None
) -> FloatArray:
    """Rechnet den Fall durch und liefert die Endwerte je Pfad.

    Args:
        context: Rahmen der Suche, insbesondere der fixierte Seed.
        goal: Anlageziel.
        contribution: Abweichende Sparrate. Ohne Angabe die des Ziels.

    Returns:
        Die Endwerte je Pfad.
    """
    ergebnis = simulate(
        context.sampler,
        context.weights,
        initial_capital=float(goal.initial_capital),
        monthly_contribution=(
            float(goal.monthly_contribution) if contribution is None else contribution
        ),
        horizon_months=goal.horizon_months,
        target_amount=float(goal.target_amount),
        policy=context.policies.simulation,
        paths=context.paths,
        seed=context.seed,
        asset_keys=context.asset_keys,
        track_bands=False,
    )
    return ergebnis.final_values


def solve_contribution(
    context: SolverContext, goal: Goal, required_probability: float
) -> BisectionResult:
    """Löst S1: erforderliche monatliche Sparrate bei unverändertem Ziel und Horizont.

    Args:
        context: Rahmen der Suche.
        goal: Anlageziel.
        required_probability: Geforderte Mindest-Erfolgswahrscheinlichkeit.

    Returns:
        Das Ergebnis der Bisektion.
    """
    ziel = float(goal.target_amount)

    def wahrscheinlichkeit(rate: float) -> float:
        """Erfolgswahrscheinlichkeit bei einer gegebenen Sparrate.

        Args:
            rate: Monatliche Sparrate in Euro.

        Returns:
            Die Erfolgswahrscheinlichkeit.
        """
        return _probability(final_values(context, goal, rate), ziel)

    return bisect_increasing(
        wahrscheinlichkeit,
        float(goal.monthly_contribution),
        MAX_CONTRIBUTION,
        required_probability,
        tolerance=CONTRIBUTION_TOLERANCE,
    )


def solve_horizon(
    context: SolverContext, goal: Goal, required_probability: float
) -> BisectionResult:
    """Löst S2: erforderlicher Horizont in Monaten bei unveränderter Sparrate.

    Gerechnet wird **eine** Simulation bis zum längsten geprüften Horizont; die
    Bisektion läuft anschließend über den Monatsindex derselben Wertmatrix.

    Args:
        context: Rahmen der Suche.
        goal: Anlageziel.
        required_probability: Geforderte Mindest-Erfolgswahrscheinlichkeit.

    Returns:
        Das Ergebnis der Bisektion; der Wert ist ein ganzzahliger Monat.
    """
    ergebnis = simulate(
        context.sampler,
        context.weights,
        initial_capital=float(goal.initial_capital),
        monthly_contribution=float(goal.monthly_contribution),
        horizon_months=MAX_HORIZON_MONTHS,
        target_amount=float(goal.target_amount),
        policy=context.policies.simulation,
        paths=context.paths,
        seed=context.seed,
        asset_keys=context.asset_keys,
        track_bands=False,
        track_values=True,
    )
    verlauf = ergebnis.value_paths
    ziel = float(goal.target_amount)

    def wahrscheinlichkeit(monat: float) -> float:
        """Erfolgswahrscheinlichkeit zum abgelesenen Monat.

        Args:
            monat: Monatsindex; wird auf einen ganzen Monat gerundet.

        Returns:
            Die Erfolgswahrscheinlichkeit.
        """
        index = int(np.clip(round(monat), 0, MAX_HORIZON_MONTHS))
        return _probability(verlauf[:, index], ziel)

    bisektion = bisect_increasing(
        wahrscheinlichkeit,
        float(goal.horizon_months),
        float(MAX_HORIZON_MONTHS),
        required_probability,
        tolerance=1.0,
    )
    monat = int(np.ceil(bisektion.value))
    return BisectionResult(
        value=float(monat),
        probability=wahrscheinlichkeit(float(monat)),
        iterations=bisektion.iterations,
        reachable=bisektion.reachable,
    )


def solve_target(
    context: SolverContext, goal: Goal, required_probability: float
) -> BisectionResult:
    """Löst S3: erreichbarer Zielbetrag bei unveränderter Sparrate und Horizont.

    Der Zielbetrag geht in die Simulation **nicht** ein — er ist nur die
    Schwelle, gegen die gezählt wird. Der gesuchte Wert ist damit eine
    Ordnungsstatistik der Endwertverteilung und kann ohne Iteration abgelesen
    werden.

    **Welche Ordnungsstatistik.** Gesucht ist der *größte* Betrag, den noch
    mindestens der Anteil ``p`` der Pfade erreicht. Bei aufsteigend sortierten
    Endwerten und dem nullbasierten Index ``k`` erreichen genau ``n − k`` Pfade
    den Wert an dieser Stelle. Aus ``n − k ≥ p·n`` folgt der größte zulässige
    Index ``k = n − ⌈p·n⌉``.

    Die naheliegende Abkürzung über ``np.quantile(..., 1 − p, method="lower")``
    trifft diesen Index **nicht**: Sie rechnet mit der Position ``(n − 1)·(1 − p)``
    und liegt damit um eine Ordnungsstatistik zu tief. Der Fehler ist klein — im
    Referenzfall rund 1 700 € auf 215 000 € — und wäre ohne die Gegenprobe gegen
    :func:`solve_target_by_bisection` nicht aufgefallen.

    Args:
        context: Rahmen der Suche.
        goal: Anlageziel.
        required_probability: Geforderte Mindest-Erfolgswahrscheinlichkeit.

    Returns:
        Das Ergebnis; ``iterations`` ist null, weil nicht iteriert wurde.
    """
    endwerte = final_values(context, goal)
    sortiert = np.sort(endwerte)
    anzahl = int(sortiert.size)
    index = anzahl - int(np.ceil(required_probability * anzahl))
    betrag = float(sortiert[int(np.clip(index, 0, anzahl - 1))])
    return BisectionResult(
        value=betrag,
        probability=_probability(endwerte, betrag),
        iterations=0,
        reachable=True,
    )


def solve_target_by_bisection(
    context: SolverContext, goal: Goal, required_probability: float
) -> BisectionResult:
    """Löst S3 zusätzlich per Bisektion — als Gegenprobe zur geschlossenen Lösung.

    Args:
        context: Rahmen der Suche.
        goal: Anlageziel.
        required_probability: Geforderte Mindest-Erfolgswahrscheinlichkeit.

    Returns:
        Das Ergebnis der Bisektion.
    """
    endwerte = final_values(context, goal)

    def wahrscheinlichkeit(betrag: float) -> float:
        """Erfolgswahrscheinlichkeit für einen Zielbetrag.

        Args:
            betrag: Zielbetrag in Euro.

        Returns:
            Die Erfolgswahrscheinlichkeit.
        """
        return _probability(endwerte, betrag)

    return bisect_decreasing(
        wahrscheinlichkeit,
        0.0,
        float(np.max(endwerte)) + 1.0,
        required_probability,
        tolerance=TARGET_TOLERANCE,
    )


def build_levers(
    context: SolverContext, client: Client, goal: Goal, required_probability: float
) -> tuple[Lever, ...]:
    """Löst alle drei Stellschrauben und bewertet ihre Umsetzbarkeit.

    Args:
        context: Rahmen der Suche; sein Seed gilt für alle drei.
        client: Kundenprofil, für die Machbarkeitsprüfung von S1.
        goal: Anlageziel.
        required_probability: Geforderte Mindest-Erfolgswahrscheinlichkeit.

    Returns:
        Die drei Stellschrauben in der Reihenfolge S1, S2, S3.
    """
    vorlage = template_for(goal.goal_type, context.policies.goals)

    return (
        _contribution_lever(
            solve_contribution(context, goal, required_probability),
            goal,
            client.household.free_savings_amount,
        ),
        _horizon_lever(
            solve_horizon(context, goal, required_probability),
            goal,
            verschiebbar=vorlage.termin_verschiebbar,
        ),
        _target_lever(
            solve_target(context, goal, required_probability),
            goal,
            reduzierbar=vorlage.betrag_reduzierbar,
        ),
    )


def _quoted_goal_type(goal: Goal) -> str:
    """Setzt den Zieltyp in deutsche Anführungszeichen.

    Args:
        goal: Anlageziel.

    Returns:
        Der Zieltyp in Anführungszeichen.
    """
    return f"{QUOTE_OPEN}{goal.goal_type.value}{QUOTE_CLOSE}"


def _contribution_lever(result: BisectionResult, goal: Goal, free_savings: Decimal) -> Lever:
    """Bewertet S1 gegen die freie Sparquote des Haushalts.

    Args:
        result: Ergebnis der Bisektion.
        goal: Anlageziel.
        free_savings: Monatlich frei verfügbarer Betrag.

    Returns:
        Die bewertete Stellschraube.
    """
    erforderlich = Decimal(str(result.value)).quantize(EURO_QUANT, rounding=ROUND_CEILING)

    if not result.reachable:
        bewertung = Feasibility.NICHT_ERREICHBAR
        hinweis = (
            f"Auch bei einer Sparrate von {format_de.euro(MAX_CONTRIBUTION, decimals=0)} "
            "monatlich wird die geforderte Erfolgswahrscheinlichkeit nicht erreicht. Die "
            "Sparrate allein löst den Konflikt nicht."
        )
    elif erforderlich > free_savings:
        bewertung = Feasibility.NICHT_UMSETZBAR
        hinweis = (
            f"Erforderlich sind {format_de.euro(erforderlich)} monatlich, frei verfügbar "
            f"sind {format_de.euro(free_savings)}. Die Differenz von "
            f"{format_de.euro(erforderlich - free_savings)} ist im erhobenen Haushalt nicht "
            "darstellbar. Die Stellschraube wird ausgewiesen, aber nicht als Vorschlag "
            "angeboten."
        )
    else:
        bewertung = Feasibility.UMSETZBAR
        hinweis = (
            f"Erforderlich sind {format_de.euro(erforderlich)} monatlich; die freie "
            f"Sparquote beträgt {format_de.euro(free_savings)}. Umsetzbar, sofern kein "
            "anderes Ziel denselben Betrag beansprucht."
        )

    return Lever(
        lever_id=LeverId.SPARRATE,
        name="Erforderliche monatliche Sparrate",
        unit="EUR je Monat",
        current_value=goal.monthly_contribution,
        required_value=erforderlich,
        achieved_probability=result.probability,
        feasibility=bewertung,
        feasibility_note=hinweis,
        iterations=result.iterations,
        method="Bisektion auf der Monte-Carlo-Erfolgswahrscheinlichkeit, Seed fixiert",
    )


def _horizon_lever(result: BisectionResult, goal: Goal, *, verschiebbar: bool) -> Lever:
    """Bewertet S2 gegen die Verschiebbarkeit des Zieltermins.

    Args:
        result: Ergebnis der Bisektion.
        goal: Anlageziel.
        verschiebbar: Ob der Zieltermin laut Zielkatalog verschiebbar ist.

    Returns:
        Die bewertete Stellschraube.
    """
    erforderlich = Decimal(int(result.value))
    zusatz = erforderlich - Decimal(goal.horizon_months)
    zieltyp = _quoted_goal_type(goal)

    if not result.reachable:
        bewertung = Feasibility.NICHT_ERREICHBAR
        hinweis = (
            f"Auch bei einer Verlängerung auf {MAX_HORIZON_MONTHS} Monate wird die "
            "geforderte Erfolgswahrscheinlichkeit nicht erreicht."
        )
    elif not verschiebbar:
        bewertung = Feasibility.NICHT_UMSETZBAR
        hinweis = (
            f"Der Zieltermin müsste um {zusatz} Monate verschoben werden. Für den Zieltyp "
            f"{zieltyp} ist der Termin laut Zielkatalog nicht verschiebbar; die "
            "Stellschraube wird ausgewiesen, aber nicht als Vorschlag angeboten."
        )
    else:
        jahre = (zusatz / Decimal(MONTHS_PER_YEAR)).quantize(Decimal("0.1"))
        bewertung = Feasibility.UMSETZBAR
        hinweis = (
            f"Der Zieltermin müsste um {zusatz} Monate ({format_de.decimal(jahre, 1)} Jahre) "
            f"nach hinten verschoben werden. Für den Zieltyp {zieltyp} ist das laut "
            "Zielkatalog möglich."
        )

    return Lever(
        lever_id=LeverId.HORIZONT,
        name="Erforderlicher Anlagehorizont",
        unit="Monate",
        current_value=Decimal(goal.horizon_months),
        required_value=erforderlich,
        achieved_probability=result.probability,
        feasibility=bewertung,
        feasibility_note=hinweis,
        iterations=result.iterations,
        method="Bisektion über den Monatsindex einer gemeinsamen Pfadschar, Seed fixiert",
    )


def _target_lever(result: BisectionResult, goal: Goal, *, reduzierbar: bool) -> Lever:
    """Bewertet S3 gegen die Reduzierbarkeit des Zielbetrags.

    Args:
        result: Ergebnis der Suche.
        goal: Anlageziel.
        reduzierbar: Ob der Zielbetrag laut Zielkatalog reduzierbar ist.

    Returns:
        Die bewertete Stellschraube.
    """
    erreichbar = Decimal(str(result.value)).quantize(EURO_QUANT, rounding=ROUND_FLOOR)
    minderung = goal.target_amount - erreichbar
    anteil = (
        (minderung / goal.target_amount * Decimal(100)).quantize(Decimal("0.1"))
        if goal.target_amount > 0
        else Decimal(0)
    )
    zieltyp = _quoted_goal_type(goal)
    kern = (
        f"Erreichbar sind {format_de.euro(erreichbar)} statt "
        f"{format_de.euro(goal.target_amount)}, also {format_de.euro(minderung)} oder "
        f"{format_de.decimal(anteil, 1, thousands=False)} % weniger."
    )

    if not reduzierbar:
        bewertung = Feasibility.NICHT_UMSETZBAR
        hinweis = (
            f"{kern} Für den Zieltyp {zieltyp} ist der Betrag laut Zielkatalog nicht "
            "reduzierbar; die Stellschraube wird ausgewiesen, aber nicht als Vorschlag "
            "angeboten."
        )
    else:
        bewertung = Feasibility.UMSETZBAR
        hinweis = f"{kern} Für den Zieltyp {zieltyp} ist eine Anpassung laut Zielkatalog möglich."

    return Lever(
        lever_id=LeverId.ZIELBETRAG,
        name="Erreichbarer Zielbetrag",
        unit="EUR",
        current_value=goal.target_amount,
        required_value=erreichbar,
        achieved_probability=result.probability,
        feasibility=bewertung,
        feasibility_note=hinweis,
        iterations=result.iterations,
        method="Empirisches Quantil der Endwertverteilung (geschlossene Lösung)",
    )
