"""Simulation der Zielerreichung über einen Block-Bootstrap.

Gefragt ist nicht, welchen Endwert ein Sparplan im Mittel erreicht, sondern
**mit welcher Wahrscheinlichkeit** er den Zielbetrag erreicht. Das ist eine
Verteilungsfrage, und sie hängt daran, wie realistisch die simulierten
Renditepfade sind.

Warum Block-Bootstrap
--------------------
Monatsrenditen sind nicht unabhängig identisch verteilt. Drei Eigenschaften
gehen bei einer i.i.d.-Ziehung verloren:

* **Volatilitätscluster.** Auf turbulente Monate folgen turbulente Monate. Eine
  i.i.d.-Ziehung verteilt die Extremmonate gleichmäßig über den Pfad, statt sie
  zu bündeln.
* **Fat Tails.** Die empirische Verteilung hat mehr Masse in den Rändern als
  eine Normalverteilung. Der Bootstrap zieht die tatsächlich beobachteten
  Extremmonate mit ihrer tatsächlichen Häufigkeit.
* **Sequence-of-Returns-Risiko.** Für einen Sparplan ist die *Reihenfolge* der
  Renditen erheblich. Ein Einbruch am Anfang trifft wenig Kapital, ein Einbruch
  kurz vor dem Zieltermin trifft alles. Nur eine Ziehung, die zusammenhängende
  Zeitabschnitte erhält, bildet das ab.

Der entscheidende Konstruktionspunkt
------------------------------------
Gezogen wird **ein Zeitindex für alle Assetklassen gemeinsam**. Ein Block ist
kein Bündel von acht unabhängigen Renditereihen, sondern ein Ausschnitt aus der
gemeinsamen Geschichte: derselbe Monat, alle acht Assetklassen.

Damit bleibt die **Korrelationsstruktur** erhalten — und zwar nicht nur im
Mittel, sondern gerade in den Krisenmonaten, in denen Korrelationen bekanntlich
steigen. Würde man je Assetklasse getrennt ziehen, entstünde ein Portfolio, das
sich besser diversifiziert verhält als jedes real mögliche. Die
Erfolgswahrscheinlichkeit wäre systematisch zu hoch.

Was der Bootstrap nicht kann
----------------------------
Er zieht nur, was historisch vorkam. Kein simulierter Monat ist schlechter als
der schlechteste beobachtete Monat; die Verteilung hat einen harten Rand, den
eine parametrische Ziehung nicht hätte. Die Historie umfasst 220 Monate von
März 2008 bis Juni 2026 und enthält weder das Zinsniveau vor 2008 noch eine
Inflationsphase wie in den 1970er Jahren noch einen Staatsanleihenausfall im
Euroraum. Der Zeitraum war für Aktien überdurchschnittlich günstig, und der
Bootstrap erbt diese Günstigkeit vollständig: Er zieht aus einer Stichprobe,
deren Mittelwert er per Konstruktion trifft. Die Blocklänge von zwölf Monaten
erfasst zudem nur Abhängigkeiten innerhalb eines Jahres – längere Zyklen
zerfallen an den Blockgrenzen.

Aufbau der Simulation
---------------------
Vektorisiert über die Pfade, Schleife über die Monate. Je Monat:

1. Renditen anwenden,
2. laufende Kosten (TER) geometrisch abziehen,
3. Sparrate einzahlen, abzüglich Transaktionskosten,
4. bei Fälligkeit auf die Zielgewichte zurückführen, abzüglich
   Transaktionskosten des Umschichtungsvolumens.

Die Pfade werden in Blöcken fester Größe verarbeitet (:data:`BATCH_SIZE`), damit
der Speicherbedarf auch bei 50 000 Pfaden beschränkt bleibt. **Die Blockgröße
ist Teil der Reproduzierbarkeitszusage**: Sie bestimmt die Reihenfolge, in der
Zufallszahlen gezogen werden. Eine Änderung verändert die Ergebnisse und ist
deshalb wie eine Policy-Änderung zu behandeln.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt

from candor.domain.policy import SimulationPolicy

#: Zahl der Pfade, die gemeinsam gerechnet werden. Teil der
#: Reproduzierbarkeitszusage — siehe Modul-Docstring.
BATCH_SIZE: Final = 2500

#: Monate je Jahr.
MONTHS_PER_YEAR: Final = 12

#: Perzentile, die standardmäßig ausgewiesen werden.
DEFAULT_PERCENTILES: Final[tuple[int, ...]] = (5, 25, 50, 75, 95)

#: Erwartete Achsenzahl der Renditematrix: Monate × Assetklassen.
HISTORY_DIMENSIONS: Final = 2

FloatArray = npt.NDArray[np.float64]


class ReturnSampler(Protocol):
    """Quelle simulierter Monatsrenditen.

    Implementierungen liefern für eine Zahl von Pfaden und Monaten ein Feld der
    Form ``(pfade, monate, assetklassen)``. Die Spaltenreihenfolge entspricht
    stets der des übergebenen Renditearchivs.
    """

    @property
    def label(self) -> str:
        """Bezeichnung des Verfahrens für Trace und Bericht."""
        ...

    def sample(self, rng: np.random.Generator, paths: int, horizon: int) -> FloatArray:
        """Zieht Renditepfade.

        Args:
            rng: Zufallsgenerator mit explizitem Seed (Prinzip P1).
            paths: Zahl der zu ziehenden Pfade.
            horizon: Zahl der Monate je Pfad.

        Returns:
            Feld der Form ``(paths, horizon, assetklassen)``.
        """
        ...


class BlockBootstrapSampler:
    """Überlappender Block-Bootstrap mit Ziehung mit Zurücklegen, zirkulär.

    Aus der historischen Renditematrix werden zusammenhängende Blöcke fester
    Länge gezogen und aneinandergehängt, bis der Horizont gefüllt ist. Die
    Blöcke überlappen sich — jeder Monat kommt als Blockanfang in Betracht,
    nicht nur jeder zwölfte.

    Warum zirkulär
    --------------
    Bei nicht zirkulärer Ziehung liegen die zulässigen Blockanfänge zwischen 0
    und ``N − L``. Ein Monat am **Rand** der Historie erscheint dann in weniger
    Blöcken als ein Monat in der Mitte: Der erste Monat nur im Block, der bei
    null beginnt, der zwölfte dagegen in zwölf verschiedenen Blöcken. Die
    Randmonate sind damit systematisch untergewichtet.

    Das ist keine Feinheit. In diesem Projekt beginnt die Historie im März 2008,
    also mitten in der Finanzkrise. Eine Untergewichtung des Randes bedeutet hier
    eine Untergewichtung der schlechtesten Monate des gesamten Zeitraums. Gemessen
    an der Referenzallokation der Klasse 3 lag das simulierte Monatsmittel bei
    0,4240 % gegenüber 0,3893 % in der Historie — eine Überschätzung um
    0,035 Prozentpunkte je Monat oder rund 0,42 Prozentpunkte je Jahr, die sich
    über zwanzig Jahre zu einer erheblich zu hohen Erfolgswahrscheinlichkeit
    aufsummiert.

    Die zirkuläre Variante (Politis & Romano) behebt das: Blockanfänge liegen
    zwischen 0 und ``N − 1``, und der Block läuft am Ende der Reihe auf den
    Anfang um. Jeder Monat erscheint dadurch in genau ``L`` Blöcken, und der
    Erwartungswert der Ziehung entspricht exakt dem historischen Mittel.

    **Der Preis:** Je Block entsteht höchstens eine künstliche Naht, an der der
    letzte Monat der Reihe auf den ersten trifft — im vorliegenden Fall Juni 2026
    auf März 2008. Diese Naht ist ökonomisch sinnlos. Bei 220 Monaten Historie
    und Blocklänge 12 betrifft sie rund 5 % der Monatsübergänge. Die Verzerrung
    des Mittelwerts wiegt schwerer als eine seltene unsinnige Naht, weshalb die
    zirkuläre Variante die Vorgabe ist. Sie lässt sich über ``circular=False``
    abschalten, was ausschließlich für den Nachweis der Verzerrung selbst
    gedacht ist.
    """

    def __init__(self, history: FloatArray, block_length: int, *, circular: bool = True) -> None:
        """Legt den Sampler an.

        Args:
            history: Historische Monatsrenditen, Form ``(monate, assetklassen)``,
                ohne fehlende Werte.
            block_length: Länge eines Blocks in Monaten.
            circular: Ob Blöcke am Ende der Reihe auf den Anfang umlaufen.
                Vorgabe ``True``; siehe Klassendokumentation.

        Raises:
            ValueError: Wenn die Historie kürzer als ein Block ist, fehlende
                Werte enthält oder die Blocklänge nicht positiv ist.
        """
        if block_length < 1:
            msg = f"Die Blocklänge muss positiv sein, ist aber {block_length}."
            raise ValueError(msg)
        if history.ndim != HISTORY_DIMENSIONS:
            msg = f"Die Historie muss zweidimensional sein, hat aber {history.ndim} Achsen."
            raise ValueError(msg)
        if not np.isfinite(history).all():
            msg = (
                "Die Historie enthält fehlende oder unendliche Werte. Eine simultane "
                "Ziehung über alle Assetklassen setzt vollständige Zeilen voraus."
            )
            raise ValueError(msg)
        if history.shape[0] < block_length:
            msg = (
                f"Die Historie umfasst {history.shape[0]} Monate und ist damit kürzer "
                f"als ein Block von {block_length} Monaten."
            )
            raise ValueError(msg)

        self._history = history
        self._block_length = block_length
        self._length = int(history.shape[0])
        self._circular = circular
        self._starts = self._length if circular else self._length - block_length + 1

    @property
    def label(self) -> str:
        """Bezeichnung des Verfahrens."""
        art = "zirkulär" if self._circular else "nicht zirkulär"
        return (
            f"Block-Bootstrap, Blocklänge {self._block_length} Monate, überlappend, "
            f"{art}, {self._starts} mögliche Blöcke"
        )

    @property
    def block_count(self) -> int:
        """Zahl der möglichen Blockanfänge in der Historie."""
        return self._starts

    def sample(self, rng: np.random.Generator, paths: int, horizon: int) -> FloatArray:
        """Zieht Renditepfade aus zusammenhängenden Blöcken.

        Args:
            rng: Zufallsgenerator.
            paths: Zahl der Pfade.
            horizon: Zahl der Monate je Pfad.

        Returns:
            Feld der Form ``(paths, horizon, assetklassen)``.
        """
        bloecke = -(-horizon // self._block_length)  # aufgerundete Division
        anfaenge = rng.integers(0, self._starts, size=(paths, bloecke))
        versatz = np.arange(self._block_length)
        # (pfade, bloecke, blocklaenge) -> (pfade, bloecke*blocklaenge) -> zuschneiden
        index = (anfaenge[:, :, None] + versatz[None, None, :]).reshape(paths, -1)[:, :horizon]
        if self._circular:
            # Umlauf am Ende der Reihe. Erst dadurch erscheint jeder Monat in
            # genau L Bloecken und der Erwartungswert bleibt unverzerrt.
            index = index % self._length
        # Ein Index je Monat, für ALLE Assetklassen gemeinsam. Genau hier bleibt
        # die Korrelationsstruktur erhalten.
        gezogen: FloatArray = self._history[index]
        return gezogen


class IIDNormalSampler:
    """Parametrische Ziehung aus einer mehrdimensionalen Normalverteilung.

    Erwartungswerte und Kovarianzmatrix werden aus derselben Historie geschätzt.
    Die **gleichzeitige** Korrelation bleibt damit erhalten; verloren gehen
    Autokorrelation, Volatilitätscluster und die empirischen Randbereiche.

    Der Sampler ist nicht als Alternative gedacht, sondern als **Vergleichsmaß**:
    Er beantwortet die Frage, ob der Mehraufwand des Bootstrap überhaupt einen
    Unterschied macht. Die gerechnete Antwort: in der Mitte der Verteilung kaum
    – die Erfolgswahrscheinlichkeit weicht über alle fünf Risikoklassen um
    höchstens 1,16 Prozentpunkte ab –, an den Rändern deutlich: Das fünfte
    Perzentil liegt im Bootstrap um bis zu 12.743 € niedriger, der schlechteste
    Rückgang um bis zu 24 Prozentpunkte tiefer. Die Normalziehung unterschätzt
    das Verlustrisiko systematisch.
    """

    def __init__(self, history: FloatArray) -> None:
        """Legt den Sampler an.

        Args:
            history: Historische Monatsrenditen, Form ``(monate, assetklassen)``.

        Raises:
            ValueError: Wenn die Historie fehlende Werte enthält oder zu kurz für
                eine Kovarianzschätzung ist.
        """
        if not np.isfinite(history).all():
            msg = "Die Historie enthält fehlende oder unendliche Werte."
            raise ValueError(msg)
        if history.shape[0] < history.shape[1] + 1:
            msg = (
                f"Für {history.shape[1]} Assetklassen sind mindestens "
                f"{history.shape[1] + 1} Monate nötig, vorhanden sind {history.shape[0]}."
            )
            raise ValueError(msg)

        self._mean = history.mean(axis=0)
        kovarianz = np.cov(history, rowvar=False)
        # Cholesky-Zerlegung; ein winziger Aufschlag auf der Diagonalen macht die
        # Zerlegung auch bei numerisch grenzwertiger Matrix durchführbar.
        self._chol = np.linalg.cholesky(kovarianz + np.eye(kovarianz.shape[0]) * 1e-12)

    @property
    def label(self) -> str:
        """Bezeichnung des Verfahrens."""
        return "i.i.d. mehrdimensional normalverteilt (Momente aus derselben Historie)"

    def sample(self, rng: np.random.Generator, paths: int, horizon: int) -> FloatArray:
        """Zieht unabhängige normalverteilte Renditepfade.

        Args:
            rng: Zufallsgenerator.
            paths: Zahl der Pfade.
            horizon: Zahl der Monate je Pfad.

        Returns:
            Feld der Form ``(paths, horizon, assetklassen)``.
        """
        standard = rng.standard_normal((paths, horizon, self._mean.shape[0]))
        gezogen: FloatArray = self._mean + standard @ self._chol.T
        return gezogen


@dataclass(frozen=True, slots=True)
class _BatchConfig:
    """Unveränderliche Parameter eines Simulationsdurchlaufs.

    Gebündelt, weil sie über alle Pfadblöcke hinweg identisch sind und die
    innere Schleife sonst sieben Einzelparameter durchreichen müsste.

    Attributes:
        weights: Zielgewichte je Assetklasse.
        initial_capital: Startkapital.
        monthly_contribution: Monatliche Sparrate.
        ter_factor: Monatlicher TER-Faktor je Assetklasse.
        spread: Einseitige Transaktionskosten je Assetklasse.
        rebalancing_months: Abstand zweier Rebalancings in Monaten.
    """

    weights: FloatArray
    initial_capital: float
    monthly_contribution: float
    ter_factor: FloatArray
    spread: FloatArray
    rebalancing_months: int


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Ergebnis einer Zielerreichungssimulation.

    Attributes:
        paths: Zahl der gerechneten Pfade.
        horizon_months: Zahl der simulierten Monate.
        seed: Verwendeter Seed.
        method: Bezeichnung des Ziehungsverfahrens.
        target_amount: Zielbetrag, gegen den geprüft wurde.
        total_contributions: Summe aus Startkapital und allen Einzahlungen.
        final_values: Endwerte je Pfad.
        max_drawdowns: Größter Rückgang je Pfad, als negative Zahl.
        value_bands: Perzentilbänder des Portfoliowerts je Monat, Schlüssel ist
            das Perzentil. Leer, wenn der Verlauf nicht mitgeschrieben wurde.
        value_paths: Vollständige Wertmatrix der Form ``(pfade, monate+1)``,
            sofern angefordert. Wird für die Stellschraube S2 benötigt, die den
            Ablesezeitpunkt variiert, ohne die Pfade zu verändern.
        policy_version: Fassung der Simulationspolicy.
    """

    paths: int
    horizon_months: int
    seed: int
    method: str
    target_amount: float
    total_contributions: float
    final_values: FloatArray
    max_drawdowns: FloatArray
    value_bands: dict[int, FloatArray]
    value_paths: FloatArray
    policy_version: str

    @property
    def success_probability(self) -> float:
        """Anteil der Pfade, die den Zielbetrag erreichen oder übertreffen."""
        return float(np.mean(self.final_values >= self.target_amount))

    @property
    def shortfall_probability(self) -> float:
        """Anteil der Pfade, die den Zielbetrag verfehlen."""
        return 1.0 - self.success_probability

    @property
    def expected_shortfall(self) -> float:
        """Erwarteter Fehlbetrag **im Misserfolgsfall**.

        Bedingter Erwartungswert: gemittelt wird ausschließlich über die Pfade,
        die das Ziel verfehlen. Die unbedingte Größe wäre irreführend — sie
        mischt geglückte und misslungene Pfade und verharmlost damit genau den
        Fall, um den es geht.

        Returns:
            Der mittlere Fehlbetrag in Euro, oder ``0.0``, wenn kein Pfad das
            Ziel verfehlt.
        """
        verfehlt = self.final_values[self.final_values < self.target_amount]
        if verfehlt.size == 0:
            return 0.0
        return float(np.mean(self.target_amount - verfehlt))

    @property
    def median_max_drawdown(self) -> float:
        """Median des größten Rückgangs über die Pfade."""
        return float(np.median(self.max_drawdowns))

    @property
    def worst_max_drawdown(self) -> float:
        """Größter Rückgang über alle Pfade."""
        return float(np.min(self.max_drawdowns))

    def percentile(self, level: int) -> float:
        """Perzentil der Endwertverteilung.

        Args:
            level: Perzentil zwischen 0 und 100.

        Returns:
            Der Endwert an dieser Stelle der Verteilung.
        """
        return float(np.percentile(self.final_values, level))

    def percentiles(self, levels: tuple[int, ...] = DEFAULT_PERCENTILES) -> dict[int, float]:
        """Mehrere Perzentile der Endwertverteilung.

        Args:
            levels: Gewünschte Perzentile.

        Returns:
            Zuordnung Perzentil auf Endwert.
        """
        return {stufe: self.percentile(stufe) for stufe in levels}

    def as_summary(self) -> dict[str, object]:
        """Verdichtet das Ergebnis für Anzeige, Trace und Bericht.

        Returns:
            Zuordnung der wesentlichen Kennzahlen.
        """
        return {
            "verfahren": self.method,
            "pfade": self.paths,
            "horizont_monate": self.horizon_months,
            "seed": self.seed,
            "zielbetrag": round(self.target_amount, 2),
            "einzahlungen_gesamt": round(self.total_contributions, 2),
            "erfolgswahrscheinlichkeit": round(self.success_probability, 6),
            "shortfall_wahrscheinlichkeit": round(self.shortfall_probability, 6),
            "erwarteter_fehlbetrag_im_misserfolg": round(self.expected_shortfall, 2),
            "perzentile": {stufe: round(wert, 2) for stufe, wert in self.percentiles().items()},
            "median_max_drawdown": round(self.median_max_drawdown, 6),
            "schlechtester_max_drawdown": round(self.worst_max_drawdown, 6),
            "policy_version": self.policy_version,
        }


def monthly_cost_factors(
    asset_keys: tuple[str, ...], policy: SimulationPolicy
) -> tuple[FloatArray, FloatArray]:
    """Bildet die monatlichen Kostenfaktoren je Assetklasse.

    Args:
        asset_keys: Assetklassen in Spaltenreihenfolge der Historie.
        policy: Simulationspolicy mit den Kostenannahmen.

    Returns:
        Paar aus dem monatlichen TER-Faktor und dem einseitigen Spread je
        Assetklasse.
    """
    ter = np.array([float(policy.kosten[schluessel].ter_jaehrlich) for schluessel in asset_keys])
    spread = np.array(
        [float(policy.kosten[schluessel].spread_einmalig) for schluessel in asset_keys]
    )
    # Geometrischer Monatsabzug: Zwölf Monate ergeben zusammen genau die
    # Jahres-TER, nicht das Zwölffache eines Zwölftels.
    ter_faktor = (1.0 - ter) ** (1.0 / MONTHS_PER_YEAR)
    return ter_faktor, spread


def simulate(  # noqa: PLR0913 — jede Angabe ist eine eigene fachliche Größe
    sampler: ReturnSampler,
    weights: FloatArray,
    *,
    initial_capital: float,
    monthly_contribution: float,
    horizon_months: int,
    target_amount: float,
    policy: SimulationPolicy,
    paths: int | None = None,
    seed: int | None = None,
    asset_keys: tuple[str, ...] = (),
    track_bands: bool = True,
    track_values: bool = False,
) -> SimulationResult:
    """Simuliert die Zielerreichung eines Sparplans.

    Args:
        sampler: Ziehungsverfahren für die Monatsrenditen.
        weights: Zielgewichte je Assetklasse, Summe eins.
        initial_capital: Startkapital.
        monthly_contribution: Monatliche Sparrate, nachschüssig.
        horizon_months: Zahl der simulierten Monate.
        target_amount: Zielbetrag, gegen den die Erfolgswahrscheinlichkeit
            gemessen wird.
        policy: Simulationspolicy mit Kosten und Rebalancing-Frequenz.
        paths: Zahl der Pfade. Ohne Angabe der Wert aus der Policy.
        seed: Seed. Ohne Angabe der Wert aus der Policy.
        asset_keys: Assetklassen in Spaltenreihenfolge, für die Kostenzuordnung.
        track_bands: Ob der Wertverlauf für die Perzentilbänder mitgeschrieben
            wird. Bei sehr vielen Pfaden ausschaltbar, um Speicher zu sparen.
        track_values: Ob die vollständige Wertmatrix im Ergebnis erhalten bleibt.
            Nötig für die Stellschraube S2; kostet ``pfade × monate``
            Gleitkommazahlen Speicher.

    Returns:
        Das Simulationsergebnis.

    Raises:
        ValueError: Wenn Gewichte, Horizont oder Beträge unzulässig sind.
    """
    if horizon_months < 1:
        msg = f"Der Horizont muss mindestens einen Monat betragen, ist aber {horizon_months}."
        raise ValueError(msg)
    if weights.ndim != 1:
        msg = "Die Gewichte müssen eindimensional sein."
        raise ValueError(msg)
    if not np.isclose(weights.sum(), 1.0, atol=1e-9):
        msg = f"Die Gewichte müssen sich zu eins addieren, ergeben aber {weights.sum()}."
        raise ValueError(msg)
    if (weights < 0).any():
        msg = "Negative Gewichte sind unzulässig (keine Leerverkäufe)."
        raise ValueError(msg)
    if initial_capital < 0 or monthly_contribution < 0:
        msg = "Startkapital und Sparrate dürfen nicht negativ sein."
        raise ValueError(msg)

    anzahl_pfade = paths if paths is not None else policy.standard_pfade
    verwendeter_seed = seed if seed is not None else policy.basis_seed
    schluessel = asset_keys or tuple(sorted(policy.kosten))
    ter_faktor, spread = monthly_cost_factors(schluessel, policy)
    rebalancing = policy.rebalancing_monate

    konfiguration = _BatchConfig(
        weights=weights,
        initial_capital=initial_capital,
        monthly_contribution=monthly_contribution,
        ter_factor=ter_faktor,
        spread=spread,
        rebalancing_months=rebalancing,
    )
    rng = np.random.default_rng(verwendeter_seed)

    endwerte = np.empty(anzahl_pfade, dtype=np.float64)
    drawdowns = np.empty(anzahl_pfade, dtype=np.float64)
    baender: list[FloatArray] = []

    begonnen = 0
    while begonnen < anzahl_pfade:
        groesse = min(BATCH_SIZE, anzahl_pfade - begonnen)
        renditen = sampler.sample(rng, groesse, horizon_months)
        werte, verlauf, block_drawdown = _run_batch(
            renditen, konfiguration, track_bands=track_bands or track_values
        )
        endwerte[begonnen : begonnen + groesse] = werte
        drawdowns[begonnen : begonnen + groesse] = block_drawdown
        if verlauf is not None:
            baender.append(verlauf)
        begonnen += groesse

    bandwerte: dict[int, FloatArray] = {}
    verlaeufe = np.concatenate(baender, axis=0) if baender else np.empty((0, 0))
    if baender and track_bands:
        for stufe in DEFAULT_PERCENTILES:
            bandwerte[stufe] = np.percentile(verlaeufe, stufe, axis=0).astype(np.float64)

    return SimulationResult(
        paths=anzahl_pfade,
        horizon_months=horizon_months,
        seed=verwendeter_seed,
        method=sampler.label,
        target_amount=target_amount,
        total_contributions=initial_capital + monthly_contribution * horizon_months,
        final_values=endwerte,
        max_drawdowns=drawdowns,
        value_bands=bandwerte,
        value_paths=verlaeufe if track_values else np.empty((0, 0)),
        policy_version=policy.version,
    )


def _run_batch(
    returns: FloatArray,
    config: _BatchConfig,
    *,
    track_bands: bool,
) -> tuple[FloatArray, FloatArray | None, FloatArray]:
    """Rechnet einen Block von Pfaden durch.

    Args:
        returns: Renditen der Form ``(pfade, monate, assetklassen)``.
        config: Unveränderliche Parameter des Durchlaufs.
        track_bands: Ob der Wertverlauf mitgeschrieben wird.

    Returns:
        Tripel aus Endwerten, Wertverlauf (oder ``None``) und größtem Rückgang
        je Pfad.
    """
    anzahl, monate, _ = returns.shape
    gewichte = config.weights
    spread = config.spread

    # Erstanlage nach Zielgewichten, abzüglich Transaktionskosten.
    anfang = config.initial_capital * gewichte
    werte = np.tile(anfang * (1.0 - spread), (anzahl, 1))

    gesamt = werte.sum(axis=1)
    hoechststand = gesamt.copy()
    groesster_rueckgang = np.zeros(anzahl, dtype=np.float64)

    verlauf: FloatArray | None = None
    if track_bands:
        verlauf = np.empty((anzahl, monate + 1), dtype=np.float64)
        verlauf[:, 0] = gesamt

    kauf_netto = config.monthly_contribution * gewichte * (1.0 - spread)

    for t in range(monate):
        werte *= 1.0 + returns[:, t, :]
        werte *= config.ter_factor
        if config.monthly_contribution > 0:
            werte += kauf_netto

        faellig = config.rebalancing_months > 0 and (t + 1) % config.rebalancing_months == 0
        if faellig:
            summe = werte.sum(axis=1, keepdims=True)
            zielwerte = summe * gewichte
            handelskosten = (np.abs(zielwerte - werte) * spread).sum(axis=1, keepdims=True)
            werte = zielwerte - handelskosten * gewichte

        gesamt = werte.sum(axis=1)
        hoechststand = np.maximum(hoechststand, gesamt)
        # Rückgang gegenüber dem bisherigen Höchststand. Bei Höchststand null —
        # weder Startkapital noch bereits eingezahlt — bleibt der Rückgang null.
        mit_stand = hoechststand > 0
        rueckgang = np.zeros(anzahl, dtype=np.float64)
        rueckgang[mit_stand] = gesamt[mit_stand] / hoechststand[mit_stand] - 1.0
        groesster_rueckgang = np.minimum(groesster_rueckgang, rueckgang)

        if verlauf is not None:
            verlauf[:, t + 1] = gesamt

    return gesamt, verlauf, groesster_rueckgang
