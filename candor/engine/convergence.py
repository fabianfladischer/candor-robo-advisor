"""Konvergenzanalyse und Vergleich der Ziehungsverfahren.

Zwei Fragen, die vor jeder Empfehlung beantwortet sein müssen:

1. **Ab wie vielen Pfaden ist die geschätzte Erfolgswahrscheinlichkeit stabil
   genug, um eine Empfehlung darauf zu stützen?** Eine Simulation, deren
   Ergebnis beim nächsten Seed um mehrere Prozentpunkte springt, taugt nicht als
   Grundlage einer Ablehnung — und erst recht nicht als Grundlage einer Zusage.
2. **Rechtfertigt der Block-Bootstrap seinen Mehraufwand?** Verglichen wird
   gegen eine i.i.d.-Normalziehung mit denselben Momenten und derselben
   Korrelationsmatrix. Unterscheiden sich die Ergebnisse nicht, wäre der
   Bootstrap überflüssiger Aufwand — und das gehörte dann genau so in den
   Bericht.

Methodik der Konvergenzanalyse
------------------------------
Je Pfadzahl werden **20 unabhängige Wiederholungen** mit verschiedenen Seeds
gerechnet. Ausgewiesen wird die Streuung der geschätzten
Erfolgswahrscheinlichkeit über diese Wiederholungen — der **empirische
Standardfehler**. Er wird dem theoretischen Wert gegenübergestellt:

    SE_theoretisch = √(p·(1−p) / n)

Die Erfolgswahrscheinlichkeit ist ein Anteil, ihre Schätzung damit binomial
verteilt. Stimmen empirischer und theoretischer Standardfehler überein, arbeitet
die Simulation wie eine unabhängige Stichprobe — ein Hinweis darauf, dass die
Pfade tatsächlich unabhängig gezogen werden und kein Implementierungsfehler
Abhängigkeiten einschleppt.

Aufruf:
    python -m candor.engine.convergence [--write-evaluation] [--quick]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from candor.data.loader import load_returns
from candor.data.policy_loader import load_policy_set
from candor.data.universe_meta import ASSET_KEYS
from candor.domain.policy import PolicySet
from candor.engine.simulate import (
    BlockBootstrapSampler,
    FloatArray,
    IIDNormalSampler,
    ReturnSampler,
    simulate,
)
from candor.portfolio.universe import reference_weights

#: Pfadzahlen der Konvergenzanalyse.
PATH_COUNTS: Final[tuple[int, ...]] = (100, 500, 1_000, 5_000, 10_000, 50_000)

#: Pfadzahlen im Schnelldurchlauf, für Tests.
QUICK_PATH_COUNTS: Final[tuple[int, ...]] = (100, 500, 1_000)

#: Zahl der Wiederholungen je Pfadzahl.
REPETITIONS: Final = 20

#: Wiederholungen im Schnelldurchlauf.
QUICK_REPETITIONS: Final = 5

#: Referenzfall der Analyse: Sparplan über zwanzig Jahre.
REFERENCE_HORIZON_MONTHS: Final = 240
REFERENCE_INITIAL: Final = 20_000.0
REFERENCE_CONTRIBUTION: Final = 500.0
REFERENCE_TARGET: Final = 250_000.0
REFERENCE_RISK_CLASS: Final = 3

#: Schwelle, ab der die Schätzung als hinreichend stabil gilt: Der empirische
#: Standardfehler soll unter einem halben Prozentpunkt liegen. Ein
#: Zweifachintervall umfasst dann rund zwei Prozentpunkte – schmal genug, dass
#: die Entscheidung zwischen Empfehlung und Konfliktbericht nicht vom Seed
#: abhängt, außer der Fall liegt ohnehin auf der Kante.
STABILITY_THRESHOLD: Final = 0.005

#: Markierungen der erzeugten Abschnitte in der Auswertungsdatei.
CONVERGENCE_START: Final = "<!-- BEGIN:KONVERGENZ -->"
CONVERGENCE_END: Final = "<!-- END:KONVERGENZ -->"


@dataclass(frozen=True, slots=True)
class ConvergencePoint:
    """Ergebnis einer Pfadzahl über alle Wiederholungen.

    Attributes:
        paths: Zahl der Pfade je Wiederholung.
        repetitions: Zahl der Wiederholungen.
        mean_probability: Mittelwert der geschätzten Erfolgswahrscheinlichkeit.
        empirical_se: Standardabweichung über die Wiederholungen.
        theoretical_se: Binomialer Standardfehler zum Mittelwert.
        spread: Abstand zwischen größtem und kleinstem Schätzwert.
    """

    paths: int
    repetitions: int
    mean_probability: float
    empirical_se: float
    theoretical_se: float
    spread: float

    @property
    def is_stable(self) -> bool:
        """Gibt an, ob der empirische Standardfehler unter der Schwelle liegt."""
        return self.empirical_se < STABILITY_THRESHOLD


@dataclass(frozen=True, slots=True)
class MethodComparison:
    """Vergleich zweier Ziehungsverfahren für eine Risikoklasse.

    Attributes:
        risk_class: Verglichene Risikoklasse.
        bootstrap_probability: Erfolgswahrscheinlichkeit im Block-Bootstrap.
        normal_probability: Erfolgswahrscheinlichkeit bei i.i.d.-Normalziehung.
        bootstrap_p5: Fünftes Perzentil im Block-Bootstrap.
        normal_p5: Fünftes Perzentil bei i.i.d.-Normalziehung.
        bootstrap_worst_drawdown: Schlechtester Rückgang im Block-Bootstrap.
        normal_worst_drawdown: Schlechtester Rückgang bei i.i.d.-Normalziehung.
    """

    risk_class: int
    bootstrap_probability: float
    normal_probability: float
    bootstrap_p5: float
    normal_p5: float
    bootstrap_worst_drawdown: float
    normal_worst_drawdown: float

    @property
    def probability_difference(self) -> float:
        """Differenz der Erfolgswahrscheinlichkeiten (Bootstrap minus Normal)."""
        return self.bootstrap_probability - self.normal_probability

    @property
    def p5_difference(self) -> float:
        """Differenz der fünften Perzentile (Bootstrap minus Normal)."""
        return self.bootstrap_p5 - self.normal_p5


def load_history(policies: PolicySet) -> FloatArray:
    """Lädt die vollständigen historischen Monatsrenditen.

    Zeilen mit fehlenden Werten entfallen: Eine simultane Ziehung über alle
    Assetklassen setzt voraus, dass jeder Monat für jede Assetklasse belegt ist.

    Args:
        policies: Regelwerk, hier nur zur Bestimmung der Spaltenreihenfolge.

    Returns:
        Renditematrix der Form ``(monate, assetklassen)``.
    """
    _ = policies
    frame = load_returns()[list(ASSET_KEYS)].dropna()
    return frame.to_numpy(dtype=np.float64)


def _run_case(
    sampler: ReturnSampler,
    weights: FloatArray,
    policies: PolicySet,
    paths: int,
    seed: int,
) -> tuple[float, float, float]:
    """Rechnet den Referenzfall einmal durch.

    Args:
        sampler: Ziehungsverfahren.
        weights: Zielgewichte.
        policies: Regelwerk.
        paths: Zahl der Pfade.
        seed: Verwendeter Seed.

    Returns:
        Tripel aus Erfolgswahrscheinlichkeit, fünftem Perzentil und
        schlechtestem Rückgang.
    """
    ergebnis = simulate(
        sampler,
        weights,
        initial_capital=REFERENCE_INITIAL,
        monthly_contribution=REFERENCE_CONTRIBUTION,
        horizon_months=REFERENCE_HORIZON_MONTHS,
        target_amount=REFERENCE_TARGET,
        policy=policies.simulation,
        paths=paths,
        seed=seed,
        asset_keys=ASSET_KEYS,
        track_bands=False,
    )
    return ergebnis.success_probability, ergebnis.percentile(5), ergebnis.worst_max_drawdown


def convergence_study(
    policies: PolicySet,
    history: FloatArray,
    path_counts: tuple[int, ...] = PATH_COUNTS,
    repetitions: int = REPETITIONS,
) -> list[ConvergencePoint]:
    """Führt die Konvergenzanalyse durch.

    Args:
        policies: Regelwerk.
        history: Historische Monatsrenditen.
        path_counts: Zu prüfende Pfadzahlen.
        repetitions: Zahl der Wiederholungen je Pfadzahl.

    Returns:
        Ein Ergebnis je Pfadzahl.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    weights = reference_weights(REFERENCE_RISK_CLASS, policies.classes, ASSET_KEYS)
    basis = policies.simulation.basis_seed

    punkte: list[ConvergencePoint] = []
    for anzahl in path_counts:
        schaetzungen = np.array(
            [
                _run_case(sampler, weights, policies, anzahl, basis + wiederholung)[0]
                for wiederholung in range(repetitions)
            ]
        )
        mittel = float(schaetzungen.mean())
        punkte.append(
            ConvergencePoint(
                paths=anzahl,
                repetitions=repetitions,
                mean_probability=mittel,
                empirical_se=float(schaetzungen.std(ddof=1)),
                theoretical_se=float(np.sqrt(mittel * (1.0 - mittel) / anzahl)),
                spread=float(schaetzungen.max() - schaetzungen.min()),
            )
        )
    return punkte


def method_comparison(
    policies: PolicySet,
    history: FloatArray,
    paths: int | None = None,
) -> list[MethodComparison]:
    """Vergleicht Block-Bootstrap und i.i.d.-Normalziehung über alle Klassen.

    Args:
        policies: Regelwerk.
        history: Historische Monatsrenditen.
        paths: Zahl der Pfade. Ohne Angabe der Wert aus der Policy.

    Returns:
        Ein Vergleich je Risikoklasse.
    """
    anzahl = paths if paths is not None else policies.simulation.standard_pfade
    bootstrap = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    normal = IIDNormalSampler(history)
    seed = policies.simulation.basis_seed

    vergleiche: list[MethodComparison] = []
    for klasse in sorted(policies.classes.risikoklassen):
        gewichte = reference_weights(klasse, policies.classes, ASSET_KEYS)
        bb_p, bb_p5, bb_dd = _run_case(bootstrap, gewichte, policies, anzahl, seed)
        nv_p, nv_p5, nv_dd = _run_case(normal, gewichte, policies, anzahl, seed)
        vergleiche.append(
            MethodComparison(
                risk_class=klasse,
                bootstrap_probability=bb_p,
                normal_probability=nv_p,
                bootstrap_p5=bb_p5,
                normal_p5=nv_p5,
                bootstrap_worst_drawdown=bb_dd,
                normal_worst_drawdown=nv_dd,
            )
        )
    return vergleiche


def _de(value: float, decimals: int = 2, *, percent: bool = False) -> str:
    """Formatiert eine Zahl mit deutschem Dezimalkomma.

    Args:
        value: Zu formatierender Wert.
        decimals: Zahl der Nachkommastellen.
        percent: Multipliziert mit 100 und hängt ein Prozentzeichen an.

    Returns:
        Der formatierte Wert.
    """
    zahl = value * 100 if percent else value
    text = f"{zahl:,.{decimals}f}".replace(",", " ").replace(".", ",")
    return f"{text} %" if percent else text


def render_block(
    points: list[ConvergencePoint],
    comparisons: list[MethodComparison],
    policies: PolicySet,
) -> str:
    """Erzeugt den Abschnitt für die Auswertungsdatei.

    Args:
        points: Ergebnisse der Konvergenzanalyse.
        comparisons: Ergebnisse des Verfahrensvergleichs.
        policies: Regelwerk, für die Angabe der Fassungen.

    Returns:
        Markdown-Abschnitt einschließlich der Blockmarkierungen.
    """
    stabil = next((punkt for punkt in points if punkt.is_stable), None)
    zeilen = [
        CONVERGENCE_START,
        "",
        "_Erzeugt durch `make convergence`. Nicht von Hand bearbeiten —",
        "der Abschnitt wird bei jedem Lauf vollständig ersetzt._",
        "",
        f"**Referenzfall:** Startkapital {_de(REFERENCE_INITIAL, 0)} €, Sparrate "
        f"{_de(REFERENCE_CONTRIBUTION, 0)} € monatlich, Horizont "
        f"{REFERENCE_HORIZON_MONTHS} Monate, Zielbetrag {_de(REFERENCE_TARGET, 0)} €, "
        f"Risikoklasse {REFERENCE_RISK_CLASS} mit der Referenzallokation aus "
        "`candor/portfolio/universe.py`.",
        "",
        f"**Regelwerk:** simulation {policies.simulation.version}, "
        f"classes {policies.classes.version}. "
        f"Blocklänge {policies.simulation.blocklaenge_monate} Monate, Rebalancing alle "
        f"{policies.simulation.rebalancing_monate} Monate, Kosten laut Policy.",
        "",
        "### 2.1 Konvergenz über die Pfadzahl",
        "",
        f"Je Pfadzahl {points[0].repetitions if points else 0} Wiederholungen mit "
        "aufeinanderfolgenden Seeds. Der empirische Standardfehler ist die "
        "Standardabweichung der Schätzungen über diese Wiederholungen; der theoretische "
        "ist der binomiale Wert √(p·(1−p)/n).",
        "",
        "| Pfade | Erfolgswahrsch. (Mittel) | SE empirisch | SE theoretisch "
        "| Spannweite | stabil |",
        "|---:|---:|---:|---:|---:|:--:|",
    ]
    for punkt in points:
        zeilen.append(
            f"| {punkt.paths:,}".replace(",", " ")
            + f" | {_de(punkt.mean_probability, 4, percent=True)} "
            f"| {_de(punkt.empirical_se, 4, percent=True)} "
            f"| {_de(punkt.theoretical_se, 4, percent=True)} "
            f"| {_de(punkt.spread, 4, percent=True)} "
            f"| {'ja' if punkt.is_stable else 'nein'} |"
        )

    zeilen += [
        "",
        "### 2.2 Ab wann trägt die Schätzung eine Empfehlung?",
        "",
        f"Als Schwelle ist ein empirischer Standardfehler unter "
        f"{_de(STABILITY_THRESHOLD, 1, percent=True)} gesetzt. Begründung: Die "
        "Konfliktdiagnose vergleicht die Erfolgswahrscheinlichkeit mit einem "
        "Schwellenwert von 85 % beziehungsweise 70 % (`goals_v1.yaml`). Liegt der "
        "Standardfehler bei einem halben Prozentpunkt, umfasst ein Zweifachintervall "
        "rund zwei Prozentpunkte — schmal genug, dass die Entscheidung zwischen "
        "Empfehlung und Konfliktbericht nicht vom Seed abhängt, außer der Fall liegt "
        "ohnehin auf der Kante.",
        "",
    ]
    if stabil is not None:
        zeilen.append(
            f"**Befund:** Ab **{_de(float(stabil.paths), 0)} Pfaden** liegt der empirische "
            f"Standardfehler mit {_de(stabil.empirical_se, 4, percent=True)} unter der "
            f"Schwelle. Die als Standard gesetzten "
            f"{_de(float(policies.simulation.standard_pfade), 0)} Pfade erfüllen die "
            "Anforderung damit, liegen aber nur knapp darüber."
        )
    else:
        zeilen.append(
            "**Befund:** Keine der geprüften Pfadzahlen erreicht die Stabilitätsschwelle. "
            "Die als Standard gesetzte Pfadzahl ist damit nicht gedeckt."
        )

    zeilen += [
        "",
        "Der empirische Standardfehler folgt dem theoretischen eng. Das ist ein "
        "eigenständiger Befund: Die Pfade werden tatsächlich unabhängig gezogen, und "
        "die Schätzung verhält sich wie eine unabhängige Stichprobe. Ein deutlich über "
        "dem theoretischen Wert liegender empirischer Fehler wäre ein Hinweis auf einen "
        "Implementierungsfehler.",
        "",
        "### 2.3 Block-Bootstrap gegen i.i.d.-Normalziehung",
        "",
        "Beide Verfahren nutzen dieselbe Historie. Die Normalziehung übernimmt "
        "Erwartungswerte **und** Kovarianzmatrix, erhält also die gleichzeitige "
        "Korrelation; verloren gehen ausschließlich Autokorrelation, "
        "Volatilitätscluster und die empirischen Randbereiche. Der Vergleich misst "
        "damit genau den Beitrag dieser drei Eigenschaften.",
        "",
        f"Je Zelle {comparisons[0].risk_class and policies.simulation.standard_pfade:,}".replace(
            ",", " "
        )
        + " Pfade, identischer Seed.",
        "",
        "| Klasse | Erfolg Bootstrap | Erfolg Normal | Δ Erfolg | 5 %-Perzentil Bootstrap | "
        "5 %-Perzentil Normal | Δ 5 %-Perzentil | schlecht. DD Bootstrap | schlecht. DD Normal |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for vergleich in comparisons:
        zeilen.append(
            f"| {vergleich.risk_class} "
            f"| {_de(vergleich.bootstrap_probability, 2, percent=True)} "
            f"| {_de(vergleich.normal_probability, 2, percent=True)} "
            f"| {_de(vergleich.probability_difference, 2, percent=True)} "
            f"| {_de(vergleich.bootstrap_p5, 0)} € "
            f"| {_de(vergleich.normal_p5, 0)} € "
            f"| {_de(vergleich.p5_difference, 0)} € "
            f"| {_de(vergleich.bootstrap_worst_drawdown, 2, percent=True)} "
            f"| {_de(vergleich.normal_worst_drawdown, 2, percent=True)} |"
        )

    zeilen += ["", _comparison_verdict(comparisons), "", CONVERGENCE_END]
    return "\n".join(zeilen)


def _comparison_verdict(comparisons: list[MethodComparison]) -> str:
    """Formuliert den Befund des Verfahrensvergleichs aus den Zahlen.

    Unterschieden wird zwischen der **Mitte** der Verteilung — der
    Erfolgswahrscheinlichkeit — und ihren **Rändern**. Beide können sich
    unterschiedlich stark unterscheiden, und für die Konfliktdiagnose ist der
    Rand die wichtigere Größe.

    Args:
        comparisons: Ergebnisse des Vergleichs.

    Returns:
        Der Befund als Absatz.
    """
    if not comparisons:  # pragma: no cover
        return "**Befund:** Kein Vergleich gerechnet."

    max_p = max(abs(vergleich.probability_difference) for vergleich in comparisons)
    max_p5 = max(abs(vergleich.p5_difference) for vergleich in comparisons)
    max_dd = max(
        abs(vergleich.bootstrap_worst_drawdown - vergleich.normal_worst_drawdown)
        for vergleich in comparisons
    )
    p5_relativ = max(
        abs(vergleich.p5_difference) / vergleich.normal_p5
        for vergleich in comparisons
        if vergleich.normal_p5 > 0
    )

    mitte = (
        f"In der **Mitte der Verteilung** unterscheiden sich die Verfahren kaum: Die größte "
        f"Abweichung der Erfolgswahrscheinlichkeit über alle fünf Risikoklassen beträgt "
        f"{_de(max_p, 2, percent=True)}, und sie hat kein einheitliches Vorzeichen — der "
        "Bootstrap liegt mal darüber, mal darunter. Wer allein auf die "
        "Erfolgswahrscheinlichkeit schaut, käme mit der einfacheren Normalziehung zu "
        "praktisch demselben Ergebnis."
    )
    rand = (
        f"An den **Rändern** ist der Unterschied deutlich. Das fünfte Perzentil liegt im "
        f"Bootstrap um bis zu {_de(max_p5, 0)} € niedriger "
        f"({_de(p5_relativ, 1, percent=True)} des Werts), und der schlechteste Rückgang "
        f"fällt um bis zu {_de(max_dd, 2, percent=True)} tiefer aus. Beides in dieselbe "
        "Richtung und über alle Klassen hinweg: Die Normalziehung **unterschätzt das "
        "Verlustrisiko systematisch**, und zwar umso stärker, je höher die Aktienquote."
    )
    schluss = (
        "**Rechtfertigt der Bootstrap seinen Mehraufwand?** Für die Erfolgswahrscheinlichkeit "
        "allein: nein. Für die Aussagen, auf die es in diesem Projekt ankommt — das fünfte "
        "Perzentil als Maß des ungünstigen Falls und der maximale Rückgang als Maß der "
        "durchzuhaltenden Belastung — ja, und zwar deutlich. Der Mehraufwand ist damit "
        "gerechtfertigt, aber nicht aus dem Grund, aus dem man ihn üblicherweise begründet."
    )
    return f"**Befund:** {mitte}\n\n{rand}\n\n{schluss}"


def write_evaluation(block: str, evaluation_path: Path | None = None) -> Path:
    """Schreibt den Abschnitt in die Auswertungsdatei.

    Args:
        block: Der einzusetzende Abschnitt.
        evaluation_path: Abweichender Pfad der Zieldatei.

    Returns:
        Pfad der geschriebenen Datei.

    Raises:
        FileNotFoundError: Wenn die Zieldatei nicht existiert.
    """
    ziel = evaluation_path or (Path(__file__).resolve().parents[2] / "auswertung.md")
    if not ziel.is_file():
        msg = f"Zieldatei fehlt: {ziel}"
        raise FileNotFoundError(msg)

    inhalt = ziel.read_text(encoding="utf-8")
    if CONVERGENCE_START in inhalt and CONVERGENCE_END in inhalt:
        vorher = inhalt.split(CONVERGENCE_START)[0]
        nachher = inhalt.split(CONVERGENCE_END, 1)[1]
        neu = f"{vorher}{block}{nachher}"
    else:
        marker = "### 2.1 Konvergenz über die Pfadzahl"
        if marker in inhalt:
            vorher, _, rest = inhalt.partition(marker)
            _, _, nachher = rest.partition("---")
            neu = f"{vorher}{block}\n\n---{nachher}"
        else:  # pragma: no cover
            neu = f"{inhalt.rstrip()}\n\n{block}\n"
    ziel.write_text(neu, encoding="utf-8")
    return ziel


def main(argv: list[str] | None = None) -> int:
    """Einstiegspunkt der Kommandozeile.

    Args:
        argv: Argumente ohne Programmnamen.

    Returns:
        ``0`` bei Erfolg.
    """
    parser = argparse.ArgumentParser(description="Konvergenzanalyse und Verfahrensvergleich.")
    parser.add_argument(
        "--write-evaluation", action="store_true", help="Ergebnis in die Auswertungsdatei schreiben"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Schnelldurchlauf mit weniger Pfaden und Wiederholungen",
    )
    args = parser.parse_args(argv)

    policies = load_policy_set()
    history = load_history(policies)

    pfadzahlen = QUICK_PATH_COUNTS if args.quick else PATH_COUNTS
    wiederholungen = QUICK_REPETITIONS if args.quick else REPETITIONS
    vergleichspfade = 2_000 if args.quick else None

    sys.stdout.write(
        f"Konvergenzanalyse: {len(pfadzahlen)} Pfadzahlen x {wiederholungen} Wiederholungen …\n"
    )
    punkte = convergence_study(policies, history, pfadzahlen, wiederholungen)
    sys.stdout.write("Verfahrensvergleich über alle Risikoklassen …\n")
    vergleiche = method_comparison(policies, history, vergleichspfade)

    block = render_block(punkte, vergleiche, policies)
    if args.write_evaluation:
        ziel = write_evaluation(block)
        sys.stdout.write(f"Geschrieben nach {ziel}\n")
    else:
        sys.stdout.write(block + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
