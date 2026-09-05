"""Erzeugung der Abbildungen zum System.

Gestaltungsregeln der Abbildungen, die hier durchgängig eingehalten werden:

* matplotlib, 300 dpi, ``bbox_inches="tight"``, PNG,
* Breite 5,2 in für einspaltige Einbindung,
* Palette Navy ``#1F3864``, Hellblau ``#DCE6F4``, Akzent ``#C00000``,
  Grau ``#7F7F7F``,
* deutsches Dezimalkomma auf allen Achsen,
* **kein Titel und keine Quellenzeile in der Grafik** — beides setzt Word darum
  herum,
* zu jeder Abbildung eine ``.caption.txt`` mit exakt zwei Zeilen.

Aufruf:
    python -m candor.report.figures [--seed N] [--out figures/]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter

from candor import format_de
from candor.data.loader import load_returns
from candor.data.policy_loader import load_policy_set
from candor.data.universe_meta import ASSET_KEYS
from candor.domain.goal import GoalType
from candor.domain.policy import PolicySet
from candor.domain.result import ConflictReport, Feasibility, LeverId
from candor.engine.convergence import (
    REFERENCE_CONTRIBUTION,
    REFERENCE_HORIZON_MONTHS,
    REFERENCE_INITIAL,
    REFERENCE_RISK_CLASS,
    REFERENCE_TARGET,
    load_history,
)
from candor.engine.simulate import BlockBootstrapSampler, SimulationResult, simulate
from candor.engine.stress import StressResult, run_all
from candor.goals.model import build_goal
from candor.portfolio.backtest import STRATEGY_EQUAL, STRATEGY_MVO, run_backtest
from candor.portfolio.benchmark import equal_weight_portfolio, sharpe_difference_test
from candor.portfolio.estimate import MomentEstimate, estimate_moments
from candor.portfolio.multiplicity import MultiplicityReport, adjust
from candor.portfolio.optimize import optimize
from candor.portfolio.universe import reference_weights
from candor.profiling.classify import DIVERGENCE_THRESHOLD
from candor.profiling.sensitivity import SensitivityReport, analyse, baseline_clients
from candor.report.factsheet import reference_conflict_case

#: Farbpalette der Abbildungen.
NAVY: Final = "#1F3864"
HELLBLAU: Final = "#DCE6F4"
AKZENT: Final = "#C00000"
GRAU: Final = "#7F7F7F"

#: Abbildungsbreite in Zoll bei einspaltiger Einbindung.
FIGURE_WIDTH_IN: Final = 5.2

#: Höhe der Abbildung in Zoll.
FIGURE_HEIGHT_IN: Final = 3.2

#: Auflösung der Ausgabe.
DPI: Final = 300

#: Schriftgröße der Abbildungen.
FONT_SIZE: Final = 10

#: Monate je Jahr.
MONTHS_PER_YEAR: Final = 12

#: Stichtag des Anlageziels in der Sensitivitätsanalyse (Abbildung 7).
SENSITIVITY_DATE: Final = date(2026, 7, 26)

#: Zahl der Stützstellen der Effizienzlinie in Abbildung 5.
FRONTIER_POINTS: Final = 60

#: Sättigungsanteil, ab dem die Zellenbeschriftung in Abbildung 2 auf Weiß
#: wechselt. Darunter bliebe weisser Text auf hellem Grund unlesbar.
LIGHT_TEXT_THRESHOLD: Final = 0.45

#: Kurzbezeichnung je Stellschraube für die Feldbeschriftung in Abbildung 4.
LEVER_SHORT_NAMES: Final[dict[LeverId, str]] = {
    LeverId.SPARRATE: "Sparrate",
    LeverId.HORIZONT: "Horizont",
    LeverId.ZIELBETRAG: "Zielbetrag",
}


def apply_style() -> None:
    """Setzt die Darstellungsvorgaben des Projekts.

    Liberation Sans ist die vorgegebene Schrift; ist sie nicht installiert,
    greift matplotlib auf DejaVu Sans zurück. Beide sind metrisch ähnlich, der
    Rückfall ist deshalb unkritisch und ausdrücklich vorgesehen.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 1,
            "axes.edgecolor": GRAU,
            "axes.linewidth": 0.8,
            "grid.color": GRAU,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.35,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def german_thousands(value: float, _position: int = 0) -> str:
    """Formatiert einen Achsenwert mit deutschem Tausenderpunkt.

    Args:
        value: Achsenwert.
        _position: Von matplotlib übergebene Position, hier ohne Bedeutung.

    Returns:
        Der formatierte Wert, etwa ``"250 000"``.
    """
    return format_de.integer(value)


def german_decimal(value: float, _position: int = 0, decimals: int = 1) -> str:
    """Formatiert einen Achsenwert mit deutschem Dezimalkomma.

    Args:
        value: Achsenwert.
        _position: Von matplotlib übergebene Position, hier ohne Bedeutung.
        decimals: Zahl der Nachkommastellen.

    Returns:
        Der formatierte Wert, etwa ``"12,5"``.
    """
    return format_de.decimal(value, decimals)


def german_percent(value: float, _position: int = 0) -> str:
    """Formatiert einen Achsenwert als Prozentangabe.

    Args:
        value: Achsenwert als Anteil zwischen 0 und 1.
        _position: Von matplotlib uebergebene Position, hier ohne Bedeutung.

    Returns:
        Der formatierte Wert, etwa ``"7,5 %"``.
    """
    return format_de.percent(value, 1)


def write_caption(path: Path, number: int, title: str) -> Path:
    """Schreibt die zweizeilige Bildunterschrift.

    Args:
        path: Pfad der Bilddatei; die Unterschrift erhält dieselbe Basis.
        number: Nummer der Abbildung.
        title: Titel der Abbildung.

    Returns:
        Pfad der geschriebenen Unterschrift.
    """
    ziel = path.with_suffix(".caption.txt")
    ziel.write_text(f"Abb. {number}: {title}\nQuelle: Eigene Darstellung.\n", encoding="utf-8")
    return ziel


def fan_chart(result: SimulationResult, output: Path) -> Path:
    """Zeichnet das Fächerdiagramm der Simulation (Abbildung 3).

    Dargestellt sind drei Perzentilbänder um den Median sowie der Zielbetrag als
    Referenzlinie. Die Darstellung beantwortet die Frage, die eine einzelne
    Endwertzahl nicht beantworten kann: **wie breit** der mögliche Ausgang ist
    und **wann** sich die Spreizung aufbaut.

    Args:
        result: Simulationsergebnis mit Perzentilbändern.
        output: Pfad der zu schreibenden PNG-Datei.

    Returns:
        Pfad der geschriebenen Bilddatei.

    Raises:
        ValueError: Wenn das Ergebnis keine Perzentilbänder enthält.
    """
    if not result.value_bands:
        msg = (
            "Das Simulationsergebnis enthält keine Perzentilbänder. Für das "
            "Fächerdiagramm ist die Simulation mit track_bands=True zu rechnen."
        )
        raise ValueError(msg)

    apply_style()
    jahre = np.arange(result.horizon_months + 1) / MONTHS_PER_YEAR
    band = result.value_bands

    figur, achse = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))

    achse.fill_between(
        jahre, band[5], band[95], color=HELLBLAU, alpha=0.75, linewidth=0, label="5 % bis 95 %"
    )
    achse.fill_between(
        jahre, band[25], band[75], color=NAVY, alpha=0.22, linewidth=0, label="25 % bis 75 %"
    )
    achse.plot(jahre, band[50], color=NAVY, linewidth=1.6, label="Median")
    achse.axhline(
        result.target_amount,
        color=AKZENT,
        linewidth=1.2,
        linestyle="--",
        label="Zielbetrag",
    )

    achse.set_xlabel("Jahre ab Anlagebeginn")
    achse.set_ylabel("Portfoliowert in Euro")
    achse.set_xlim(0, float(jahre[-1]))
    achse.set_ylim(0, float(band[95][-1]) * 1.05)
    achse.xaxis.set_major_formatter(FuncFormatter(german_decimal))
    achse.yaxis.set_major_formatter(FuncFormatter(german_thousands))
    achse.grid(visible=True, axis="y")
    achse.spines["top"].set_visible(False)
    achse.spines["right"].set_visible(False)
    # Unten rechts: Der Faecher steigt, dieser Bereich bleibt frei. Oben links
    # wuerde die Legende die Zielbetragslinie kreuzen.
    achse.legend(loc="lower right", frameon=False, labelspacing=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    figur.savefig(output, format="png")
    plt.close(figur)
    return output


def lever_chart(report: ConflictReport, output: Path) -> Path:
    """Zeichnet die drei Stellschrauben mit Ist- und Sollwert (Abbildung 4).

    Drei getrennte Felder statt eines gemeinsamen Diagramms: Die Stellschrauben
    haben verschiedene Einheiten — Euro je Monat, Monate, Euro — und wären auf
    einer gemeinsamen Achse nicht vergleichbar, sondern irreführend.

    Nicht umsetzbare Stellschrauben werden in der Akzentfarbe gezeichnet. Sie
    verschwinden nicht aus der Abbildung: Der Kunde soll sehen, was nötig wäre,
    auch wenn es nicht geht.

    Args:
        report: Der Konfliktbericht mit den drei Stellschrauben.
        output: Pfad der zu schreibenden PNG-Datei.

    Returns:
        Pfad der geschriebenen Bilddatei.

    Raises:
        ValueError: Wenn der Bericht keine Stellschrauben enthält.
    """
    if not report.levers:
        msg = "Der Konfliktbericht enthält keine Stellschrauben."
        raise ValueError(msg)

    apply_style()
    figur, achsen = plt.subplots(
        len(report.levers), 1, figsize=(FIGURE_WIDTH_IN, 1.15 * len(report.levers) + 0.5)
    )
    felder = achsen if isinstance(achsen, np.ndarray) else np.array([achsen])

    for achse, hebel in zip(felder, report.levers, strict=True):
        ist = float(hebel.current_value)
        soll = float(hebel.required_value)
        umsetzbar = hebel.feasibility is Feasibility.UMSETZBAR
        sollfarbe = NAVY if umsetzbar else AKZENT

        achse.barh([1], [ist], color=HELLBLAU, edgecolor=GRAU, linewidth=0.6, height=0.6)
        achse.barh([0], [soll], color=sollfarbe, height=0.6)
        achse.set_yticks([0, 1])
        achse.set_yticklabels(["soll", "ist"])
        achse.set_xlim(0, max(ist, soll) * 1.35)
        achse.set_xticks([])
        achse.spines["top"].set_visible(False)
        achse.spines["right"].set_visible(False)
        achse.spines["bottom"].set_visible(False)

        einheit = "Monate" if hebel.lever_id is LeverId.HORIZONT else "€"
        stellen = 0
        achse.text(
            ist * 1.03,
            1,
            f"{german_decimal(ist, decimals=stellen)} {einheit}",
            va="center",
            fontsize=FONT_SIZE - 1,
            color=GRAU,
        )
        achse.text(
            soll * 1.03,
            0,
            f"{german_decimal(soll, decimals=stellen)} {einheit}",
            va="center",
            fontsize=FONT_SIZE - 1,
            color=sollfarbe,
            fontweight="bold",
        )
        kurzname = LEVER_SHORT_NAMES[hebel.lever_id]
        vermerk = "" if umsetzbar else "\n(nicht umsetzbar)"
        achse.set_ylabel(
            f"{hebel.lever_id.value}\n{kurzname}{vermerk}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
        )

    figur.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figur.savefig(output, format="png")
    plt.close(figur)
    return output


def efficient_frontier(
    moments: MomentEstimate,
    policies: PolicySet,
    output: Path,
    points: int = FRONTIER_POINTS,
) -> Path:
    """Zeichnet die Effizienzlinie mit den Klassenportfolios und 1/N (Abbildung 5).

    Die Effizienzlinie entsteht durch Variation der Zielrendite zwischen dem
    kleinsten und dem größten erreichbaren Wert **ohne** Klassenrestriktionen,
    aber mit dem Leerverkaufsverbot. Sie ist damit die theoretische Obergrenze;
    die fünf Klassenportfolios liegen zwangsläufig darunter, weil ihre Bänder
    zusätzlich binden. Genau dieser Abstand ist die Aussage der Abbildung: Er
    ist der Preis der Restriktionen — und zugleich der Schutz vor der
    Fehlerverstärkung der unrestringierten Optimierung.

    Args:
        moments: Geschätzte Momente.
        policies: Vollständiges Regelwerk.
        output: Zielpfad der PNG-Datei.
        points: Zahl der Stützstellen der Effizienzlinie.

    Returns:
        Der Pfad der geschriebenen Datei.
    """
    apply_style()
    figur, achse = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN + 0.4))

    kurve_vol, kurve_rendite = _frontier_points(moments, points)
    achse.plot(
        kurve_vol,
        kurve_rendite,
        color=GRAU,
        linewidth=1.3,
        linestyle="--",
        label="Effizienzlinie ohne Klassenbänder",
        zorder=2,
    )

    klassen_vol: list[float] = []
    klassen_rendite: list[float] = []
    for klasse in sorted(policies.classes.risikoklassen):
        referenz = reference_weights(klasse, policies.classes, ASSET_KEYS)
        ergebnis = optimize(
            moments,
            klasse,
            policies.classes,
            policies.portfolio,
            ASSET_KEYS,
            target_return=float(referenz @ moments.mean),
        )
        klassen_vol.append(ergebnis.volatility * np.sqrt(MONTHS_PER_YEAR))
        klassen_rendite.append((1.0 + ergebnis.expected_return) ** MONTHS_PER_YEAR - 1.0)

    achse.plot(
        klassen_vol,
        klassen_rendite,
        color=NAVY,
        linewidth=1.6,
        marker="o",
        markersize=5,
        label="Portfolios der Risikoklassen 1–5",
        zorder=3,
    )
    for klasse, x, y in zip(
        sorted(policies.classes.risikoklassen), klassen_vol, klassen_rendite, strict=True
    ):
        achse.annotate(
            str(klasse),
            (x, y),
            textcoords="offset points",
            xytext=(6, -9),
            color=NAVY,
            fontsize=FONT_SIZE - 1,
            fontweight="bold",
        )

    gleich = equal_weight_portfolio(len(ASSET_KEYS))
    achse.scatter(
        [moments.portfolio_volatility(gleich) * np.sqrt(MONTHS_PER_YEAR)],
        [(1.0 + moments.portfolio_return(gleich)) ** MONTHS_PER_YEAR - 1.0],
        color=AKZENT,
        s=55,
        marker="D",
        label="1/N-Benchmark",
        zorder=4,
    )

    achse.set_xlabel("Volatilität p. a.")
    achse.set_ylabel("Erwartete Rendite p. a.")
    achse.xaxis.set_major_formatter(FuncFormatter(german_percent))
    achse.yaxis.set_major_formatter(FuncFormatter(german_percent))
    achse.grid(visible=True, linestyle="-")
    achse.spines["top"].set_visible(False)
    achse.spines["right"].set_visible(False)
    achse.legend(loc="lower right", frameon=False)

    figur.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figur.savefig(output, format="png")
    plt.close(figur)
    return output


def _frontier_points(moments: MomentEstimate, points: int) -> tuple[list[float], list[float]]:
    """Berechnet die Stützstellen der unrestringierten Effizienzlinie.

    Restringiert wird allein durch Vollinvestition und Leerverkaufsverbot; die
    Klassenbänder bleiben außen vor. Nicht lösbare Stützstellen werden
    übersprungen — am oberen Rand der Renditeachse ist das der Normalfall.

    Args:
        moments: Geschätzte Momente.
        points: Zahl der Stützstellen.

    Returns:
        Annualisierte Volatilitäten und Renditen der lösbaren Stützstellen.
    """
    untergrenze = float(moments.mean.min())
    obergrenze = float(moments.mean.max())
    volatilitaeten: list[float] = []
    renditen: list[float] = []

    for ziel in np.linspace(untergrenze, obergrenze, points):
        gewichte = cp.Variable(len(moments.asset_keys))
        problem = cp.Problem(
            cp.Minimize(cp.quad_form(gewichte, cp.psd_wrap(moments.covariance))),
            [cp.sum(gewichte) == 1, gewichte >= 0, moments.mean @ gewichte >= ziel],
        )
        try:
            problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:  # pragma: no cover — am Rand der Kurve
            continue
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or gewichte.value is None:
            continue
        loesung = np.asarray(gewichte.value, dtype=np.float64)
        volatilitaeten.append(moments.portfolio_volatility(loesung) * np.sqrt(MONTHS_PER_YEAR))
        renditen.append((1.0 + moments.portfolio_return(loesung)) ** MONTHS_PER_YEAR - 1.0)

    return volatilitaeten, renditen


def build_figure_3(output_dir: Path, seed: int | None = None) -> tuple[Path, Path]:
    """Erzeugt Abbildung 3 aus dem eingefrorenen Datenstand.

    Args:
        output_dir: Zielverzeichnis.
        seed: Abweichender Seed. Ohne Angabe der Wert aus der Policy.

    Returns:
        Paar aus Bild- und Unterschriftpfad.
    """
    policies = load_policy_set()
    history = load_history(policies)
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    weights = reference_weights(REFERENCE_RISK_CLASS, policies.classes, ASSET_KEYS)

    ergebnis = simulate(
        sampler,
        weights,
        initial_capital=REFERENCE_INITIAL,
        monthly_contribution=REFERENCE_CONTRIBUTION,
        horizon_months=REFERENCE_HORIZON_MONTHS,
        target_amount=REFERENCE_TARGET,
        policy=policies.simulation,
        seed=seed,
        asset_keys=ASSET_KEYS,
        track_bands=True,
    )

    bild = fan_chart(ergebnis, output_dir / "abb3_faecherdiagramm.png")
    unterschrift = write_caption(
        bild,
        3,
        "Verteilung des Portfoliowerts über die Laufzeit (Block-Bootstrap, "
        "10 000 Pfade, Perzentilbänder)",
    )
    return bild, unterschrift


def build_figure_4(output_dir: Path, seed: int | None = None) -> tuple[Path, Path]:
    """Erzeugt Abbildung 4 aus dem Referenz-Konfliktfall.

    Args:
        output_dir: Zielverzeichnis.
        seed: Abweichender Seed.

    Returns:
        Paar aus Bild- und Unterschriftpfad.
    """
    bericht = reference_conflict_case(seed=seed).conflict_report
    if bericht is None:  # pragma: no cover — der Referenzfall ist ein Konflikt
        msg = "Der Referenzfall ergibt keinen Konflikt; Abbildung 4 ist nicht erzeugbar."
        raise ValueError(msg)

    bild = lever_chart(bericht, output_dir / "abb4_stellschrauben.png")
    unterschrift = write_caption(
        bild,
        4,
        "Konfliktfall: die drei Stellschrauben mit Ist- und Sollwert",
    )
    return bild, unterschrift


def build_figure_5(output_dir: Path, seed: int | None = None) -> tuple[Path, Path]:
    """Erzeugt Abbildung 5 aus dem eingefrorenen Datenstand.

    Args:
        output_dir: Zielverzeichnis.
        seed: Ohne Wirkung — die Optimierung enthält keine Zufallsquelle. Der
            Parameter existiert allein, damit alle Erzeuger dieselbe Signatur
            haben.

    Returns:
        Paar aus Bild- und Unterschriftpfad.
    """
    del seed
    policies = load_policy_set()
    history = load_history(policies)
    momente = estimate_moments(history, policies.portfolio.schaetzung, ASSET_KEYS)

    bild = efficient_frontier(momente, policies, output_dir / "abb5_effizienzlinie.png")
    unterschrift = write_caption(
        bild,
        5,
        "Effizienzlinie, Portfolios der Risikoklassen 1 bis 5 und der 1/N-Benchmark",
    )
    return bild, unterschrift


def build_figure_1(output_dir: Path, seed: int | None = None) -> tuple[Path, Path]:
    """Erzeugt Abbildung 1 — die Verarbeitungskette.

    Args:
        output_dir: Zielverzeichnis.
        seed: Ohne Wirkung; die Abbildung enthält keine Zufallsquelle.

    Returns:
        Paar aus Bild- und Unterschriftpfad.
    """
    del seed
    bild = process_chain(output_dir / "abb1_verarbeitungskette.png")
    unterschrift = write_caption(
        bild,
        1,
        "Verarbeitungskette von der Erhebung bis zum Dokument, mit der Verzweigung im Konfliktfall",
    )
    return bild, unterschrift


def build_figure_2(output_dir: Path, seed: int | None = None) -> tuple[Path, Path]:
    """Erzeugt Abbildung 2 — die Zwei-Achsen-Matrix.

    Args:
        output_dir: Zielverzeichnis.
        seed: Ohne Wirkung; die Abbildung enthält keine Zufallsquelle.

    Returns:
        Paar aus Bild- und Unterschriftpfad.
    """
    del seed
    bild = axis_matrix(load_policy_set(), output_dir / "abb2_achsenmatrix.png")
    unterschrift = write_caption(
        bild,
        2,
        "Ergebnisklasse als Minimum beider Achsen; in Klammern der Wert einer Mittelung, "
        "schraffiert die Divergenzzone",
    )
    return bild, unterschrift


def build_figure_6(output_dir: Path, seed: int | None = None) -> tuple[Path, Path]:
    """Erzeugt Abbildung 6 — die Stresstest-Drawdowns.

    Args:
        output_dir: Zielverzeichnis.
        seed: Ohne Wirkung; der Stresstest rechnet auf echten Renditen.

    Returns:
        Paar aus Bild- und Unterschriftpfad.
    """
    del seed
    policies = load_policy_set()
    ergebnisse = run_all(load_returns(), policies, ASSET_KEYS)
    bild = stress_chart(ergebnisse, output_dir / "abb6_stresstests.png")
    unterschrift = write_caption(
        bild,
        6,
        "Größter Rückgang je Krisenfenster und Risikoklasse gegen die zugesagte Verlusterwartung",
    )
    return bild, unterschrift


def multiplicity_chart(bericht: MultiplicityReport, output: Path) -> Path:
    """Zeichnet die p-Werte vor und nach Mehrfachvergleichskorrektur (Abbildung 8).

    Gruppierte Balken je Risikoklasse: unkorrigiert, Bonferroni,
    Holm-Bonferroni, Benjamini-Hochberg; dazu das Niveau von 5 % als
    waagerechte Referenzlinie.

    Die Abbildung zeigt, was die zugehörige Tabelle zwar enthält,
    aber nicht sichtbar macht: **wie weit** die beiden unkorrigiert knapp
    gesicherten Klassen nach der Korrektur von der Schwelle entfernt liegen. Sie
    ist das Bild zu Systemgrenze G-14.

    Args:
        bericht: Ergebnis der Korrektur über die fünf Klassentests.
        output: Pfad der zu schreibenden PNG-Datei.

    Returns:
        Pfad der geschriebenen Bilddatei.
    """
    apply_style()
    geordnet = sorted(bericht.results, key=lambda eintrag: int(eintrag.label))
    klassen = [eintrag.label for eintrag in geordnet]

    reihen: tuple[tuple[str, list[float], str], ...] = (
        ("unkorrigiert", [e.p_raw for e in geordnet], NAVY),
        ("Bonferroni", [e.p_bonferroni for e in geordnet], HELLBLAU),
        ("Holm-Bonferroni", [e.p_holm for e in geordnet], GRAU),
        ("Benjamini-Hochberg", [e.p_benjamini_hochberg for e in geordnet], "#9DB2D4"),
    )

    figur, achse = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.2))
    positionen = np.arange(len(klassen), dtype=float)
    breite = 0.20

    for lauf, (name, werte, farbe) in enumerate(reihen):
        versatz = (lauf - (len(reihen) - 1) / 2.0) * breite
        achse.bar(
            positionen + versatz,
            werte,
            width=breite,
            color=farbe,
            edgecolor=GRAU,
            linewidth=0.5,
            label=name,
            zorder=2,
        )

    achse.axhline(bericht.level, color=AKZENT, linewidth=1.2, linestyle="--", zorder=3)
    achse.annotate(
        f"Niveau {format_de.percent(bericht.level, 0)}",
        xy=(len(klassen) - 0.5, bericht.level),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=AKZENT,
        fontsize=FONT_SIZE - 3,
    )

    achse.set_xticks(positionen)
    achse.set_xticklabels([f"Klasse {kennung}" for kennung in klassen])
    achse.set_ylabel("p-Wert")
    achse.set_ylim(0.0, 1.08)
    achse.yaxis.set_major_formatter(FuncFormatter(lambda wert, _: german_decimal(wert, decimals=2)))
    achse.grid(visible=True, axis="y")
    achse.set_axisbelow(True)
    for rand in ("top", "right"):
        achse.spines[rand].set_visible(False)
    achse.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE - 3,
    )

    figur.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figur.savefig(output, format="png")
    plt.close(figur)
    return output


def build_figure_8(output_dir: Path, seed: int | None = None) -> tuple[Path, Path]:
    """Erzeugt Abbildung 8 — p-Werte vor und nach Mehrfachvergleichskorrektur.

    Die p-Werte werden aus demselben Backtest neu gerechnet, aus dem auch
    die Auswertungstabelle entsteht – nicht aus der Dokumentation gelesen.
    Dadurch können Abbildung und Tabelle nicht auseinanderlaufen.

    Args:
        output_dir: Zielverzeichnis.
        seed: Ohne Wirkung. Der parametrische Test enthält keine Zufallsquelle;
            der Bootstrap-Test geht in diese Abbildung nicht ein.

    Returns:
        Paar aus Bild- und Unterschriftpfad.
    """
    del seed
    policies = load_policy_set()
    renditen = load_returns().dropna().to_numpy(dtype=float)

    rohe: dict[str, float] = {}
    for klasse in sorted(policies.classes.risikoklassen):
        lauf = run_backtest(renditen, policies, ASSET_KEYS, risk_class=klasse)
        rohe[str(klasse)] = sharpe_difference_test(
            lauf.strategies[STRATEGY_MVO].monthly_returns,
            lauf.strategies[STRATEGY_EQUAL].monthly_returns,
            lauf.risk_free_monthly,
            MONTHS_PER_YEAR,
        ).p_value

    bild = multiplicity_chart(adjust(rohe), output_dir / "abb8_mehrfachvergleich.png")
    unterschrift = write_caption(
        bild,
        8,
        "p-Werte des Vergleichs MVO gegen 1/N je Risikoklasse, vor und nach "
        "Korrektur für multiples Testen",
    )
    return bild, unterschrift


def build_figure_7(output_dir: Path, seed: int | None = None) -> tuple[Path, Path]:
    """Erzeugt Abbildung 7 — den Hebel je Fragebogen-Item.

    Args:
        output_dir: Zielverzeichnis.
        seed: Ohne Wirkung; die Analyse enthält keine Zufallsquelle.

    Returns:
        Paar aus Bild- und Unterschriftpfad.
    """
    del seed
    policies = load_policy_set()
    anlageziel = build_goal(
        "Z-SENS-A",
        GoalType.ALTERSVORSORGE,
        Decimal("300000.00"),
        SENSITIVITY_DATE,
        policies.goals,
        initial_capital=Decimal("20000.00"),
        monthly_contribution=Decimal("500.00"),
    )
    berichte = [
        analyse(kunde, anlageziel, policies, name) for name, kunde in baseline_clients().items()
    ]
    bild = tornado_chart(berichte, output_dir / "abb7_sensitivitaet.png")
    unterschrift = write_caption(
        bild,
        7,
        "Hebel einzelner Fragebogenangaben auf die Risikoklasse, größter Wert "
        "über drei Basisprofile",
    )
    return bild, unterschrift


def main(argv: list[str] | None = None) -> int:
    """Einstiegspunkt der Kommandozeile.

    Args:
        argv: Argumente ohne Programmnamen.

    Returns:
        ``0`` bei Erfolg.
    """
    parser = argparse.ArgumentParser(description="Abbildungen erzeugen.")
    parser.add_argument("--seed", type=int, default=None, help="abweichender Seed")
    parser.add_argument("--out", type=Path, default=Path("figures"), help="Zielverzeichnis")
    parser.add_argument(
        "--nur",
        type=int,
        nargs="*",
        choices=range(1, 9),
        help="nur diese Abbildungen erzeugen; ohne Angabe alle",
    )
    args = parser.parse_args(argv)

    erzeuger_je_nummer = {
        1: build_figure_1,
        2: build_figure_2,
        3: build_figure_3,
        4: build_figure_4,
        5: build_figure_5,
        6: build_figure_6,
        7: build_figure_7,
        8: build_figure_8,
    }
    gewaehlt = args.nur or sorted(erzeuger_je_nummer)
    for nummer in gewaehlt:
        erzeuger = erzeuger_je_nummer[nummer]
        bild, unterschrift = erzeuger(args.out, args.seed)
        sys.stdout.write(f"Abbildung {nummer}: {bild}\n")
        sys.stdout.write(f"  Unterschrift: {unterschrift}\n")
    return 0


def process_chain(output: Path) -> Path:
    """Zeichnet die fünfstufige Verarbeitungskette (Abbildung 1).

    Die Abbildung zeigt nicht nur die Reihenfolge, sondern auch die Stelle, an
    der sich der Pfad **teilt**: Nach dem Zielabgleich entscheidet sich, ob eine
    Empfehlung ergeht oder ein Konfliktbericht. Das ist die einzige Verzweigung
    des Systems und zugleich seine fachliche Besonderheit.

    Args:
        output: Zielpfad der PNG-Datei.

    Returns:
        Der Pfad der geschriebenen Datei.
    """
    apply_style()
    figur, achse = plt.subplots(figsize=(FIGURE_WIDTH_IN, 2.6))

    stufen = (
        ("1  Erhebung", "Haushalt, Ziel,\nBereitschaft,\nKenntnisse"),
        ("2  Profilierung", "drei Achsen,\nbindendes\nMinimum"),
        ("3  Zielabgleich", "Block-Bootstrap,\n10 000 Pfade"),
        ("4  Portfolio", "restringierte\nOptimierung"),
        ("5  Dokument", "Geeignetheits-\nerklärung"),
    )
    breite, hoehe, abstand = 1.80, 0.98, 0.28
    gesamt = len(stufen) * breite + (len(stufen) - 1) * abstand
    unten = 1.25

    achse.set_xlim(-0.05, gesamt + 0.05)
    achse.set_ylim(0, unten + hoehe + 0.1)
    achse.axis("off")

    for i, (titel, inhalt) in enumerate(stufen):
        links = i * (breite + abstand)
        letzte = i == len(stufen) - 1
        farbe = NAVY if letzte else HELLBLAU
        textfarbe = "white" if letzte else NAVY
        achse.add_patch(
            Rectangle((links, unten), breite, hoehe, facecolor=farbe, edgecolor=NAVY, linewidth=0.9)
        )
        achse.text(
            links + breite / 2,
            unten + hoehe * 0.76,
            titel,
            ha="center",
            va="center",
            fontsize=FONT_SIZE - 3,
            fontweight="bold",
            color=textfarbe,
        )
        achse.text(
            links + breite / 2,
            unten + hoehe * 0.33,
            inhalt,
            ha="center",
            va="center",
            fontsize=FONT_SIZE - 4.5,
            color=textfarbe,
            linespacing=1.25,
        )
        if not letzte:
            achse.annotate(
                "",
                xy=(links + breite + abstand, unten + hoehe / 2),
                xytext=(links + breite, unten + hoehe / 2),
                arrowprops={"arrowstyle": "-|>", "color": NAVY, "linewidth": 1.0},
            )

    # Die Verzweigung nach Stufe 3 — die einzige des Systems.
    mitte_x = 2 * (breite + abstand) + breite / 2
    kasten_breite = breite * 1.75
    kasten_hoehe = 0.62
    achse.add_patch(
        Rectangle(
            (mitte_x - kasten_breite / 2, 0.18),
            kasten_breite,
            kasten_hoehe,
            facecolor="white",
            edgecolor=AKZENT,
            linewidth=1.1,
        )
    )
    achse.text(
        mitte_x,
        0.18 + kasten_hoehe / 2,
        "Konflikt: keine Empfehlung,\ndrei Stellschrauben",
        ha="center",
        va="center",
        fontsize=FONT_SIZE - 4,
        color=AKZENT,
        fontweight="bold",
        linespacing=1.25,
    )
    achse.annotate(
        "",
        xy=(mitte_x, 0.18 + kasten_hoehe),
        xytext=(mitte_x, unten),
        arrowprops={"arrowstyle": "-|>", "color": AKZENT, "linewidth": 1.0},
    )
    achse.text(
        mitte_x + 0.10,
        (unten + 0.18 + kasten_hoehe) / 2,
        "Ziel nicht erreichbar",
        ha="left",
        va="center",
        fontsize=FONT_SIZE - 4,
        color=AKZENT,
    )

    figur.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figur.savefig(output, format="png")
    plt.close(figur)
    return output


def axis_matrix(policies: PolicySet, output: Path) -> Path:
    """Zeichnet die Zwei-Achsen-Matrix mit Konfliktzone (Abbildung 2).

    **Das Kernbild des Systems.** Es zeigt in einem Blick, was die Arbeit von
    einem üblichen Robo-Advisor unterscheidet:

    * Jede Zelle trägt die **Ergebnisklasse** — das Minimum beider Achsen, nicht
      ihren Mittelwert. Die Zahl in der Zelle ist damit die Antwort auf die
      Frage, warum nicht gemittelt wird: Auf der Diagonalen wären beide Regeln
      gleich, überall sonst nicht.
    * Die schraffierten Zellen sind die **Divergenzzone**: Dort liegen
      Tragfähigkeit und Bereitschaft mindestens zwei Stufen auseinander. Das
      System mittelt sie nicht weg, sondern weist den Abstand aus.
    * Unter jeder Zahl steht in Klammern, was eine Mittelung ergäbe. Die
      Differenz ist der eigentliche Gegenstand der Abbildung.

    Args:
        policies: Regelwerk, für die Klassenzahl.
        output: Zielpfad der PNG-Datei.

    Returns:
        Der Pfad der geschriebenen Datei.
    """
    apply_style()
    klassen = sorted(policies.classes.risikoklassen)
    anzahl = len(klassen)
    figur, achse = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.92))

    for zeile, bereitschaft in enumerate(klassen):
        for spalte, tragfaehigkeit in enumerate(klassen):
            ergebnis = min(tragfaehigkeit, bereitschaft)
            abstand = abs(tragfaehigkeit - bereitschaft)
            gemittelt = (tragfaehigkeit + bereitschaft) / 2
            divergent = abstand >= DIVERGENCE_THRESHOLD

            # Sättigung nach Ergebnisklasse: je höher die Klasse, desto dunkler.
            anteil = (ergebnis - 1) / (anzahl - 1)
            flaeche = Rectangle(
                (spalte, zeile),
                1,
                1,
                facecolor=_mischfarbe(HELLBLAU, NAVY, anteil),
                edgecolor="white",
                linewidth=1.4,
                zorder=1,
            )
            achse.add_patch(flaeche)
            if divergent:
                # Die Schraffur markiert, sie darf aber nicht dominieren: Die
                # Aussage der Abbildung sind die Zahlen in den Zellen, nicht die
                # Zone. Deshalb duenne, halbtransparente Linien und eine
                # kraeftige Umrandung statt einer dichten Fuellung.
                achse.add_patch(
                    Rectangle(
                        (spalte, zeile),
                        1,
                        1,
                        facecolor="none",
                        edgecolor=AKZENT,
                        hatch="////",
                        linewidth=0.0,
                        alpha=0.30,
                        zorder=2,
                    )
                )
                achse.add_patch(
                    Rectangle(
                        (spalte + 0.02, zeile + 0.02),
                        0.96,
                        0.96,
                        facecolor="none",
                        edgecolor=AKZENT,
                        linewidth=1.3,
                        zorder=2,
                    )
                )

            textfarbe = "white" if anteil > LIGHT_TEXT_THRESHOLD else NAVY
            achse.text(
                spalte + 0.5,
                zeile + 0.60,
                str(ergebnis),
                ha="center",
                va="center",
                fontsize=FONT_SIZE + 3,
                fontweight="bold",
                color=textfarbe,
                zorder=3,
            )
            if gemittelt != ergebnis:
                achse.text(
                    spalte + 0.5,
                    zeile + 0.26,
                    f"({german_decimal(gemittelt, decimals=1)})",
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE - 3,
                    color=textfarbe,
                    zorder=3,
                )

    achse.set_xlim(0, anzahl)
    achse.set_ylim(0, anzahl)
    achse.set_xticks([i + 0.5 for i in range(anzahl)])
    achse.set_yticks([i + 0.5 for i in range(anzahl)])
    achse.set_xticklabels([str(k) for k in klassen])
    achse.set_yticklabels([str(k) for k in klassen])
    achse.set_xlabel("Risikotragfähigkeit (objektiv)")
    achse.set_ylabel("Risikobereitschaft (subjektiv)")
    achse.set_aspect("equal")
    for rand in ("top", "right", "bottom", "left"):
        achse.spines[rand].set_visible(False)
    achse.tick_params(length=0)

    legende = [
        Patch(
            facecolor=_mischfarbe(HELLBLAU, NAVY, 0.5), edgecolor="white", label="Ergebnisklasse"
        ),
        Patch(
            facecolor="none",
            edgecolor=AKZENT,
            hatch="////",
            linewidth=1.3,
            label=f"Divergenz ab {DIVERGENCE_THRESHOLD} Stufen",
        ),
    ]
    achse.legend(
        handles=legende,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )

    figur.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figur.savefig(output, format="png")
    plt.close(figur)
    return output


def _mischfarbe(hell: str, dunkel: str, anteil: float) -> tuple[float, float, float]:
    """Mischt zwei Farben linear.

    Args:
        hell: Farbe bei Anteil null.
        dunkel: Farbe bei Anteil eins.
        anteil: Mischungsanteil zwischen 0 und 1.

    Returns:
        Die gemischte Farbe als RGB-Tripel.
    """
    von = np.array(to_rgb(hell))
    bis = np.array(to_rgb(dunkel))
    gemischt = von + (bis - von) * float(np.clip(anteil, 0.0, 1.0))
    return (float(gemischt[0]), float(gemischt[1]), float(gemischt[2]))


def stress_chart(results: Sequence[StressResult], output: Path) -> Path:
    """Zeichnet die Stresstest-Drawdowns je Krise und Klasse (Abbildung 6).

    Die zugesagte Verlusterwartung jeder Klasse steht als waagerechter Strich
    über der jeweiligen Balkengruppe. Wo ein Balken darüber hinausragt, wurde
    die Zusage gebrochen — diese Balken sind in der Akzentfarbe gezeichnet und
    nicht etwa weggelassen.

    Args:
        results: Ergebnisse der Stresstests.
        output: Zielpfad der PNG-Datei.

    Returns:
        Der Pfad der geschriebenen Datei.
    """
    apply_style()
    # Reihenfolge des ersten Auftretens, nicht alphabetisch: Die Fenster sind
    # in CRISIS_WINDOWS chronologisch deklariert, und chronologisch gehören sie
    # auch auf die Achse.
    fenster = list(dict.fromkeys(ergebnis.window.key for ergebnis in results))
    beschriftung = {ergebnis.window.key: ergebnis.window.label for ergebnis in results}
    klassen = sorted({ergebnis.risk_class for ergebnis in results})

    figur, achse = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.5))
    breite = 0.8 / len(klassen)

    for i, klasse in enumerate(klassen):
        positionen = []
        werte = []
        farben = []
        for j, schluessel in enumerate(fenster):
            treffer = next(
                e for e in results if e.window.key == schluessel and e.risk_class == klasse
            )
            positionen.append(j - 0.4 + i * breite + breite / 2)
            werte.append(abs(treffer.max_drawdown) * 100.0)
            farben.append(
                AKZENT
                if not treffer.promise_held
                else _mischfarbe(HELLBLAU, NAVY, (klasse - 1) / (len(klassen) - 1))
            )
        achse.bar(
            positionen,
            werte,
            width=breite * 0.92,
            color=farben,
            edgecolor="white",
            linewidth=0.5,
            label=f"Klasse {klasse}",
            zorder=2,
        )

    # Zugesagte Verlusterwartung je Klasse als Strich über der Balkengruppe.
    for j, schluessel in enumerate(fenster):
        for i, klasse in enumerate(klassen):
            treffer = next(
                e for e in results if e.window.key == schluessel and e.risk_class == klasse
            )
            mitte = j - 0.4 + i * breite + breite / 2
            achse.plot(
                [mitte - breite * 0.46, mitte + breite * 0.46],
                [treffer.promised_loss * 100.0] * 2,
                color=GRAU,
                linewidth=1.1,
                zorder=3,
            )

    achse.set_xticks(range(len(fenster)))
    achse.set_xticklabels([_umbrechen(beschriftung[s]) for s in fenster], fontsize=FONT_SIZE - 2)
    achse.set_ylabel("Größter Rückgang")
    achse.yaxis.set_major_formatter(
        FuncFormatter(lambda w, _: f"{german_decimal(w, decimals=0)} %")
    )
    achse.grid(visible=True, axis="y")
    achse.spines["top"].set_visible(False)
    achse.spines["right"].set_visible(False)

    handles: list[Patch | Line2D] = [
        Patch(
            facecolor=_mischfarbe(HELLBLAU, NAVY, (k - 1) / (len(klassen) - 1)),
            label=f"Klasse {k}",
        )
        for k in klassen
    ]
    handles.append(Patch(facecolor=AKZENT, label="Zusage gebrochen"))
    handles.append(Line2D([0], [0], color=GRAU, linewidth=1.1, label="zugesagte Erwartung"))
    achse.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
        fontsize=FONT_SIZE - 3,
        columnspacing=1.1,
        handlelength=1.4,
    )

    figur.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figur.savefig(output, format="png")
    plt.close(figur)
    return output


def tornado_chart(reports: Sequence[SensitivityReport], output: Path) -> Path:
    """Zeichnet den Hebel je Fragebogen-Item als Tornado-Diagramm (Abbildung 7).

    Ausgewiesen ist der **größte** über alle Basisprofile gemessene Hebel je
    Item, getrennt nach Richtung: wie viele Klassenstufen das Item abwärts und
    wie viele es aufwärts bewegen kann. Die Trennung ist wesentlich — ein Item,
    das nur abwärts wirkt, ist ein Ausschlusskriterium und keine Skala.

    Args:
        reports: Berichte je Basisprofil.
        output: Zielpfad der PNG-Datei.

    Returns:
        Der Pfad der geschriebenen Datei.
    """
    apply_style()
    gesammelt: dict[str, tuple[str, str, int, int]] = {}
    for bericht in reports:
        for eintrag in bericht.items:
            label, achse_name, abwaerts, aufwaerts = gesammelt.get(
                eintrag.item, (eintrag.label, eintrag.axis, 0, 0)
            )
            gesammelt[eintrag.item] = (
                label,
                achse_name,
                max(abwaerts, eintrag.downside),
                max(aufwaerts, eintrag.upside),
            )

    geordnet = sorted(gesammelt.values(), key=lambda e: (e[2] + e[3], e[3]))
    figur, achse = plt.subplots(figsize=(FIGURE_WIDTH_IN, 0.34 * len(geordnet) + 1.1))
    positionen = np.arange(len(geordnet))

    achse.barh(
        positionen,
        [-e[2] for e in geordnet],
        color=AKZENT,
        edgecolor="white",
        linewidth=0.5,
        label="senkt die Klasse",
    )
    achse.barh(
        positionen,
        [e[3] for e in geordnet],
        color=NAVY,
        edgecolor="white",
        linewidth=0.5,
        label="hebt die Klasse",
    )

    achse.set_yticks(positionen)
    achse.set_yticklabels([_kurz(e[0]) for e in geordnet], fontsize=FONT_SIZE - 2.5)
    achse.axvline(0, color=GRAU, linewidth=0.9)
    achse.set_xlabel("Veränderung der Risikoklasse in Stufen")
    achse.xaxis.set_major_formatter(FuncFormatter(lambda w, _: german_decimal(abs(w), decimals=0)))
    achse.grid(visible=True, axis="x")
    for rand in ("top", "right", "left"):
        achse.spines[rand].set_visible(False)
    achse.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE - 3,
    )

    figur.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figur.savefig(output, format="png")
    plt.close(figur)
    return output


def _umbrechen(label: str) -> str:
    """Bricht eine Fensterbeschriftung vor der Jahreszahl um.

    Args:
        label: Vollständige Bezeichnung des Krisenfensters.

    Returns:
        Die Bezeichnung mit einem Umbruch vor der Jahresangabe.
    """
    kopf, _, jahr = label.rpartition(" ")
    return f"{kopf}\n{jahr}" if kopf else label


def _kurz(label: str, laenge: int = 34) -> str:
    """Kürzt eine Beschriftung auf eine für die Achse verträgliche Länge.

    Args:
        label: Vollständige Beschriftung.
        laenge: Höchstlänge.

    Returns:
        Die gekürzte Beschriftung.
    """
    return label if len(label) <= laenge else label[: laenge - 1].rstrip() + "…"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
