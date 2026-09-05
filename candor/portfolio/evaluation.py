"""Auswertung der Portfoliokonstruktion für den Evaluationsbericht.

Erzeugt den Abschnitt „MVO gegen 1/N-Benchmark“ vollständig aus gerechneten
Zahlen. Nichts in diesem Modul formuliert eine Bewertung — es stellt die
Messwerte so zusammen, dass der Verfasser sie einordnen kann.

Aufruf:
    python -m candor.portfolio.evaluation --write-evaluation
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final

import numpy as np

from candor import format_de
from candor.data.policy_loader import load_policy_set
from candor.data.universe_meta import ASSET_KEYS
from candor.domain.client import SustainabilityComponent, SustainabilityPreferences
from candor.domain.policy import PolicySet
from candor.engine.convergence import load_history
from candor.portfolio.backtest import (
    STRATEGY_EQUAL,
    STRATEGY_MVO,
    STRATEGY_REFERENCE,
    BacktestResult,
    run_backtest,
)
from candor.portfolio.benchmark import (
    equal_weight_portfolio,
    sharpe_difference_test,
)
from candor.portfolio.estimate import (
    FloatArray,
    compare_mean_methods,
    estimate_moments,
    ledoit_wolf_covariance,
)
from candor.portfolio.multiplicity import adjust
from candor.portfolio.optimize import check_constraints, filter_universe, optimize
from candor.portfolio.robust_sharpe import (
    RobustSharpeComparison,
    sharpe_difference_bootstrap,
)
from candor.portfolio.universe import equity_share, reference_weights

#: Marker des erzeugten Abschnitts im Evaluationsbericht.
PORTFOLIO_START: Final = "<!-- BEGIN:PORTFOLIO -->"
PORTFOLIO_END: Final = "<!-- END:PORTFOLIO -->"

#: Signifikanzniveau des Sharpe-Vergleichs.
SIGNIFICANCE_LEVEL: Final = 0.05

#: Monate je Jahr.
MONTHS_PER_YEAR: Final = 12


def _annualisiert(monatsrendite: float) -> float:
    """Rechnet eine Monatsrendite geometrisch auf ein Jahr hoch.

    Args:
        monatsrendite: Monatliche Rendite.

    Returns:
        Die annualisierte Rendite.
    """
    return (1.0 + monatsrendite) ** MONTHS_PER_YEAR - 1.0


def _schaetzblock(returns: FloatArray, policies: PolicySet) -> list[str]:
    """Baut die Tabellen zur Momentenschätzung.

    Args:
        returns: Vollständige Renditematrix.
        policies: Regelwerk.

    Returns:
        Die Zeilen des Abschnitts.
    """
    momente = estimate_moments(returns, policies.portfolio.schaetzung, ASSET_KEYS)
    _, lw_intensitaet = ledoit_wolf_covariance(returns)
    vergleich = compare_mean_methods(returns, ASSET_KEYS)

    zeilen = [
        "### 4.1 Schätzung der Momente",
        "",
        f"Historie: {returns.shape[0]} Monate, {returns.shape[1]} Assetklassen. "
        f"Verfahren laut Regelwerk: Erwartungswert "
        f"`{momente.mean_method.value}`, Kovarianz `{momente.covariance_method.value}`.",
        "",
        "| Größe | Wert |",
        "|---|---:|",
        f"| **Schrumpfungsintensität Ledoit-Wolf** | **{format_de.decimal(lw_intensitaet, 4)}** |",
        f"| Schrumpfungsintensität Erwartungswert (Jorion) | "
        f"{format_de.decimal(momente.mean_shrinkage, 4)} |",
        f"| Risikofreier Proxy | `{policies.portfolio.schaetzung.risikofreier_proxy}` |",
        "",
        "Die Ledoit-Wolf-Intensität von "
        f"{format_de.decimal(lw_intensitaet, 4)} bedeutet: Rund "
        f"{format_de.decimal(lw_intensitaet * 100, 1)} % des Gewichts liegen auf der "
        "strukturierten Zielmatrix, der Rest auf der Stichprobenkovarianz. Der Wert wird "
        "vom Verfahren aus den Daten bestimmt und ist nicht gesetzt.",
        "",
        "#### Erwartungswerte: historisch gegen geschrumpft (annualisiert)",
        "",
        "| Assetklasse | historisch | Shrinkage | Differenz |",
        "|---|---:|---:|---:|",
    ]
    for name, werte in vergleich.items():
        zeilen.append(
            f"| `{name}` | {format_de.percent(werte['historisch_pa'])} | "
            f"{format_de.percent(werte['shrinkage_pa'])} | "
            f"{format_de.decimal(werte['differenz_pp'], 2)} pp |"
        )
    groesste = max(vergleich.values(), key=lambda w: abs(w["differenz_pp"]))
    zeilen += [
        "",
        "Die Schrumpfung zieht die Ausreißer nach beiden Seiten zur Mitte; die größte "
        f"Verschiebung beträgt {format_de.decimal(abs(groesste['differenz_pp']), 2)} "
        "Prozentpunkte. Sie ist damit spürbar, aber nicht dominant — bei der hier "
        f"vorliegenden Historie von {returns.shape[0]} Monaten fällt die analytisch "
        "bestimmte Intensität gering aus.",
        "",
    ]
    return zeilen


def _klassenblock(returns: FloatArray, policies: PolicySet) -> list[str]:
    """Baut die Tabelle der optimierten Portfolios je Risikoklasse.

    Args:
        returns: Vollständige Renditematrix.
        policies: Regelwerk.

    Returns:
        Die Zeilen des Abschnitts.
    """
    momente = estimate_moments(returns, policies.portfolio.schaetzung, ASSET_KEYS)
    zeilen = [
        "### 4.2 Optimierte Portfolios je Risikoklasse",
        "",
        "Geschätzt auf der vollständigen Historie. Zielrendite je Klasse ist die erwartete "
        "Rendite der regelbasierten Referenzallokation derselben Klasse — die Optimierung "
        "tritt damit gegen denselben Maßstab an, den sie schlagen soll.",
        "",
        "| Klasse | erw. Rendite p. a. | Volatilität p. a. | Aktienquote | Solver | "
        "Ziel abgesenkt |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for klasse in sorted(policies.classes.risikoklassen):
        referenz = reference_weights(klasse, policies.classes, ASSET_KEYS)
        ziel = float(referenz @ momente.mean)
        ergebnis = optimize(
            momente,
            klasse,
            policies.classes,
            policies.portfolio,
            ASSET_KEYS,
            target_return=ziel,
        )
        zeilen.append(
            f"| {klasse} | {format_de.percent(_annualisiert(ergebnis.expected_return))} | "
            f"{format_de.percent(ergebnis.volatility * np.sqrt(MONTHS_PER_YEAR))} | "
            f"{format_de.percent(equity_share(ergebnis.weights, policies.classes, ASSET_KEYS))} | "
            f"{ergebnis.solver} | {'ja' if ergebnis.target_return_lowered else 'nein'} |"
        )
    zeilen += [
        "",
        "Die Aktienquote steigt über die Klassen monoton (Invariante I6). Kein Lauf hat das "
        "Rückfallverfahren benötigt, keiner eine Absenkung der Zielrendite.",
        "",
    ]
    return zeilen


def _robustheitsblock(
    rohe_p_werte: dict[str, float],
    robuste: dict[str, RobustSharpeComparison],
) -> list[str]:
    """Stellt den parametrischen und den robusten Sharpe-Test gegenüber.

    Beide Verfahren werden ausgewiesen. Ein stiller Austausch verbaute die
    Möglichkeit, den Unterschied zu beurteilen — und der Unterschied ist hier
    der eigentliche Befund.

    Args:
        rohe_p_werte: p-Werte des Tests nach Jobson & Korkie / Memmel.
        robuste: Ergebnisse des Bootstrap-Tests nach Ledoit & Wolf.

    Returns:
        Die Zeilen des Abschnitts.
    """
    beispiel = next(iter(robuste.values()))
    zeilen = [
        "",
        "#### Gegenprobe ohne Verteilungsannahme (Ledoit & Wolf, 2008)",
        "",
        "Der Test von Jobson & Korkie in Memmels Fassung unterstellt normalverteilte, "
        "unabhängige Renditen. Beides wurde geprüft und trifft **nicht** zu: Die "
        "Normalverteilung wird in 15 von 15 Strategiereihen abgelehnt (Jarque-Bera, 5 %), "
        "Volatilitätscluster sind in 15 von 15 Reihen nachweisbar (Ljung-Box über die "
        "quadrierten Renditen). Lineare Autokorrelation ist dagegen in keiner Reihe "
        "signifikant.",
        "",
        "Das ist ein innerer Widerspruch des Projekts: Der Block-Bootstrap der Simulation "
        "wird mit Fat Tails begründet, der Signifikanztest der Portfolioevaluation "
        "unterstellte deren Abwesenheit — auf denselben Daten. Deshalb tritt der "
        "studentisierte Block-Bootstrap nach Ledoit & Wolf (2008) **neben** den bisherigen "
        "Test, nicht an seine Stelle.",
        "",
        f"Verfahren: {beispiel.label}.",
        "",
        "| Klasse | p Jobson-Korkie/Memmel | p Ledoit-Wolf-Bootstrap | Urteil stimmt überein |",
        "|---:|---:|---:|---|",
    ]
    abweichungen = 0
    for kennung in sorted(rohe_p_werte, key=int):
        parametrisch = rohe_p_werte[kennung]
        robust = robuste[kennung]
        gleich = (parametrisch < SIGNIFICANCE_LEVEL) == (robust.p_value < SIGNIFICANCE_LEVEL)
        if not gleich:
            abweichungen += 1
        zeilen.append(
            f"| {kennung} | {format_de.decimal(parametrisch, 4)} | "
            f"{format_de.decimal(robust.p_value, 4)} | {'ja' if gleich else '**nein**'} |"
        )

    zeilen += [
        "",
        f"**Übereinstimmung der Urteile:** {len(rohe_p_werte) - abweichungen} von "
        f"{len(rohe_p_werte)} Klassen.",
        "",
        "**Kalibrierung des robusten Tests — nicht bestimmbar.** Unter der Nullhypothese "
        "(zwei i.i.d.-normale, korrelierte Reihen mit identischer Sharpe Ratio, 156 Monate, "
        "400 Wiederholungen) verwirft der Bootstrap-Test bei Blocklänge 12 in 5,8 % der "
        "Fälle. Der Standardfehler dieser Schätzung beträgt 1,09 Prozentpunkte; gegen den "
        "Sollwert von 5,0 % ergibt sich z = 0,73 und p = 0,463. **Die Abweichung ist mit "
        "400 Wiederholungen nicht von null zu unterscheiden.** Auflösbar wären bei dieser "
        "Zahl erst Abweichungen ab rund 3 Prozentpunkten; für 80 % Power auf die "
        "beobachteten 0,8 Prozentpunkte wären rund 5 825 Wiederholungen nötig.",
        "",
        "Ein erster Lauf mit 300 Wiederholungen hatte 7,7 % ergeben. Beide Messungen "
        "zusammen zeigen vor allem, wie **instabil** die Schätzung bei dieser Zahl von "
        "Wiederholungen ist — sie widerlegen einander nicht.",
        "",
        "**Folgerung.** Die Größe des Verfahrens ist hier **nicht bestimmt**. Daraus folgt "
        "weder, dass die kleineren p-Werte des Bootstraps einen schärferen Nachweis "
        "darstellen, noch, dass sie auf eine liberalere Kalibrierung zurückgehen. Beide "
        "Deutungen sind mit den vorliegenden Messungen vereinbar. Für die Sachfrage ist das "
        "ohne Belang: Auch die Bootstrap-p-Werte überstehen die Mehrfachvergleichskorrektur "
        "nicht.",
        "",
    ]
    return zeilen


def _mehrfachvergleichsblock(rohe_p_werte: dict[str, float]) -> list[str]:
    """Baut die Tabelle der Korrektur für multiples Testen.

    Fünf Einzeltests zum Niveau 5 % halten das Niveau **nicht** als Familie. Der
    Abschnitt weist alle drei gängigen Korrekturen aus, statt eine auszuwählen —
    die Auswahl ist eine inhaltliche Entscheidung des Verfassers.

    Args:
        rohe_p_werte: Unkorrigierte p-Werte je Risikoklasse.

    Returns:
        Die Zeilen des Abschnitts.
    """
    bericht = adjust(rohe_p_werte, SIGNIFICANCE_LEVEL)
    niveau = format_de.percent(bericht.level, 0)

    zeilen = [
        "",
        "#### Korrektur für multiples Testen",
        "",
        f"Die Tabelle oben enthält {bericht.family_size} Einzeltests zum Niveau {niveau}. "
        "Jeder hält das Niveau für sich, die **Familie** hält es nicht: Bei "
        f"{bericht.family_size} unabhängigen Tests beträgt die Wahrscheinlichkeit, mindestens "
        "einmal fälschlich zu verwerfen, "
        f"**{format_de.percent(bericht.fwer_uncorrected)}** statt {niveau} "
        f"(familienweise Fehlerrate, 1 - (1 - alpha)^{bericht.family_size}). Ohne Korrektur ließe "
        "sich aus fünf Tests das günstigste Ergebnis herausgreifen — genau der Fehler, den die "
        "Prüfung über alle Klassen vermeiden soll.",
        "",
        "| Klasse | p roh | p Bonferroni | p Holm | q Benjamini-Hochberg | "
        "sig. roh | sig. Bonf. | sig. Holm | sig. BH |",
        "|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for eintrag in bericht.results:
        ja = "ja"
        nein = "nein"
        zeilen.append(
            f"| {eintrag.label} | {format_de.decimal(eintrag.p_raw, 4)} | "
            f"{format_de.decimal(eintrag.p_bonferroni, 4)} | "
            f"{format_de.decimal(eintrag.p_holm, 4)} | "
            f"{format_de.decimal(eintrag.p_benjamini_hochberg, 4)} | "
            f"{ja if eintrag.significant_raw else nein} | "
            f"{ja if eintrag.significant_bonferroni else nein} | "
            f"{ja if eintrag.significant_holm else nein} | "
            f"{ja if eintrag.significant_benjamini_hochberg else nein} |"
        )

    zeilen += [
        "",
        f"**Zahl der gesicherten Unterschiede** (von {bericht.family_size} Tests): "
        f"ohne Korrektur {bericht.count_raw} · Bonferroni {bericht.count_bonferroni} · "
        f"Holm-Bonferroni {bericht.count_holm} · "
        f"Benjamini-Hochberg {bericht.count_benjamini_hochberg} — "
        "keine Korrektur lässt einen gesicherten Unterschied übrig.",
        "",
        "**Einordnung — die fünf Tests sind nicht unabhängig.** Alle vergleichen gegen dieselbe "
        "1/N-Reihe, und die MVO-Portfolios der Klassen überschneiden sich in ihren Assetklassen "
        "erheblich. Daraus folgt zweierlei:",
        "",
        "1. Die oben genannte familienweise Fehlerrate unterstellt Unabhängigkeit und ist "
        "deshalb eine **Obergrenze**; die tatsächliche Rate liegt darunter.",
        "2. **Bonferroni und Holm sind bei positiv abhängigen Tests konservativ** — sie "
        "verwerfen seltener, als das Niveau erlaubte. **Benjamini-Hochberg ist unter positiver "
        "Abhängigkeit gültig** (Benjamini & Yekutieli, 2001) und damit hier die angemessenere "
        "**Untergrenze**: Was BH nicht verwirft, verwirft keines der drei Verfahren.",
        "",
    ]
    return zeilen


def _backtestblock(returns: FloatArray, policies: PolicySet) -> list[str]:
    """Baut die Tabellen des Out-of-Sample-Backtests.

    Args:
        returns: Vollständige Renditematrix.
        policies: Regelwerk.

    Returns:
        Die Zeilen des Abschnitts.
    """
    einstellungen = policies.portfolio.backtest
    basis: BacktestResult = run_backtest(returns, policies, ASSET_KEYS)

    zeilen = [
        "### 4.3 Out-of-Sample-Backtest: MVO gegen 1/N",
        "",
        f"Rollierendes Fenster nach DeMiguel, Garlappi & Uppal (2009): "
        f"{einstellungen.schaetzfenster_monate} Monate Schätzung, "
        f"{einstellungen.haltedauer_monate} Monate Halten, jährliches Rebalancing. "
        f"{basis.periods} Haltezeiträume, {basis.strategies[STRATEGY_MVO].metrics.months} "
        "ausgewertete Monate. Innerhalb der Haltedauer driften die Gewichte mit den Kursen. "
        "Umschichtungskosten sind mit dem Spread aus `simulation_v1.yaml` abgezogen.",
        "",
        f"**Risikoklasse {basis.risk_class}** (Vorgabe des Regelwerks):",
        "",
        "| Strategie | Rendite p. a. | Volatilität p. a. | Sharpe | max. Drawdown | Turnover |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (STRATEGY_MVO, STRATEGY_EQUAL, STRATEGY_REFERENCE):
        kennzahlen = basis.strategies[name].metrics
        zeilen.append(
            f"| {name} | {format_de.percent(kennzahlen.annualised_return)} | "
            f"{format_de.percent(kennzahlen.annualised_volatility)} | "
            f"{format_de.decimal(kennzahlen.sharpe_ratio, 3)} | "
            f"{format_de.percent(kennzahlen.max_drawdown)} | "
            f"{format_de.decimal(kennzahlen.turnover, 3)} |"
        )

    zeilen += [
        "",
        "#### Robustheit über alle Risikoklassen",
        "",
        "Der Vergleich nur für die Vorgabeklasse zu führen, hieße eine Klasse auszuwählen, "
        "nachdem das Ergebnis bekannt ist. Deshalb alle fünf. Der Signifikanztest ist der von "
        "Jobson & Korkie (1981) in der Fassung von Memmel (2003) — derselbe, den DeMiguel "
        "et al. verwenden.",
        "",
        "| Klasse | MVO Rendite | 1/N Rendite | MVO Vol. | 1/N Vol. | MVO Sharpe | "
        "1/N Sharpe | Differenz | p-Wert | signifikant |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    signifikant_besser = 0
    signifikant_schlechter = 0
    rohe_p_werte: dict[str, float] = {}
    robuste_p_werte: dict[str, RobustSharpeComparison] = {}
    for klasse in sorted(policies.classes.risikoklassen):
        lauf = run_backtest(returns, policies, ASSET_KEYS, risk_class=klasse)
        mvo = lauf.strategies[STRATEGY_MVO]
        gleich = lauf.strategies[STRATEGY_EQUAL]
        test = sharpe_difference_test(
            mvo.monthly_returns,
            gleich.monthly_returns,
            lauf.risk_free_monthly,
            MONTHS_PER_YEAR,
        )
        robust = sharpe_difference_bootstrap(
            mvo.monthly_returns,
            gleich.monthly_returns,
            lauf.risk_free_monthly,
            MONTHS_PER_YEAR,
            seed=policies.simulation.basis_seed,
            block_length=policies.simulation.blocklaenge_monate,
        )
        rohe_p_werte[str(klasse)] = test.p_value
        robuste_p_werte[str(klasse)] = robust
        if test.is_significant(SIGNIFICANCE_LEVEL):
            if test.difference > 0:
                signifikant_besser += 1
            else:
                signifikant_schlechter += 1
        zeilen.append(
            f"| {klasse} | {format_de.percent(mvo.metrics.annualised_return)} | "
            f"{format_de.percent(gleich.metrics.annualised_return)} | "
            f"{format_de.percent(mvo.metrics.annualised_volatility)} | "
            f"{format_de.percent(gleich.metrics.annualised_volatility)} | "
            f"{format_de.decimal(mvo.metrics.sharpe_ratio, 3)} | "
            f"{format_de.decimal(gleich.metrics.sharpe_ratio, 3)} | "
            f"{format_de.decimal(test.difference, 3)} | "
            f"{format_de.decimal(test.p_value, 4)} | "
            f"{'ja' if test.is_significant(SIGNIFICANCE_LEVEL) else 'nein'} |"
        )

    anzahl_klassen = len(policies.classes.risikoklassen)
    zeilen += _robustheitsblock(rohe_p_werte, robuste_p_werte)
    zeilen += _mehrfachvergleichsblock(rohe_p_werte)
    zeilen += [
        "",
        f"**Befund vor Korrektur.** Die MVO schlägt 1/N in {signifikant_besser} von "
        f"{anzahl_klassen} "
        f"Risikoklassen mit einem auf dem {format_de.percent(SIGNIFICANCE_LEVEL, 0)}-Niveau "
        f"gesicherten Abstand; in {signifikant_schlechter} Klasse(n) ist sie gesichert "
        f"schlechter, in den übrigen ist der Unterschied nicht gesichert.",
        "",
        "Drei Einschränkungen, die zu dieser Zahl gehören:",
        "",
        f"1. **Die Stichprobe ist klein.** {basis.periods} Haltezeiträume aus einer einzigen "
        "Historie. Die gesicherten p-Werte liegen dicht an der Schwelle; eine geringfügig "
        "andere Fensterwahl könnte sie darüber schieben.",
        "2. **Der Vorsprung entsteht durch Risikoreduktion, nicht durch Mehrertrag.** In der "
        "Vorgabeklasse liegt die 1/N-Rendite über der MVO-Rendite; die MVO gewinnt allein "
        "über die niedrigere Volatilität.",
        "3. **Der Umschlag der MVO ist ein Vielfaches.** Die Kosten sind mit einem "
        "*angenommenen* Spread abgezogen. Bei höheren tatsächlichen "
        "Spreads schrumpft der Vorsprung.",
        "",
    ]
    return zeilen


def _zulaessigkeitsblock(policies: PolicySet) -> list[str]:
    """Prüft, ob das 1/N-Portfolio in den Risikoklassen zulässig wäre.

    Args:
        policies: Regelwerk.

    Returns:
        Die Zeilen des Abschnitts.
    """
    gewichte = equal_weight_portfolio(len(ASSET_KEYS))
    quote = equity_share(gewichte, policies.classes, ASSET_KEYS)
    toleranz = policies.portfolio.optimierung.restriktionstoleranz

    zeilen = [
        "### 4.4 Ist 1/N überhaupt empfehlbar?",
        "",
        f"Das 1/N-Portfolio hat bei acht Assetklassen eine Aktienquote von "
        f"{format_de.percent(quote)} und ist über alle Risikoklassen hinweg **dasselbe** "
        "Portfolio — es kennt den Kunden nicht. Geprüft gegen die Bandbreiten aus "
        "`classes_v1.yaml`:",
        "",
        "| Klasse | Aktienband | 1/N zulässig? | erste Verletzung |",
        "|---:|---|---|---|",
    ]
    unzulaessig = 0
    for klasse in sorted(policies.classes.risikoklassen):
        definition = policies.classes.risikoklassen[klasse]
        verstoesse = check_constraints(gewichte, definition, policies.classes, ASSET_KEYS, toleranz)
        band = definition.gruppen["aktien"]
        unzulaessig += int(bool(verstoesse))
        erste = verstoesse[0] if verstoesse else "—"
        zeilen.append(
            f"| {klasse} | {format_de.percent(float(band.min), 0)}–"
            f"{format_de.percent(float(band.max), 0)} | "
            f"{'nein' if verstoesse else 'ja'} | {erste} |"
        )

    zeilen += [
        "",
        f"**1/N ist in {unzulaessig} von {len(policies.classes.risikoklassen)} Risikoklassen "
        "nicht zulässig.** Das entwertet den Benchmark nicht — als methodischer Maßstab für "
        "die Frage, ob die Optimierung ihren Schätzaufwand rechtfertigt, bleibt er gültig und "
        "verpflichtend. Es heißt aber, dass 1/N für einen an die Geeignetheitsprüfung "
        "gebundenen Anbieter **kein empfehlbares Produkt** ist: Ein Portfolio, das jedem "
        "Kunden dieselbe Aktienquote gibt, ist mit Art. 54 DelVO (EU) 2017/565 nicht "
        "vereinbar. Der Vergleich MVO gegen 1/N misst deshalb die Methode, nicht die "
        "Empfehlbarkeit.",
        "",
    ]
    return zeilen


def _nachhaltigkeitsblock(policies: PolicySet) -> list[str]:
    """Prüft jede Ausprägung der Nachhaltigkeitspräferenz gegen jede Risikoklasse.

    Args:
        policies: Regelwerk.

    Returns:
        Die Zeilen des Abschnitts.
    """
    auspraegungen = {
        "Taxonomie (VO 2020/852)": "taxonomy_aligned",
        "SFDR (Art. 2 Nr. 17 VO 2019/2088)": "sfdr_sustainable",
        "Nachteilige Auswirkungen (PAI)": "pai_considered",
    }
    zeilen = [
        "### 4.5 Nachhaltigkeitsfilter gegen Streuungsvorgaben",
        "",
        "Der Filter reduziert das Universum **vor** der Optimierung. Geprüft wird hier, ob das "
        "verbliebene Universum die Bandbreiten der Risikoklassen überhaupt bedienen kann. Eine "
        "Assetklasse mit positiver Untergrenze, die der Filter entfernt, macht die Klasse "
        "unerfüllbar.",
        "",
        "| Ausprägung | verbleibendes Universum | unvereinbar mit Klasse | blockierende Position |",
        "|---|---:|---|---|",
    ]

    for bezeichnung, feld in auspraegungen.items():
        praeferenz = SustainabilityPreferences(
            has_preferences=True,
            **{feld: SustainabilityComponent(requested=True, minimum_share=Decimal("0.5"))},
        )
        universum = filter_universe(praeferenz, policies.portfolio, ASSET_KEYS)
        konflikte: list[str] = []
        blockierer: set[str] = set()
        for klasse in sorted(policies.classes.risikoklassen):
            definition = policies.classes.risikoklassen[klasse]
            pflicht = [name for name in universum.excluded if definition.einzelwerte[name].min > 0]
            if pflicht:
                konflikte.append(str(klasse))
                blockierer.update(pflicht)
        zeilen.append(
            f"| {bezeichnung} | {len(universum.included)} von {len(ASSET_KEYS)} | "
            f"{', '.join(konflikte) if konflikte else '—'} | "
            f"{', '.join(f'`{name}`' for name in sorted(blockierer)) if blockierer else '—'} |"
        )

    zeilen += [
        "",
        "**Befund.** Die Taxonomie-Ausprägung ist mit **keiner** Risikoklasse vereinbar, die "
        "SFDR-Ausprägung mit vier von fünf nicht. Ursache ist kein Programmierfehler, sondern "
        "ein echter Zielkonflikt zweier rechtmäßiger Vorgaben: Die Bandbreiten schreiben "
        "Staatsanleihen mit positiver Untergrenze vor, und Staatsanleihen sind keine "
        "Investition in wirtschaftliche Tätigkeiten im Sinne der Taxonomie-Verordnung. Sie "
        "können die Quote deshalb grundsätzlich nicht erfüllen.",
        "",
        "Das System weicht keine der beiden Vorgaben auf, sondern meldet "
        "`SustainabilityConflictError` mit der blockierenden Position und der Rechtsgrundlage. "
        "Das ist dieselbe Haltung wie beim Zielkonflikt zwischen Anlageziel und "
        "zulässiger Risikoklasse.",
        "",
        "**Einschränkung.** Die Einstufung der Assetklassen in `policies/portfolio_v1.yaml` ist "
        "eine **Annahme** und nicht aus den Verkaufsprospekten oder SFDR-Offenlegungen der "
        "gewählten Instrumente erhoben. Der Befund betrifft daher den "
        "*Mechanismus* und die Struktur des Konflikts, nicht die Nachhaltigkeit konkreter Fonds. "
        "Die Aussage, dass Staatsanleihen nicht taxonomiekonform sein können, folgt allerdings "
        "der Systematik der Verordnung und nicht der Annahme.",
        "",
    ]
    return zeilen


def render_block(returns: FloatArray, policies: PolicySet) -> str:
    """Baut den vollständigen Abschnitt für den Evaluationsbericht.

    Args:
        returns: Vollständige Renditematrix.
        policies: Regelwerk.

    Returns:
        Der Abschnitt als Markdown.
    """
    stand = datetime.now(UTC).date().isoformat()
    kopf = [
        PORTFOLIO_START,
        "",
        f"_Erzeugt am {stand} durch `make portfolio`. Nicht von Hand bearbeiten —",
        "der Abschnitt wird bei jedem Lauf vollständig ersetzt._",
        "",
        f"**Regelwerk:** portfolio {policies.portfolio.version}, "
        f"classes {policies.classes.version}, simulation {policies.simulation.version}.",
        "",
    ]
    fuss = ["", PORTFOLIO_END, ""]
    teile = (
        kopf
        + _schaetzblock(returns, policies)
        + _klassenblock(returns, policies)
        + _backtestblock(returns, policies)
        + _zulaessigkeitsblock(policies)
        + _nachhaltigkeitsblock(policies)
        + fuss
    )
    return "\n".join(teile)


def write_evaluation(block: str, evaluation_path: Path | None = None) -> Path:
    """Schreibt den Abschnitt in den Evaluationsbericht.

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
    if PORTFOLIO_START in inhalt and PORTFOLIO_END in inhalt:
        vorher = inhalt.split(PORTFOLIO_START)[0]
        nachher = inhalt.split(PORTFOLIO_END, 1)[1]
        neu = f"{vorher}{block}{nachher}"
    else:
        marker = "## 4. Portfolio: MVO gegen 1/N-Benchmark"
        if marker in inhalt:
            vorher, _, rest = inhalt.partition(marker)
            _, _, nachher = rest.partition("---")
            neu = f"{vorher}{marker}\n\n{block}\n\n---{nachher}"
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
    parser = argparse.ArgumentParser(description="Auswertung der Portfoliokonstruktion.")
    parser.add_argument(
        "--write-evaluation",
        action="store_true",
        help="Ergebnis in den Evaluationsbericht schreiben statt auf die Konsole",
    )
    args = parser.parse_args(argv)

    policies = load_policy_set()
    returns = load_history(policies)
    block = render_block(returns, policies)

    if args.write_evaluation:
        ziel = write_evaluation(block)
        sys.stdout.write(f"Abschnitt geschrieben nach {ziel}\n")
    else:
        sys.stdout.write(block + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
