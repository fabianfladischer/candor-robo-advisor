"""Rollierender Out-of-Sample-Backtest: MVO gegen 1/N.

Aufbau
------
Rollierendes Fenster nach DeMiguel, Garlappi & Uppal (2009):

1. Schätze Momente auf den letzten ``S`` Monaten (Vorgabe 60).
2. Optimiere darauf ein Portfolio unter den Restriktionen der Risikoklasse.
3. Halte es ``H`` Monate (Vorgabe 12) und miss die **tatsächlich erzielten**
   Renditen. Innerhalb der Haltedauer driften die Gewichte mit den Kursen.
4. Schiebe das Fenster um ``H`` Monate weiter und beginne von vorn.

Der entscheidende Punkt ist die strikte Trennung: In Schritt 2 wird
ausschließlich mit Daten *vor* dem Haltezeitraum gerechnet. Ein Backtest, der
diese Trennung verletzt — etwa indem er die Momente über die gesamte Historie
schätzt —, misst nicht die Prognosegüte, sondern die Anpassungsgüte, und fällt
zwangsläufig zugunsten der Optimierung aus.

Was der Backtest nicht kann
---------------------------
* **Er misst eine einzige Historie.** Bei 220 Monaten und einem Fenster von 60
  bleiben rund 13 Haltezeiträume. Das ist eine kleine Stichprobe; Unterschiede
  in der Sharpe Ratio sind auf dieser Basis selten statistisch gesichert. Die
  Ergebnistabelle weist deshalb auch die Zahl der Perioden aus.
* **Transaktionskosten sind angesetzt, aber nicht recherchiert.** Der Umschlag
  wird ausgewiesen und mit dem Spread aus ``simulation_v1.yaml`` bewertet; die
  Spreads selbst sind eine Annahme.
* **Der Zeitraum ab 2008 war für Aktien überdurchschnittlich günstig.** Ein
  Verfahren, das stärker in Aktien geht, ist dadurch begünstigt — unabhängig
  von seiner Qualität.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from candor.domain.errors import DomainError
from candor.domain.policy import PolicySet
from candor.portfolio.benchmark import (
    PerformanceMetrics,
    equal_weight_portfolio,
    evaluate_performance,
    turnover_from_weights,
)
from candor.portfolio.estimate import estimate_moments, risk_free_monthly
from candor.portfolio.optimize import OptimizationError, optimize
from candor.portfolio.universe import reference_weights

FloatArray = npt.NDArray[np.float64]

#: Kennung der drei verglichenen Strategien.
STRATEGY_MVO: Final = "MVO"
STRATEGY_EQUAL: Final = "1/N"
STRATEGY_REFERENCE: Final = "Referenzallokation"


class BacktestError(DomainError):
    """Der Backtest lässt sich mit den vorliegenden Daten nicht durchführen."""


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """Ergebnis einer Strategie im Backtest.

    Attributes:
        name: Kennung der Strategie.
        monthly_returns: Realisierte Monatsrenditen über alle Haltezeiträume.
        metrics: Kennzahlen der Reihe.
        rebalancings: Zahl der durchgeführten Rebalancings.
        fallback_count: Wie oft das Rückfallverfahren nötig war.
        lowered_count: Wie oft die Zielrendite abgesenkt werden musste.
    """

    name: str
    monthly_returns: FloatArray
    metrics: PerformanceMetrics
    rebalancings: int
    fallback_count: int
    lowered_count: int


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Ergebnis des gesamten Backtests.

    Attributes:
        strategies: Ergebnis je Strategie.
        risk_class: Risikoklasse, deren Restriktionen galten.
        estimation_window: Länge des Schätzfensters in Monaten.
        holding_period: Haltedauer in Monaten.
        periods: Zahl der Haltezeiträume.
        risk_free_monthly: Verwendeter monatlicher risikofreier Zins.
        first_month: Index des ersten ausgewerteten Monats in der Historie.
    """

    strategies: dict[str, StrategyResult]
    risk_class: int
    estimation_window: int
    holding_period: int
    periods: int
    risk_free_monthly: float
    first_month: int

    @property
    def winner_by_sharpe(self) -> str:
        """Kennung der Strategie mit der höchsten Sharpe Ratio."""
        return max(self.strategies, key=lambda name: self.strategies[name].metrics.sharpe_ratio)

    def sharpe_gap(self) -> float:
        """Abstand der Sharpe Ratio von MVO zu 1/N.

        Returns:
            Positiv, wenn die MVO vorne liegt.
        """
        return (
            self.strategies[STRATEGY_MVO].metrics.sharpe_ratio
            - self.strategies[STRATEGY_EQUAL].metrics.sharpe_ratio
        )


def _drift_weights(
    weights: FloatArray, period_returns: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Hält ein Portfolio über mehrere Monate und lässt die Gewichte driften.

    Ohne Zwischenrebalancing verschieben sich die Gewichte mit den Kursen. Das
    ist kein Nebeneffekt, sondern die realistische Abbildung: Zwischen zwei
    Rebalancingterminen wird nicht gehandelt.

    Args:
        weights: Zielgewichte zu Beginn des Haltezeitraums.
        period_returns: Monatsrenditen der Assetklassen im Haltezeitraum.

    Returns:
        Die realisierten Monatsrenditen des Portfolios und die Gewichte am Ende
        des Haltezeitraums.
    """
    aktuell = weights.copy()
    realisiert = np.zeros(period_returns.shape[0])
    for monat in range(period_returns.shape[0]):
        rendite = float(aktuell @ period_returns[monat])
        realisiert[monat] = rendite
        gewachsen = aktuell * (1.0 + period_returns[monat])
        summe = float(gewachsen.sum())
        aktuell = gewachsen / summe if summe > 0.0 else aktuell
    return realisiert, aktuell


def run_backtest(
    returns: FloatArray,
    policies: PolicySet,
    asset_keys: tuple[str, ...],
    risk_class: int | None = None,
) -> BacktestResult:
    """Führt den rollierenden Out-of-Sample-Backtest durch.

    Verglichen werden drei Strategien auf **derselben** Renditereihe und
    denselben Zeitpunkten: die Mean-Variance-Optimierung, das gleichgewichtete
    1/N-Portfolio und die regelbasierte Referenzallokation. Die dritte ist kein
    Pflichtbestandteil, aber aufschlussreich: Sie kommt wie 1/N ohne Schätzung
    erwarteter Renditen aus, hält sich aber an die Bänder der Risikoklasse.

    Args:
        returns: Renditematrix der Form ``(monate, assetklassen)``.
        policies: Vollständiges Regelwerk.
        asset_keys: Assetklassen in Spaltenreihenfolge.
        risk_class: Abweichende Risikoklasse. Ohne Angabe die aus der Policy.

    Returns:
        Das Ergebnis des Backtests.

    Raises:
        BacktestError: Wenn die Historie für Fenster und Haltedauer zu kurz ist.
    """
    einstellungen = policies.portfolio.backtest
    fenster = einstellungen.schaetzfenster_monate
    haltedauer = einstellungen.haltedauer_monate
    klasse = risk_class if risk_class is not None else einstellungen.risikoklasse
    monate = int(returns.shape[0])

    if monate < fenster + haltedauer:
        msg = (
            f"Die Historie umfasst {monate} Monate; für ein Schätzfenster von {fenster} "
            f"und eine Haltedauer von {haltedauer} Monaten sind mindestens "
            f"{fenster + haltedauer} nötig."
        )
        raise BacktestError(msg)

    startpunkte = list(range(fenster, monate - haltedauer + 1, haltedauer))
    rf = risk_free_monthly(returns, policies.portfolio, asset_keys)

    # Spread je Assetklasse aus dem Regelwerk. Er belastet jede Umschichtung und
    # ist damit die Größe, über die sich ein hoher Umschlag rächt. Ohne diesen
    # Abzug fiele der Vergleich systematisch zugunsten des Verfahrens mit dem
    # höheren Umschlag aus — also zugunsten der Optimierung.
    spreads = np.array(
        [float(policies.simulation.kosten[name].spread_einmalig) for name in asset_keys]
    )

    reihen: dict[str, list[FloatArray]] = {
        STRATEGY_MVO: [],
        STRATEGY_EQUAL: [],
        STRATEGY_REFERENCE: [],
    }
    ziele: dict[str, list[FloatArray]] = {name: [] for name in reihen}
    drifts: dict[str, list[FloatArray]] = {name: [] for name in reihen}
    letzte_drift: dict[str, FloatArray | None] = dict.fromkeys(reihen)

    gleichgewicht = equal_weight_portfolio(len(asset_keys))
    referenz = reference_weights(klasse, policies.classes, asset_keys)
    fallbacks = 0
    absenkungen = 0

    for start in startpunkte:
        schaetzfenster = returns[start - fenster : start]
        haltefenster = returns[start : start + haltedauer]

        momente = estimate_moments(schaetzfenster, policies.portfolio.schaetzung, asset_keys)
        zielrendite = float(referenz @ momente.mean)
        try:
            optimum = optimize(
                momente,
                klasse,
                policies.classes,
                policies.portfolio,
                asset_keys,
                target_return=zielrendite,
            )
        except OptimizationError as fehler:
            msg = (
                f"Die Optimierung ist im Fenster ab Monat {start} gescheitert: {fehler}. "
                "Der Backtest bricht ab, statt diese Periode auszulassen — eine "
                "ausgelassene Periode würde das Ergebnis beschönigen."
            )
            raise BacktestError(msg) from fehler

        fallbacks += int(optimum.fallback_used)
        absenkungen += int(optimum.target_return_lowered)

        for name, gewichte in (
            (STRATEGY_MVO, optimum.weights),
            (STRATEGY_EQUAL, gleichgewicht),
            (STRATEGY_REFERENCE, referenz),
        ):
            vorher = letzte_drift[name]
            kosten = 0.0
            if vorher is not None:
                ziele[name].append(gewichte)
                drifts[name].append(vorher)
                kosten = float(np.abs(gewichte - vorher) @ spreads)
            realisiert, endgewichte = _drift_weights(gewichte, haltefenster)
            # Die Umschichtungskosten belasten den ersten Monat des
            # Haltezeitraums, weil dann gehandelt wird. Sie über die Periode zu
            # verteilen wäre glatter, aber falsch.
            realisiert[0] -= kosten
            reihen[name].append(realisiert)
            letzte_drift[name] = endgewichte

    jahre = (len(startpunkte) * haltedauer) / einstellungen.monate_pro_jahr
    ergebnisse: dict[str, StrategyResult] = {}
    for name, teilreihen in reihen.items():
        gesamtreihe = np.concatenate(teilreihen)
        umschlag = turnover_from_weights(ziele[name], drifts[name], jahre)
        ergebnisse[name] = StrategyResult(
            name=name,
            monthly_returns=gesamtreihe,
            metrics=evaluate_performance(gesamtreihe, rf, einstellungen.monate_pro_jahr, umschlag),
            rebalancings=len(startpunkte),
            fallback_count=fallbacks if name == STRATEGY_MVO else 0,
            lowered_count=absenkungen if name == STRATEGY_MVO else 0,
        )

    return BacktestResult(
        strategies=ergebnisse,
        risk_class=klasse,
        estimation_window=fenster,
        holding_period=haltedauer,
        periods=len(startpunkte),
        risk_free_monthly=rf,
        first_month=fenster,
    )
