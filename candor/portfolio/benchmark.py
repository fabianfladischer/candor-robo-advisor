"""Pflicht-Benchmark 1/N und die Kennzahlen des Leistungsvergleichs.

Warum 1/N Pflicht ist
---------------------
DeMiguel, Garlappi & Uppal (2009) haben vierzehn Modelle der optimalen
Portfoliowahl gegen das gleichgewichtete 1/N-Portfolio getestet — darunter
Mean-Variance, Minimum-Varianz, Bayes-Stein und mehrere Shrinkage-Varianten.
**Keines** schlug 1/N über alle Datensätze hinweg verlässlich in Sharpe Ratio,
Sicherheitsäquivalent oder Turnover. Der Grund ist nicht, dass die Modelle
falsch wären, sondern dass der Schätzfehler in den erwarteten Renditen den
theoretischen Vorteil aufzehrt: Die Autoren schätzen, dass ein Datensatz mit
25 Assets rund **3 000 Monate** Historie bräuchte, damit die Mean-Variance-Regel
1/N zuverlässig schlägt.

Daraus folgt die Regel dieses Projekts: Eine Optimierung, die nicht gegen 1/N
gemessen wurde, ist keine belegte Verbesserung, sondern eine Behauptung.
1/N ist deshalb nicht ein Vergleich unter mehreren, sondern **die** Messlatte.

1/N braucht keine Schätzung erwarteter Renditen, keine Kovarianzmatrix und
keinen Solver. Es kann damit auch nicht an einem Schätzfehler scheitern — das
ist sein eigentlicher Vorzug und der Grund, warum es so schwer zu schlagen ist.

Kennzahlen
----------
Alle Kennzahlen werden aus einer **realisierten** Renditereihe berechnet, nicht
aus den Schätzungen. Der Unterschied ist der ganze Zweck des Backtests: Die
erwartete Rendite eines optimierten Portfolios ist per Konstruktion hoch, seine
realisierte kann beliebig davon abweichen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from candor.domain.errors import DomainError

FloatArray = npt.NDArray[np.float64]

#: Quelle der Benchmarkpflicht.
BENCHMARK_SOURCE: Final = "DeMiguel, Garlappi & Uppal (2009)"

#: Mindestzahl an Monaten für eine aussagekräftige Kennzahl.
MIN_PERIODS: Final = 12


class PerformanceError(DomainError):
    """Die Renditereihe lässt keine Kennzahlenberechnung zu."""


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Kennzahlen einer realisierten Renditereihe.

    Attributes:
        annualised_return: Geometrisch annualisierte Rendite.
        annualised_volatility: Annualisierte Standardabweichung der Monatsrenditen.
        sharpe_ratio: Annualisierte Sharpe Ratio gegen den risikofreien Zins.
        max_drawdown: Größter Rückgang vom bisherigen Höchststand, negativ.
        turnover: Mittlerer jährlicher Umschlag; ``0.0`` ohne Rebalancing.
        months: Zahl der ausgewerteten Monate.
        total_return: Gesamtrendite über den gesamten Zeitraum.
    """

    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float
    months: int
    total_return: float

    def as_row(self) -> dict[str, float | int]:
        """Verdichtet die Kennzahlen für die Ergebnistabelle.

        Returns:
            Zuordnung Kennzahlname auf Wert.
        """
        return {
            "rendite_pa": self.annualised_return,
            "volatilitaet_pa": self.annualised_volatility,
            "sharpe": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "monate": self.months,
            "gesamtrendite": self.total_return,
        }


def equal_weight_portfolio(asset_count: int) -> FloatArray:
    """Bildet das gleichgewichtete 1/N-Portfolio.

    Args:
        asset_count: Zahl der Assetklassen.

    Returns:
        Gewichte von je ``1/N``.

    Raises:
        PerformanceError: Wenn keine Assetklasse übergeben wird.
    """
    if asset_count < 1:
        msg = "Für ein 1/N-Portfolio wird mindestens eine Assetklasse benötigt."
        raise PerformanceError(msg)
    return np.full(asset_count, 1.0 / asset_count, dtype=np.float64)


def max_drawdown(monthly_returns: FloatArray) -> float:
    """Größter Rückgang vom bisherigen Höchststand.

    Gerechnet wird auf dem kumulierten Wertindex, nicht auf den Monatsrenditen.
    Der größte Monatsverlust und der größte Drawdown sind verschiedene Größen:
    Ein Drawdown kann sich über viele Monate aufbauen, in denen kein einzelner
    besonders schlecht war.

    Args:
        monthly_returns: Realisierte Monatsrenditen.

    Returns:
        Der größte Rückgang als negative Zahl; ``0.0`` ohne Rückgang.
    """
    if monthly_returns.size == 0:
        return 0.0
    index = np.cumprod(1.0 + monthly_returns)
    hoechststand = np.maximum.accumulate(index)
    return float(np.min(index / hoechststand - 1.0))


def turnover_from_weights(
    target_weights: list[FloatArray], drifted_weights: list[FloatArray], years: float
) -> float:
    """Mittlerer jährlicher Umschlag nach DeMiguel et al. (2009).

    Der Umschlag misst, wie viel des Portfolios bei jedem Rebalancing bewegt
    wird. Verglichen wird das **neue Zielgewicht** mit dem Gewicht, das die
    Position durch die Kursentwicklung seit dem letzten Rebalancing erreicht hat
    — nicht mit dem alten Zielgewicht. Sonst würde die Drift unterschlagen und
    der Umschlag systematisch zu niedrig ausgewiesen.

    Args:
        target_weights: Zielgewichte je Rebalancing, ab dem zweiten.
        drifted_weights: Gewichte unmittelbar vor demselben Rebalancing.
        years: Länge des Auswertungszeitraums in Jahren.

    Returns:
        Der mittlere jährliche Umschlag; ``0.0`` ohne Rebalancing.

    Raises:
        PerformanceError: Wenn die beiden Listen verschieden lang sind.
    """
    if len(target_weights) != len(drifted_weights):
        msg = (
            f"Zielgewichte ({len(target_weights)}) und driftende Gewichte "
            f"({len(drifted_weights)}) müssen gleich viele Einträge haben."
        )
        raise PerformanceError(msg)
    if not target_weights or years <= 0.0:
        return 0.0
    gesamt = sum(
        float(np.abs(ziel - drift).sum())
        for ziel, drift in zip(target_weights, drifted_weights, strict=True)
    )
    return gesamt / years


def evaluate_performance(
    monthly_returns: FloatArray,
    risk_free_monthly: float,
    months_per_year: int,
    turnover: float = 0.0,
) -> PerformanceMetrics:
    """Berechnet alle Kennzahlen einer realisierten Renditereihe.

    Die Annualisierung der Rendite ist **geometrisch** (Verkettung), die der
    Volatilität folgt der Wurzel-aus-der-Zeit-Regel. Beide Konventionen sind
    üblich, aber nicht ineinander überführbar; sie werden deshalb ausdrücklich
    benannt.

    Args:
        monthly_returns: Realisierte Monatsrenditen.
        risk_free_monthly: Monatlicher risikofreier Zins.
        months_per_year: Monate je Jahr.
        turnover: Mittlerer jährlicher Umschlag.

    Returns:
        Die Kennzahlen.

    Raises:
        PerformanceError: Wenn die Reihe zu kurz ist oder nicht endliche Werte
            enthält.
    """
    if monthly_returns.ndim != 1:
        msg = f"Erwartet wird eine eindimensionale Reihe, erhalten {monthly_returns.shape}."
        raise PerformanceError(msg)
    if monthly_returns.size < MIN_PERIODS:
        msg = (
            f"Nur {monthly_returns.size} Monate; mindestens {MIN_PERIODS} sind für eine "
            "annualisierte Kennzahl nötig."
        )
        raise PerformanceError(msg)
    if not np.all(np.isfinite(monthly_returns)):
        msg = "Die Renditereihe enthält nicht endliche Werte."
        raise PerformanceError(msg)

    monate = int(monthly_returns.size)
    jahre = monate / months_per_year
    gesamt = float(np.prod(1.0 + monthly_returns) - 1.0)
    rendite_pa = float((1.0 + gesamt) ** (1.0 / jahre) - 1.0)
    vol_pa = float(monthly_returns.std(ddof=1) * np.sqrt(months_per_year))

    ueberschuss = monthly_returns - risk_free_monthly
    streuung = float(ueberschuss.std(ddof=1))
    sharpe = (
        float(ueberschuss.mean() / streuung * np.sqrt(months_per_year)) if streuung > 0.0 else 0.0
    )

    return PerformanceMetrics(
        annualised_return=rendite_pa,
        annualised_volatility=vol_pa,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown(monthly_returns),
        turnover=turnover,
        months=monate,
        total_return=gesamt,
    )


@dataclass(frozen=True, slots=True)
class SharpeComparison:
    """Ergebnis des Signifikanztests zweier Sharpe Ratios.

    Attributes:
        sharpe_a: Annualisierte Sharpe Ratio der ersten Reihe.
        sharpe_b: Annualisierte Sharpe Ratio der zweiten Reihe.
        difference: Differenz ``sharpe_a − sharpe_b``.
        z_statistic: Teststatistik nach Jobson & Korkie mit Memmel-Korrektur.
        p_value: Zweiseitiger p-Wert.
        months: Zahl der ausgewerteten Monate.
    """

    sharpe_a: float
    sharpe_b: float
    difference: float
    z_statistic: float
    p_value: float
    months: int

    def is_significant(self, level: float = 0.05) -> bool:
        """Gibt an, ob die Differenz auf dem gegebenen Niveau gesichert ist.

        Args:
            level: Signifikanzniveau.

        Returns:
            ``True``, wenn der p-Wert unter dem Niveau liegt.
        """
        return self.p_value < level


def sharpe_difference_test(
    returns_a: FloatArray,
    returns_b: FloatArray,
    risk_free_monthly: float,
    months_per_year: int,
) -> SharpeComparison:
    """Prüft, ob sich zwei Sharpe Ratios statistisch gesichert unterscheiden.

    Verwendet wird der Test von Jobson & Korkie (1981) in der von Memmel (2003)
    korrigierten Fassung — derselbe Test, den DeMiguel, Garlappi & Uppal (2009)
    für ihren Vergleich gegen 1/N heranziehen. Er berücksichtigt, dass die
    beiden Reihen aus demselben Zeitraum stammen und stark korreliert sind; ein
    einfacher Zweistichprobentest würde die Differenz für zu gesichert halten.

    Die Teststatistik lautet

        z = (sigma_b · mu_a − sigma_a · mu_b) / sqrt(theta)

    mit

        theta = (1/T) · [ 2·sigma_a²·sigma_b² − 2·sigma_a·sigma_b·sigma_ab
                          + 0,5·mu_a²·sigma_b² + 0,5·mu_b²·sigma_a²
                          − (mu_a·mu_b / (sigma_a·sigma_b))·sigma_ab² ]

    wobei mu, sigma die Momente der **Überschussrenditen** sind und sigma_ab
    ihre Kovarianz. Unter der Nullhypothese gleicher Sharpe Ratios ist z
    asymptotisch standardnormalverteilt.

    **Warum der Test hier steht.** Bei 13 Haltezeiträumen ist eine Differenz in
    der Sharpe Ratio schnell sichtbar und selten gesichert. Ohne diesen Test
    ließe sich aus dem Backtest eine Überlegenheit ablesen, die die Datenlage
    nicht trägt — genau der Fehler, vor dem DeMiguel et al. warnen.

    Args:
        returns_a: Monatsrenditen der ersten Strategie.
        returns_b: Monatsrenditen der zweiten Strategie.
        risk_free_monthly: Monatlicher risikofreier Zins.
        months_per_year: Monate je Jahr für die Annualisierung.

    Returns:
        Das Testergebnis.

    Raises:
        PerformanceError: Wenn die Reihen verschieden lang oder zu kurz sind.
    """
    if returns_a.shape != returns_b.shape:
        msg = f"Die Reihen haben verschiedene Länge: {returns_a.shape} und {returns_b.shape}."
        raise PerformanceError(msg)
    if returns_a.size < MIN_PERIODS:
        msg = f"Nur {returns_a.size} Monate; mindestens {MIN_PERIODS} sind nötig."
        raise PerformanceError(msg)

    ueber_a = returns_a - risk_free_monthly
    ueber_b = returns_b - risk_free_monthly
    anzahl = int(ueber_a.size)
    wurzel = float(np.sqrt(months_per_year))

    mu_a, mu_b = float(ueber_a.mean()), float(ueber_b.mean())
    sig_a, sig_b = float(ueber_a.std(ddof=1)), float(ueber_b.std(ddof=1))
    if sig_a <= 0.0 or sig_b <= 0.0:
        msg = "Eine der Reihen hat keine Streuung; eine Sharpe Ratio ist nicht definiert."
        raise PerformanceError(msg)

    kovarianz = float(np.cov(ueber_a, ueber_b, ddof=1)[0, 1])
    theta = (
        2.0 * sig_a**2 * sig_b**2
        - 2.0 * sig_a * sig_b * kovarianz
        + 0.5 * mu_a**2 * sig_b**2
        + 0.5 * mu_b**2 * sig_a**2
        - (mu_a * mu_b / (sig_a * sig_b)) * kovarianz**2
    ) / anzahl

    if theta <= 0.0:
        z_wert, p_wert = 0.0, 1.0
    else:
        z_wert = (sig_b * mu_a - sig_a * mu_b) / float(np.sqrt(theta))
        # Zweiseitiger p-Wert der Standardnormalverteilung:
        # P(|Z| > |z|) = erfc(|z| / sqrt(2))
        p_wert = float(math.erfc(abs(z_wert) / math.sqrt(2.0)))

    sharpe_a = mu_a / sig_a * wurzel
    sharpe_b = mu_b / sig_b * wurzel
    return SharpeComparison(
        sharpe_a=sharpe_a,
        sharpe_b=sharpe_b,
        difference=sharpe_a - sharpe_b,
        z_statistic=z_wert,
        p_value=p_wert,
        months=anzahl,
    )
