"""Historische Stresstests der fünf Risikoklassen.

Die Frage, um die es geht
-------------------------
Das System sagt einem Kunden der Klasse 3 einen zwischenzeitlichen Rückgang von
20 Prozent zu. **Hielt diese Zusage 2008?** Die Antwort ist kein Nebenergebnis,
sondern der Kern der Wirksamkeitsprüfung: Eine Verlusterwartung, die in der
schwersten beobachteten Krise nicht hält, ist keine Erwartung, sondern eine
Beschönigung.

Anders als die Zielerreichungssimulation zieht dieses Modul **nichts**. Es rechnet
die tatsächlichen Monatsrenditen der Krisenfenster auf die Referenzallokation
jeder Klasse. Kein Seed, kein Bootstrap, keine Verteilungsannahme — nur das, was
geschehen ist.

Warum die Referenzallokation und nicht das optimierte Portfolio
---------------------------------------------------------------
Die optimierte Allokation hängt an geschätzten Momenten, die ihrerseits aus
einem Zeitfenster stammen. Ein Stresstest darauf würde messen, wie gut die
Schätzung das Krisenfenster kannte — nicht, wie die Klasse sich
verhalten hätte. Die regelbasierte Referenzallokation kommt ohne jede Schätzung
aus und ist damit die ehrlichere Grundlage. Zum Vergleich wird die optimierte
Allokation zusätzlich ausgewiesen; die Differenz gehört in die Auswertung.

Was die Fenster nicht abdecken
------------------------------
Der Snapshot beginnt im **Februar 2008**. Die Finanzkrise begann früher; die
Monate Oktober 2007 bis Januar 2008 fehlen und mit ihnen der erste Teil des
Rückgangs. Der ausgewiesene Drawdown von 2008 ist damit eine **Untergrenze**.
Das Modul weist die tatsächlich verfügbare Fensterlänge aus, statt die
angeforderte zu behaupten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from candor.domain.errors import DomainError
from candor.domain.policy import PolicySet
from candor.portfolio.universe import reference_weights

FloatArray = npt.NDArray[np.float64]

#: Monate je Jahr.
MONTHS_PER_YEAR: Final = 12

#: Höchstzahl der Monate, die für die Erholung geprüft wird. Wird der alte
#: Höchststand innerhalb dieser Frist nicht erreicht, gilt die Erholung als
#: nicht eingetreten — eine offene Suche würde bis ans Ende der Reihe laufen
#: und den Wert von der Länge des Snapshots abhängig machen.
MAX_RECOVERY_MONTHS: Final = 120


class StressTestError(DomainError):
    """Der Stresstest lässt sich mit den vorliegenden Daten nicht durchführen."""


@dataclass(frozen=True, slots=True)
class CrisisWindow:
    """Ein historisches Krisenfenster.

    Attributes:
        key: Kurzkennung, etwa ``"finanzkrise"``.
        label: Bezeichnung für Tabellen und Abbildungen.
        start: Erster Monat des Fensters im Format ``YYYY-MM``.
        end: Letzter Monat des Fensters im Format ``YYYY-MM``.
        description: Was in diesem Fenster geschah.
    """

    key: str
    label: str
    start: str
    end: str
    description: str


#: Die drei geprüften Krisenfenster.
CRISIS_WINDOWS: Final[tuple[CrisisWindow, ...]] = (
    CrisisWindow(
        key="finanzkrise",
        label="Finanzkrise 2007–2009",
        start="2007-10",
        end="2009-03",
        description=(
            "Vom Höchststand der Aktienmärkte im Oktober 2007 bis zum Tiefpunkt im März 2009. "
            "Der schwerste Rückgang des Beobachtungszeitraums."
        ),
    ),
    CrisisWindow(
        key="corona",
        label="Corona-Einbruch 2020",
        start="2020-02",
        end="2020-03",
        description=(
            "Zwei Monate. Der schnellste Rückgang der Reihe — und derjenige, der die "
            "Erholungsdauer am stärksten von der Rückgangstiefe entkoppelt."
        ),
    ),
    CrisisWindow(
        key="zinsschock",
        label="Zins- und Inflationsschock 2022",
        start="2022-01",
        end="2022-12",
        description=(
            "Der Sonderfall: Aktien und Anleihen fielen gemeinsam. Die Diversifikation über "
            "Anleihen, auf der die konservativen Klassen beruhen, wirkte nicht."
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class StressResult:
    """Ergebnis eines Krisenfensters für eine Risikoklasse.

    Attributes:
        window: Das geprüfte Fenster.
        risk_class: Geprüfte Risikoklasse.
        allocation: Kennung der verwendeten Allokation.
        max_drawdown: Größter Rückgang im Fenster, negativ.
        total_return: Gesamtrendite über das Fenster.
        months: Zahl der tatsächlich verfügbaren Monate.
        requested_months: Zahl der angeforderten Monate.
        recovery_months: Monate bis zum Wiedererreichen des Höchststands;
            ``None``, wenn er innerhalb der Frist nicht erreicht wurde.
        promised_loss: Erwarteter Maximalverlust der Klasse laut Regelwerk,
            positiv als Anteil.
    """

    window: CrisisWindow
    risk_class: int
    allocation: str
    max_drawdown: float
    total_return: float
    months: int
    requested_months: int
    recovery_months: int | None
    promised_loss: float

    @property
    def is_truncated(self) -> bool:
        """Gibt an, ob das Fenster kürzer ist als angefordert."""
        return self.months < self.requested_months

    @property
    def deviation(self) -> float:
        """Abweichung des Rückgangs von der zugesagten Erwartung in Prozentpunkten.

        Positiv bedeutet: Der tatsächliche Rückgang war **tiefer** als zugesagt.
        """
        return (abs(self.max_drawdown) - self.promised_loss) * 100.0

    @property
    def promise_held(self) -> bool:
        """Gibt an, ob die zugesagte Verlusterwartung eingehalten wurde."""
        return abs(self.max_drawdown) <= self.promised_loss


def _fenster(returns: pd.DataFrame, window: CrisisWindow) -> pd.DataFrame:
    """Schneidet das Krisenfenster aus der Renditereihe.

    Args:
        returns: Monatsrenditen mit Datumsindex.
        window: Das gewünschte Fenster.

    Returns:
        Die Zeilen des Fensters, gegebenenfalls verkürzt.

    Raises:
        StressTestError: Wenn kein einziger Monat des Fensters vorliegt.
    """
    ausschnitt = returns.loc[window.start : window.end]
    if ausschnitt.empty:
        msg = (
            f"Für das Fenster {window.label} ({window.start} bis {window.end}) liegt kein "
            f"einziger Monat vor. Der Snapshot beginnt im {returns.index[0]:%Y-%m}."
        )
        raise StressTestError(msg)
    return ausschnitt


def _angeforderte_monate(window: CrisisWindow) -> int:
    """Zählt die Monate des angeforderten Fensters.

    Args:
        window: Das Fenster.

    Returns:
        Die Zahl der Monate, Anfang und Ende eingeschlossen.
    """
    start = pd.Period(window.start, freq="M")
    ende = pd.Period(window.end, freq="M")
    return int((ende - start).n) + 1


def max_drawdown(monthly_returns: FloatArray) -> float:
    """Größter Rückgang vom bisherigen Höchststand innerhalb der Reihe.

    Args:
        monthly_returns: Monatsrenditen des Portfolios.

    Returns:
        Der Rückgang als negative Zahl; ``0.0`` ohne Rückgang.
    """
    if monthly_returns.size == 0:
        return 0.0
    index = np.cumprod(1.0 + monthly_returns)
    hoechststand = np.maximum.accumulate(index)
    return float(np.min(index / hoechststand - 1.0))


def _erholungsdauer(returns: pd.DataFrame, weights: FloatArray, window: CrisisWindow) -> int | None:
    """Ermittelt, wie viele Monate bis zum Wiedererreichen des Höchststands vergingen.

    Gerechnet wird **über das Fenster hinaus**: Der Höchststand vor dem Einbruch
    wird im Fenster bestimmt, die Erholung danach in der weiterlaufenden Reihe
    gesucht. Ein Stresstest, der die Erholung nur innerhalb des Fensters suchte,
    fände sie nie — Krisenfenster enden per Definition im Tief.

    Args:
        returns: Vollständige Monatsrenditen mit Datumsindex.
        weights: Portfoliogewichte.
        window: Das geprüfte Fenster.

    Returns:
        Die Zahl der Monate vom Tiefpunkt bis zum Wiedererreichen des
        Höchststands; ``None``, wenn er innerhalb der Frist nicht erreicht wurde.
    """
    ab_fensterbeginn = returns.loc[window.start :]
    reihe = (ab_fensterbeginn.to_numpy(dtype=np.float64) @ weights)[:MAX_RECOVERY_MONTHS]
    if reihe.size == 0:
        return None

    index = np.cumprod(1.0 + reihe)
    hoechststand = np.maximum.accumulate(index)
    relativ = index / hoechststand - 1.0
    tiefpunkt = int(np.argmin(relativ))
    if relativ[tiefpunkt] >= 0.0:
        return 0

    zu_erreichen = float(hoechststand[tiefpunkt])
    danach = index[tiefpunkt + 1 :]
    erreicht = np.flatnonzero(danach >= zu_erreichen)
    return int(erreicht[0]) + 1 if erreicht.size else None


def run_stress_test(  # noqa: PLR0913, PLR0917 — jede Angabe ist eine eigene Groesse
    returns: pd.DataFrame,
    weights: FloatArray,
    window: CrisisWindow,
    risk_class: int,
    policies: PolicySet,
    allocation: str = "Referenzallokation",
) -> StressResult:
    """Rechnet ein Krisenfenster auf eine gegebene Allokation.

    Args:
        returns: Vollständige Monatsrenditen mit Datumsindex.
        weights: Portfoliogewichte.
        window: Das geprüfte Fenster.
        risk_class: Geprüfte Risikoklasse.
        policies: Regelwerk, für die zugesagte Verlusterwartung.
        allocation: Kennung der Allokation für die Ausgabe.

    Returns:
        Das Ergebnis des Stresstests.
    """
    ausschnitt = _fenster(returns, window)
    reihe = ausschnitt.to_numpy(dtype=np.float64) @ weights
    definition = policies.classes.risikoklassen[risk_class]

    return StressResult(
        window=window,
        risk_class=risk_class,
        allocation=allocation,
        max_drawdown=max_drawdown(reihe),
        total_return=float(np.prod(1.0 + reihe) - 1.0),
        months=int(reihe.size),
        requested_months=_angeforderte_monate(window),
        recovery_months=_erholungsdauer(returns, weights, window),
        promised_loss=float(definition.erwarteter_maximalverlust),
    )


def run_all(
    returns: pd.DataFrame,
    policies: PolicySet,
    asset_keys: tuple[str, ...],
) -> tuple[StressResult, ...]:
    """Rechnet alle Krisenfenster auf alle Risikoklassen.

    Args:
        returns: Vollständige Monatsrenditen mit Datumsindex.
        policies: Vollständiges Regelwerk.
        asset_keys: Assetklassen in Spaltenreihenfolge.

    Returns:
        Die Ergebnisse, nach Fenster und Klasse geordnet.
    """
    geordnet = returns[list(asset_keys)].dropna()
    ergebnisse: list[StressResult] = []
    for fenster in CRISIS_WINDOWS:
        for klasse in sorted(policies.classes.risikoklassen):
            gewichte = reference_weights(klasse, policies.classes, asset_keys)
            ergebnisse.append(run_stress_test(geordnet, gewichte, fenster, klasse, policies))
    return tuple(ergebnisse)


def broken_promises(results: tuple[StressResult, ...]) -> tuple[StressResult, ...]:
    """Filtert die Fälle, in denen die zugesagte Verlusterwartung nicht hielt.

    Args:
        results: Ergebnisse der Stresstests.

    Returns:
        Die Fälle mit gebrochener Zusage, nach Abweichung absteigend geordnet.
    """
    gebrochen = [ergebnis for ergebnis in results if not ergebnis.promise_held]
    return tuple(sorted(gebrochen, key=lambda ergebnis: ergebnis.deviation, reverse=True))
