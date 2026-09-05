"""Schätzung von Erwartungswert und Kovarianz der Assetklassenrenditen.

Dieses Modul liefert die beiden Eingangsgrößen der Mean-Variance-Optimierung.
Beide sind **Schätzungen**, und die Qualität der Optimierung hängt fast
vollständig an ihnen – das ist der Kern der Kritik an der MVO.

Warum je zwei Verfahren
-----------------------
Für beide Momente ist ein naives und ein geschrumpftes Verfahren umgesetzt.
Nicht aus Vollständigkeit, sondern weil die **Differenz** die eigentliche
Aussage ist: Sie zeigt, wie weit die Optimierung von der Schätzmethode abhängt.
Ein Ergebnis, das sich zwischen historischem und geschrumpftem Erwartungswert
stark verschiebt, ist kein Ergebnis, sondern ein Artefakt.

**Erwartungswerte.** Der historische Mittelwert ist erwartungstreu, aber
extrem verrauscht: Bei 60 Monatsbeobachtungen und einer Monatsvolatilität von
4 % beträgt der Standardfehler des Monatsmittels rund 0,52 Prozentpunkte —
mehr als das Monatsmittel selbst. Die Optimierung liest diesen Fehler als
Signal und konzentriert auf die Assetklasse mit dem größten Schätzfehler nach
oben (Michaud, 1989: „error maximization“). Der geschrumpfte Schätzer nach
Jorion (1986) zieht jede Assetklasse zum Gesamtmittel und dämpft genau das.

**Kovarianz.** Die Stichprobenkovarianz ist bei *T* Beobachtungen und *N*
Assetklassen umso schlechter konditioniert, je näher *N* an *T* liegt. Ledoit &
Wolf (2004) schrumpfen sie zu einer strukturierten Zielmatrix; die Intensität
wird analytisch aus den Daten bestimmt und ist **keine** frei gesetzte Größe.
Sie wird ausgewiesen, weil sie angibt, wie wenig die Stichprobe allein hergibt.

Alle Funktionen sind rein: gleiche Renditematrix ⇒ gleiches Ergebnis. Es gibt
keine Zufallsquelle und damit auch keinen Seed (Prinzip P1).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Final

import numpy as np
import numpy.typing as npt
from sklearn.covariance import LedoitWolf

from candor.domain.errors import PolicyError
from candor.domain.policy import (
    CovarianceMethod,
    EstimationSettings,
    MeanMethod,
    PortfolioPolicy,
)

FloatArray = npt.NDArray[np.float64]

#: Monate je Jahr für die Annualisierung.
MONTHS_PER_YEAR: Final = 12

#: Kleinste zulässige Zahl an Beobachtungen. Unterhalb davon ist jede
#: Momentenschätzung Zahlenspielerei; der Fehler ist deshalb hart.
MIN_OBSERVATIONS: Final = 12

#: Erwartete Zahl der Dimensionen der Renditematrix (Monate × Assetklassen).
MATRIX_DIMENSIONS: Final = 2


class EstimationError(PolicyError):
    """Die Renditematrix lässt keine belastbare Momentenschätzung zu."""


@dataclass(frozen=True, slots=True)
class MomentEstimate:
    """Geschätzte Momente einer Renditematrix.

    Alle Größen sind **monatlich**; die Annualisierung geschieht erst bei der
    Ausgabe, damit im Rechenweg keine Vermischung entsteht.

    Attributes:
        mean: Erwartete Monatsrendite je Assetklasse.
        covariance: Kovarianzmatrix der Monatsrenditen.
        mean_method: Verwendetes Verfahren für die Erwartungswerte.
        covariance_method: Verwendetes Verfahren für die Kovarianz.
        mean_shrinkage: Schrumpfungsintensität der Erwartungswerte; ``0.0`` beim
            historischen Mittel.
        covariance_shrinkage: Schrumpfungsintensität der Kovarianz; ``0.0`` bei
            der Stichprobenkovarianz.
        observations: Zahl der zugrunde liegenden Monate.
        asset_keys: Assetklassen in Spaltenreihenfolge.
    """

    mean: FloatArray
    covariance: FloatArray
    mean_method: MeanMethod
    covariance_method: CovarianceMethod
    mean_shrinkage: float
    covariance_shrinkage: float
    observations: int
    asset_keys: tuple[str, ...]

    @property
    def annualised_mean(self) -> FloatArray:
        """Annualisierte erwartete Rendite je Assetklasse (geometrisch)."""
        return (1.0 + self.mean) ** MONTHS_PER_YEAR - 1.0

    @property
    def annualised_volatility(self) -> FloatArray:
        """Annualisierte Volatilität je Assetklasse."""
        return np.sqrt(np.diag(self.covariance) * MONTHS_PER_YEAR)

    def portfolio_return(self, weights: FloatArray) -> float:
        """Erwartete Monatsrendite eines Portfolios.

        Args:
            weights: Gewichte je Assetklasse.

        Returns:
            Die erwartete Monatsrendite.
        """
        return float(weights @ self.mean)

    def portfolio_volatility(self, weights: FloatArray) -> float:
        """Monatliche Volatilität eines Portfolios.

        Args:
            weights: Gewichte je Assetklasse.

        Returns:
            Die monatliche Standardabweichung.
        """
        varianz = float(weights @ self.covariance @ weights)
        return float(np.sqrt(max(varianz, 0.0)))

    def as_summary(self) -> dict[str, str | int | float]:
        """Verdichtet die Schätzung für Trace und Bericht.

        Returns:
            Zuordnung der wesentlichen Kennzahlen.
        """
        return {
            "erwartungswert_verfahren": self.mean_method.value,
            "kovarianz_verfahren": self.covariance_method.value,
            "schrumpfung_erwartungswert": round(self.mean_shrinkage, 6),
            "schrumpfung_kovarianz": round(self.covariance_shrinkage, 6),
            "beobachtungen": self.observations,
        }


def _pruefe_matrix(returns: FloatArray) -> None:
    """Prüft die Renditematrix auf Form und Mindestumfang.

    Args:
        returns: Renditematrix der Form ``(monate, assetklassen)``.

    Raises:
        EstimationError: Bei falscher Form, zu wenigen Beobachtungen oder
            nicht endlichen Werten.
    """
    if returns.ndim != MATRIX_DIMENSIONS:
        msg = f"Erwartet wird eine Matrix (Monate, Assetklassen), erhalten {returns.shape}."
        raise EstimationError(msg)
    monate, spalten = returns.shape
    if spalten < 1:
        msg = "Die Renditematrix enthält keine Assetklasse."
        raise EstimationError(msg)
    if monate < MIN_OBSERVATIONS:
        msg = (
            f"Nur {monate} Monatsbeobachtungen; mindestens {MIN_OBSERVATIONS} sind nötig. "
            "Eine Momentenschätzung auf kürzerer Historie ist nicht belastbar."
        )
        raise EstimationError(msg)
    if not np.all(np.isfinite(returns)):
        msg = "Die Renditematrix enthält nicht endliche Werte (NaN oder Inf)."
        raise EstimationError(msg)


def historical_mean(returns: FloatArray) -> FloatArray:
    """Arithmetisches Mittel der Monatsrenditen je Assetklasse.

    Der erwartungstreue, aber verrauschte Schätzer. Er ist der Bezugspunkt, an
    dem sich die Schrumpfung messen lassen muss.

    Args:
        returns: Renditematrix der Form ``(monate, assetklassen)``.

    Returns:
        Erwartete Monatsrendite je Assetklasse.

    Raises:
        EstimationError: Wenn die Matrix keine Schätzung zulässt.
    """
    _pruefe_matrix(returns)
    return np.asarray(returns.mean(axis=0), dtype=np.float64)


def shrunk_mean(returns: FloatArray, intensity: float | None = None) -> tuple[FloatArray, float]:
    """Zum Gesamtmittel geschrumpfter Erwartungswert nach Jorion (1986).

    Der Schätzer lautet

        mu_shrink = (1 − phi) · r_quer + phi · r_gesamt

    mit dem Gesamtmittel ``r_gesamt`` über alle Assetklassen. Ohne vorgegebene
    Intensität wird ``phi`` analytisch bestimmt als

        phi = (N + 2) / (N + 2 + T · d' Sigma^-1 d)   mit  d = r_quer − r_gesamt

    Die Formel folgt der Bayes-Stein-Herleitung: Je weiter die Einzelmittel
    auseinanderliegen (gemessen in Einheiten der Kovarianz) und je länger die
    Historie, desto weniger wird geschrumpft. Bei kurzer Historie und eng
    beieinanderliegenden Mitteln geht ``phi`` gegen eins — dann sagt die
    Stichprobe über die Unterschiede nichts aus, und der Schätzer gibt allen
    Assetklassen dieselbe erwartete Rendite.

    Args:
        returns: Renditematrix der Form ``(monate, assetklassen)``.
        intensity: Feste Intensität zwischen 0 und 1. Ohne Angabe analytisch.

    Returns:
        Der geschrumpfte Erwartungswert und die verwendete Intensität.

    Raises:
        EstimationError: Wenn die Matrix keine Schätzung zulässt oder eine
            vorgegebene Intensität außerhalb von [0, 1] liegt.
    """
    _pruefe_matrix(returns)
    if intensity is not None and not 0.0 <= intensity <= 1.0:
        msg = f"Die Schrumpfungsintensität {intensity} liegt außerhalb von [0, 1]."
        raise EstimationError(msg)

    mittel = historical_mean(returns)
    gesamtmittel = float(mittel.mean())
    abweichung = mittel - gesamtmittel

    if intensity is None:
        monate, anzahl = returns.shape
        kovarianz = np.cov(returns, rowvar=False, ddof=1)
        kovarianz = np.atleast_2d(kovarianz)
        try:
            quadratform = float(abweichung @ np.linalg.solve(kovarianz, abweichung))
        except np.linalg.LinAlgError:
            # Singuläre Kovarianz: Die Stichprobe trägt die Unterschiede nicht.
            # Vollständige Schrumpfung ist dann die ehrliche Antwort.
            quadratform = 0.0
        zaehler = anzahl + 2.0
        nenner = zaehler + monate * max(quadratform, 0.0)
        intensity = float(np.clip(zaehler / nenner, 0.0, 1.0))

    geschrumpft = (1.0 - intensity) * mittel + intensity * gesamtmittel
    return np.asarray(geschrumpft, dtype=np.float64), float(intensity)


def sample_covariance(returns: FloatArray) -> FloatArray:
    """Unbearbeitete Stichprobenkovarianz der Monatsrenditen.

    Args:
        returns: Renditematrix der Form ``(monate, assetklassen)``.

    Returns:
        Die Kovarianzmatrix.

    Raises:
        EstimationError: Wenn die Matrix keine Schätzung zulässt.
    """
    _pruefe_matrix(returns)
    return np.atleast_2d(np.cov(returns, rowvar=False, ddof=1)).astype(np.float64)


def ledoit_wolf_covariance(returns: FloatArray) -> tuple[FloatArray, float]:
    """Kovarianzschätzung mit Schrumpfung nach Ledoit & Wolf (2004).

    Die Intensität bestimmt das Verfahren selbst aus den Daten; sie ist der
    einzige Wert des Projekts, der aus ``scikit-learn`` stammt. Das Paket ist
    im Projekt ausschließlich hierfür zugelassen – der Schätzer ist ein
    geschlossener Ausdruck und kein trainiertes Modell und verletzt Prinzip P2
    deshalb nicht.

    ``assume_centered=False`` ist bewusst gesetzt: Die Renditen werden vor der
    Schätzung um ihren eigenen Mittelwert zentriert. Bei ``True`` würde gegen
    null zentriert, was die Varianz um das Quadrat des Mittelwerts überschätzt.

    Args:
        returns: Renditematrix der Form ``(monate, assetklassen)``.

    Returns:
        Die geschrumpfte Kovarianzmatrix und die verwendete Intensität.

    Raises:
        EstimationError: Wenn die Matrix keine Schätzung zulässt.
    """
    _pruefe_matrix(returns)
    schaetzer = LedoitWolf(assume_centered=False)
    schaetzer.fit(returns)
    kovarianz = np.asarray(schaetzer.covariance_, dtype=np.float64)
    return kovarianz, float(schaetzer.shrinkage_)


def estimate_moments(
    returns: FloatArray,
    settings: EstimationSettings,
    asset_keys: tuple[str, ...],
) -> MomentEstimate:
    """Schätzt beide Momente nach dem im Regelwerk gewählten Verfahren.

    Args:
        returns: Renditematrix der Form ``(monate, assetklassen)``.
        settings: Parameter der Momentenschätzung aus dem Regelwerk.
        asset_keys: Assetklassen in Spaltenreihenfolge.

    Returns:
        Die geschätzten Momente.

    Raises:
        EstimationError: Wenn die Matrix keine Schätzung zulässt oder die
            Spaltenzahl nicht zu ``asset_keys`` passt.
    """
    _pruefe_matrix(returns)
    if returns.shape[1] != len(asset_keys):
        msg = (
            f"Die Renditematrix hat {returns.shape[1]} Spalten, es wurden aber "
            f"{len(asset_keys)} Assetklassen übergeben."
        )
        raise EstimationError(msg)

    if settings.erwartungswert_verfahren is MeanMethod.SHRINKAGE:
        # ShrinkageTarget kennt derzeit genau ein Ziel; eine Fallunterscheidung
        # wäre toter Code. Kommt ein zweites Ziel hinzu, ist hier zu verzweigen.
        vorgabe = settings.erwartungswert_intensitaet
        mittel, mittel_intensitaet = shrunk_mean(
            returns, None if vorgabe is None else float(vorgabe)
        )
    else:
        mittel, mittel_intensitaet = historical_mean(returns), 0.0

    if settings.kovarianz_verfahren is CovarianceMethod.LEDOIT_WOLF:
        kovarianz, kovarianz_intensitaet = ledoit_wolf_covariance(returns)
    else:
        kovarianz, kovarianz_intensitaet = sample_covariance(returns), 0.0

    return MomentEstimate(
        mean=mittel,
        covariance=kovarianz,
        mean_method=settings.erwartungswert_verfahren,
        covariance_method=settings.kovarianz_verfahren,
        mean_shrinkage=mittel_intensitaet,
        covariance_shrinkage=kovarianz_intensitaet,
        observations=int(returns.shape[0]),
        asset_keys=asset_keys,
    )


def compare_mean_methods(
    returns: FloatArray, asset_keys: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Stellt historischen und geschrumpften Erwartungswert gegenüber.

    Die Gegenüberstellung ist der Beleg dafür, wie stark die Optimierung von
    der Schätzmethode abhängt. Sie gehört in die Auswertung, nicht nur in den Code.

    Args:
        returns: Renditematrix der Form ``(monate, assetklassen)``.
        asset_keys: Assetklassen in Spaltenreihenfolge.

    Returns:
        Zuordnung Assetklasse auf die annualisierten Werte beider Verfahren und
        ihre Differenz in Prozentpunkten.
    """
    historisch = historical_mean(returns)
    geschrumpft, _ = shrunk_mean(returns)

    def annualisiert(monatswerte: FloatArray) -> FloatArray:
        """Rechnet Monatsrenditen geometrisch auf ein Jahr hoch.

        Args:
            monatswerte: Monatliche Renditen.

        Returns:
            Die annualisierten Renditen.
        """
        return (1.0 + monatswerte) ** MONTHS_PER_YEAR - 1.0

    hist_pa = annualisiert(historisch)
    shrink_pa = annualisiert(geschrumpft)
    return {
        name: {
            "historisch_pa": float(hist_pa[i]),
            "shrinkage_pa": float(shrink_pa[i]),
            "differenz_pp": float((shrink_pa[i] - hist_pa[i]) * 100.0),
        }
        for i, name in enumerate(asset_keys)
    }


def risk_free_monthly(
    returns: FloatArray, policy: PortfolioPolicy, asset_keys: tuple[str, ...]
) -> float:
    """Ermittelt den monatlichen risikofreien Zins aus dem gewählten Proxy.

    Args:
        returns: Renditematrix der Form ``(monate, assetklassen)``.
        policy: Regelwerk der Portfoliokonstruktion.
        asset_keys: Assetklassen in Spaltenreihenfolge.

    Returns:
        Die mittlere Monatsrendite des Proxys, oder ``0.0`` ohne Proxy.

    Raises:
        EstimationError: Wenn der Proxy nicht unter den Assetklassen ist.
    """
    proxy = policy.schaetzung.risikofreier_proxy
    if proxy is None:
        return 0.0
    if proxy not in asset_keys:
        msg = f"Der risikofreie Proxy {proxy!r} fehlt in den übergebenen Assetklassen."
        raise EstimationError(msg)
    return float(returns[:, asset_keys.index(proxy)].mean())


def shrinkage_as_decimal(value: float) -> Decimal:
    """Formt eine Schrumpfungsintensität für Trace und Bericht.

    Args:
        value: Intensität zwischen 0 und 1.

    Returns:
        Die auf sechs Stellen gerundete Intensität.
    """
    return Decimal(str(round(value, 6)))
