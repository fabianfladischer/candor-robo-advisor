"""Anlageuniversum und regelbasierte Zielgewichte je Risikoklasse.

Simulation, Beratung und Stresstest brauchen konkrete Portfoliogewichte. Dieses
Modul liefert sie als **regelbasierte Referenzallokation** aus den Bandbreiten
des Regelwerks.

Sie dient zugleich als Vergleichsmaßstab für die optimierte Allokation, weil
sie ohne jede Schätzung erwarteter Renditen auskommt und damit nicht unter der
Fehlerverstärkung der Mean-Variance-Optimierung leidet (Michaud, 1989).

Konstruktion
------------
Zwei Stufen, beide nach demselben Prinzip: Jede Position startet auf ihrer
Untergrenze; der verbleibende Rest wird **proportional zum Spielraum**
``max − min`` verteilt.

1. Über die Assetgruppen: Aktien, Anleihen, Sachwerte.
2. Innerhalb jeder Gruppe auf die einzelnen Assetklassen.

Das Verfahren hält die Bandbreiten des Regelwerks per Konstruktion ein, solange
``Σ min ≤ 1 ≤ Σ max`` gilt — was der Policy-Loader bereits beim Einlesen prüft
(Invariante I5). Es braucht keine Optimierung und keine Iteration.

Warum proportional zum Spielraum und nicht gleichgewichtet: Eine Position mit
weitem Band ist im Regelwerk als flexibel gedacht, eine mit engem Band als
festgelegt. Die Verteilung folgt dieser Absicht, statt sie einzuebnen.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final

import numpy as np
import numpy.typing as npt

from candor.domain.errors import PolicyError
from candor.domain.policy import ClassesPolicy, RiskClassDefinition

FloatArray = npt.NDArray[np.float64]

#: Toleranz beim Prüfen der Gewichtssumme.
WEIGHT_TOLERANCE: Final = 1e-9


def _fill_within_bounds(total: float, minima: FloatArray, maxima: FloatArray) -> FloatArray:
    """Verteilt eine Summe auf Positionen mit Unter- und Obergrenzen.

    Jede Position erhält zunächst ihre Untergrenze; der Rest wird proportional
    zum verbleibenden Spielraum ``max − min`` verteilt. Das Ergebnis liegt per
    Konstruktion innerhalb der Grenzen.

    Args:
        total: Zu verteilende Summe.
        minima: Untergrenzen je Position.
        maxima: Obergrenzen je Position.

    Returns:
        Die Verteilung.

    Raises:
        PolicyError: Wenn die Summe außerhalb des durch die Grenzen
            aufgespannten Bereichs liegt.
    """
    summe_min = float(minima.sum())
    summe_max = float(maxima.sum())
    if total < summe_min - WEIGHT_TOLERANCE or total > summe_max + WEIGHT_TOLERANCE:
        msg = (
            f"Die Summe {total} liegt außerhalb der Bandbreiten "
            f"[{summe_min}, {summe_max}] und ist nicht verteilbar."
        )
        raise PolicyError(msg)

    spielraum = maxima - minima
    rest = total - summe_min
    if rest <= WEIGHT_TOLERANCE or spielraum.sum() <= WEIGHT_TOLERANCE:
        return minima.copy()
    return minima + rest * spielraum / spielraum.sum()


def reference_weights(
    risk_class: int, policy: ClassesPolicy, asset_keys: tuple[str, ...]
) -> FloatArray:
    """Bildet die vorläufigen Zielgewichte einer Risikoklasse.

    Args:
        risk_class: Risikoklasse von 1 bis 5.
        policy: Regelwerk der Risikoklassen mit Gruppen- und Einzelbändern.
        asset_keys: Assetklassen in der gewünschten Spaltenreihenfolge.

    Returns:
        Gewichte in der Reihenfolge von ``asset_keys``, Summe eins.

    Raises:
        PolicyError: Wenn die Risikoklasse unbekannt ist oder die Bänder keine
            zulässige Allokation zulassen.
    """
    definition = policy.risikoklassen.get(risk_class)
    if definition is None:
        msg = f"Unbekannte Risikoklasse {risk_class}. Bekannt sind: {sorted(policy.risikoklassen)}."
        raise PolicyError(msg)

    gruppensummen = _group_totals(definition, policy)
    gewichte = dict.fromkeys(asset_keys, 0.0)

    for gruppe, mitglieder in policy.gruppenzuordnung.items():
        vorhanden = tuple(name for name in mitglieder if name in gewichte)
        if not vorhanden:
            continue
        minima = np.array([float(definition.einzelwerte[name].min) for name in vorhanden])
        maxima = np.array([float(definition.einzelwerte[name].max) for name in vorhanden])
        verteilung = _fill_within_bounds(gruppensummen[gruppe], minima, maxima)
        for name, anteil in zip(vorhanden, verteilung, strict=True):
            gewichte[name] = float(anteil)

    ergebnis = np.array([gewichte[name] for name in asset_keys], dtype=np.float64)
    summe = float(ergebnis.sum())
    if not np.isclose(summe, 1.0, atol=1e-6):
        msg = (
            f"Die Referenzallokation der Klasse {risk_class} summiert sich auf {summe} "
            "statt auf eins. Das Regelwerk ist in sich widersprüchlich."
        )
        raise PolicyError(msg)
    return ergebnis / summe


def _group_totals(definition: RiskClassDefinition, policy: ClassesPolicy) -> dict[str, float]:
    """Verteilt die Gesamtsumme eins auf die Assetgruppen.

    Args:
        definition: Definition der Risikoklasse.
        policy: Regelwerk mit der Gruppenzuordnung.

    Returns:
        Zuordnung Gruppenname auf Anteil.
    """
    gruppen = tuple(policy.gruppenzuordnung)
    minima = np.array([float(definition.gruppen[name].min) for name in gruppen])
    maxima = np.array([float(definition.gruppen[name].max) for name in gruppen])
    verteilung = _fill_within_bounds(1.0, minima, maxima)
    return dict(zip(gruppen, (float(wert) for wert in verteilung), strict=True))


def equal_weights(asset_keys: tuple[str, ...]) -> FloatArray:
    """Bildet das gleichgewichtete 1/N-Portfolio.

    Pflicht-Benchmark nach DeMiguel et al. (2009). Es ignoriert die
    Risikoklassen bewusst und dient als Vergleichsmaßstab für die
    Mean-Variance-Optimierung.

    Args:
        asset_keys: Assetklassen.

    Returns:
        Gleiche Gewichte, Summe eins.

    Raises:
        ValueError: Wenn keine Assetklassen übergeben werden.
    """
    if not asset_keys:
        msg = "Für ein 1/N-Portfolio wird mindestens eine Assetklasse benötigt."
        raise ValueError(msg)
    return np.full(len(asset_keys), 1.0 / len(asset_keys), dtype=np.float64)


def equity_share(weights: FloatArray, policy: ClassesPolicy, asset_keys: tuple[str, ...]) -> float:
    """Ermittelt die Aktienquote einer Allokation.

    Args:
        weights: Gewichte je Assetklasse.
        policy: Regelwerk mit der Gruppenzuordnung.
        asset_keys: Assetklassen in der Reihenfolge der Gewichte.

    Returns:
        Der Anteil der Gruppe ``aktien`` am Portfolio.
    """
    aktien = set(policy.gruppenzuordnung.get("aktien", ()))
    return float(
        sum(gewicht for name, gewicht in zip(asset_keys, weights, strict=True) if name in aktien)
    )


def weights_as_mapping(weights: FloatArray, asset_keys: tuple[str, ...]) -> dict[str, Decimal]:
    """Formt Gewichte in eine benannte Zuordnung für Trace und Bericht.

    Args:
        weights: Gewichte je Assetklasse.
        asset_keys: Assetklassen in der Reihenfolge der Gewichte.

    Returns:
        Zuordnung Assetklasse auf Gewicht, auf vier Stellen gerundet.
    """
    return {
        name: Decimal(str(round(float(gewicht), 4)))
        for name, gewicht in zip(asset_keys, weights, strict=True)
    }
