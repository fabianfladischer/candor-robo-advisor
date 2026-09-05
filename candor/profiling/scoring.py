"""Gemeinsame Bausteine der punktebasierten Bewertung.

Tragfähigkeit und Bereitschaft verwenden dieselbe Mechanik: Ein Messwert oder
eine Antwort wird über das Regelwerk auf Punkte abgebildet, die Punkte werden
gewichtet summiert und die Summe auf eine Risikoklasse abgebildet. Die Mechanik
steht hier, die Werte im Regelwerk (Prinzip P3).

Kein Baustein dieses Moduls kennt eine Zahl aus dem Regelwerk. Wer eine Schwelle
sucht, findet sie in ``policies/*.yaml`` und nirgends sonst.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Final

from candor.domain.errors import PolicyError
from candor.domain.goal import Goal
from candor.domain.policy import CapacityDimension, ClassThreshold

#: Genauigkeit, auf die Punktbeiträge und Summen gerundet werden.
POINT_QUANT: Final = Decimal("0.01")

#: Genauigkeit der ausgewiesenen Anteile.
SHARE_QUANT: Final = Decimal("0.0001")


def share(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Bildet einen Anteil und behandelt den Nenner null ausdrücklich.

    Ein Nenner von null tritt nur in genau bestimmten Fällen auf — etwa bei einem
    Haushalt ohne Einkommen, der nach der Validierung in
    :class:`~candor.domain.client.Household` dann auch weder Schuldendienst noch
    freie Sparquote haben kann. In diesen Fällen ist der Anteil sachlich null und
    nicht undefiniert. Der Fall wird hier benannt statt in einer stillen
    Ausnahmebehandlung zu verschwinden.

    Args:
        numerator: Zähler.
        denominator: Nenner.

    Returns:
        Der Anteil, gerundet, oder null bei Nenner null.
    """
    if denominator == 0:
        return Decimal(0)
    return (numerator / denominator).quantize(SHARE_QUANT, rounding=ROUND_HALF_UP)


def planned_investment(goal: Goal) -> Decimal:
    """Ermittelt das über die Laufzeit eingezahlte Kapital.

    Summe aus Startkapital und allen geplanten Sparraten, **undiskontiert und
    ohne Rendite**. Der Wert dient zwei Zwecken, in beiden ist er bewusst grob:

    * als Bezugsgröße für die in Euro dargestellten Verlustszenarien
      (:mod:`candor.profiling.tolerance`),
    * als Zähler des Klumpenrisikos
      (:mod:`candor.profiling.capacity`).

    Für beide Zwecke wäre eine Barwertrechnung eine Scheingenauigkeit: Sie
    verlangte eine Renditeannahme, die zum Zeitpunkt der Profilierung noch gar
    nicht getroffen ist.

    Args:
        goal: Das Anlageziel.

    Returns:
        Das eingezahlte Kapital in Euro.
    """
    return goal.initial_capital + goal.monthly_contribution * Decimal(goal.horizon_months)


def resolve_band(dimension: CapacityDimension, value: Decimal) -> tuple[int, str]:
    """Bildet einen Messwert über die Punktbänder auf Punkte ab.

    Die Obergrenze ``bis`` ist einschließlich zu verstehen. Das letzte Band ist
    nach oben offen; das prüft bereits das Regelwerksmodell.

    Args:
        dimension: Die auszuwertende Dimension.
        value: Gemessener Wert.

    Returns:
        Paar aus Punktwert und Bandbezeichnung im Klartext.

    Raises:
        PolicyError: Wenn die Dimension keine Bänder führt.
    """
    if dimension.baender is None:
        raise PolicyError(f"Dimension '{dimension.beschreibung}' führt keine Punktbänder.")
    untergrenze: Decimal | None = None
    for band in dimension.baender:
        if band.bis is None or value <= band.bis:
            if band.bis is None:
                bezeichnung = f"über {untergrenze}" if untergrenze is not None else "beliebig"
            elif untergrenze is None:
                bezeichnung = f"bis {band.bis}"
            else:
                bezeichnung = f"über {untergrenze} bis {band.bis}"
            return band.punkte, bezeichnung
        untergrenze = band.bis
    raise PolicyError(  # pragma: no cover — durch das Modell ausgeschlossen
        f"Dimension '{dimension.beschreibung}': kein Band trifft auf {value} zu."
    )


def resolve_option(dimension: CapacityDimension, key: str, context: str) -> tuple[int, str]:
    """Bildet eine Ausprägung auf Punkte ab.

    Args:
        dimension: Die auszuwertende Dimension.
        key: Gewählte Ausprägung.
        context: Bezeichnung der Dimension für die Fehlermeldung.

    Returns:
        Paar aus Punktwert und Ausprägung.

    Raises:
        PolicyError: Wenn die Dimension keine Ausprägungen führt oder die
            gewählte unbekannt ist. Bewusst ein harter Fehler: Eine unbekannte
            Antwort stillschweigend mit null Punkten zu bewerten, hieße eine
            nicht erhobene Angabe wie eine erhobene zu behandeln (Prinzip P5).
    """
    if dimension.optionen is None:
        raise PolicyError(f"Dimension {context!r} führt keine Ausprägungen.")
    if key not in dimension.optionen:
        raise PolicyError(
            f"Dimension {context!r}: unbekannte Ausprägung {key!r}. "
            f"Zulässig sind: {', '.join(sorted(dimension.optionen))}."
        )
    return dimension.optionen[key], key


def weighted(points: int, weight: Decimal) -> Decimal:
    """Gewichtet einen Punktwert.

    Args:
        points: Punktwert von 0 bis 100.
        weight: Gewicht der Dimension.

    Returns:
        Der gewichtete Beitrag zur Gesamtpunktzahl.
    """
    return (Decimal(points) * weight).quantize(POINT_QUANT, rounding=ROUND_HALF_UP)


def total_points(contributions: list[Decimal]) -> Decimal:
    """Summiert die gewichteten Beiträge.

    Args:
        contributions: Die gewichteten Beiträge.

    Returns:
        Die Gesamtpunktzahl von 0 bis 100.
    """
    return sum(contributions, Decimal(0)).quantize(POINT_QUANT, rounding=ROUND_HALF_UP)


def risk_class_for_points(thresholds: tuple[ClassThreshold, ...], points: Decimal) -> int:
    """Bildet eine Gesamtpunktzahl auf eine Risikoklasse ab.

    Maßgeblich ist die höchste Schwelle, die der Punktwert erreicht. Die
    Schwellen sind laut Regelwerksmodell aufsteigend sortiert, beginnen bei null
    und führen zu aufsteigenden Klassen — die Abbildung ist damit **monoton**
    (Invarianten I1 und I2).

    Args:
        thresholds: Die Klassenschwellen des Regelwerks.
        points: Erreichte Gesamtpunktzahl.

    Returns:
        Die erreichte Risikoklasse.

    Raises:
        PolicyError: Wenn keine Schwelle greift.
    """
    erreicht: int | None = None
    for schwelle in thresholds:
        if points >= schwelle.ab_punkte:
            erreicht = schwelle.klasse
        else:
            break
    if erreicht is None:
        raise PolicyError(  # pragma: no cover — durch das Modell ausgeschlossen
            f"Keine Klassenschwelle greift für {points} Punkte."
        )
    return erreicht
