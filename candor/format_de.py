"""Deutsche Zahlformatierung — eine Regel, ein Ort.

Jede Zahl, die das System nach außen gibt, trägt ein Dezimalkomma und einen
Punkt als Tausendertrennzeichen. Die Regel gilt für Abbildungen,
Tabellen, Traces, Konfliktberichte und Faktenblätter gleichermaßen.

**Warum dieses Modul entstanden ist.** Ein Test hat aufgedeckt, dass
die Konfliktdiagnose Beträge als ``900000.00 €`` ausgab — englischer Punkt, kein
Tausenderpunkt. Die Ursache war nicht der einzelne f-String, sondern die
Verteilung der Regel über mehrere gleichlautende Hilfsfunktionen in verschiedenen
Modulen: Wer eine neue Ausgabestelle schrieb, hatte keinen offensichtlichen Ort,
von dem er sie hätte holen können. Dieses Modul ist dieser Ort.

Das Modul ist absichtlich frei von Abhängigkeiten innerhalb des Projekts. Es darf
deshalb von jeder Schicht importiert werden, auch vom Domänenkern, ohne dessen
Freiheit von Ein- und Ausgabe zu verletzen (Prinzip P4).
"""

from __future__ import annotations

from decimal import Decimal

#: Tausendertrennzeichen: geschütztes Leerzeichen nach DIN 5008. Nicht der Punkt
#: — sonst stünden in einem Dokument zwei Konventionen nebeneinander, und die
#: übrigen Ausgabestellen des Projekts verwenden bereits das Leerzeichen.
THOUSANDS_SEPARATOR = "\xa0"

#: Dezimaltrennzeichen.
DECIMAL_SEPARATOR = ","


def decimal(value: Decimal | float, decimals: int = 2, *, thousands: bool = True) -> str:
    """Formatiert eine Zahl nach deutscher Schreibweise.

    Args:
        value: Zu formatierender Wert.
        decimals: Zahl der Nachkommastellen.
        thousands: Ob Tausendertrennzeichen gesetzt werden.

    Returns:
        Die formatierte Zahl, etwa ``"1 234,56"`` mit geschütztem Leerzeichen.

    Examples:
        >>> decimal(1234.5) == "1" + THOUSANDS_SEPARATOR + "234,50"
        True
        >>> decimal(0.85, decimals=4, thousands=False)
        '0,8500'
    """
    muster = f"{{:,.{decimals}f}}" if thousands else f"{{:.{decimals}f}}"
    roh = muster.format(float(value))
    return roh.replace(",", THOUSANDS_SEPARATOR).replace(".", DECIMAL_SEPARATOR)


def euro(value: Decimal | float, decimals: int = 2) -> str:
    """Formatiert einen Geldbetrag mit Währungszeichen.

    Args:
        value: Betrag in Euro.
        decimals: Zahl der Nachkommastellen.

    Returns:
        Der formatierte Betrag, etwa ``"900 000,00 €"``.

    Examples:
        >>> euro(900000) == "900" + THOUSANDS_SEPARATOR + "000,00 €"
        True
        >>> euro(48227.11, decimals=0) == "48" + THOUSANDS_SEPARATOR + "227 €"
        True
    """
    return f"{decimal(value, decimals)} €"


def percent(value: Decimal | float, decimals: int = 2) -> str:
    """Formatiert einen Anteil als Prozentangabe.

    Args:
        value: Anteil zwischen 0 und 1.
        decimals: Zahl der Nachkommastellen.

    Returns:
        Die formatierte Angabe, etwa ``"84,56 %"``.

    Examples:
        >>> percent(0.8456)
        '84,56 %'
    """
    return f"{decimal(float(value) * 100.0, decimals, thousands=False)} %"


def percentage_points(value: Decimal | float, decimals: int = 2) -> str:
    """Formatiert eine Differenz zweier Anteile als Prozentpunkte.

    Getrennt von :func:`percent`, weil die Verwechslung von Prozent und
    Prozentpunkten eine der häufigsten Fehlangaben in Finanzberichten ist.

    Args:
        value: Differenz zweier Anteile.
        decimals: Zahl der Nachkommastellen.

    Returns:
        Die formatierte Angabe, etwa ``"5,44 Prozentpunkte"``.

    Examples:
        >>> percentage_points(0.0544)
        '5,44 Prozentpunkte'
    """
    return f"{decimal(float(value) * 100.0, decimals, thousands=False)} Prozentpunkte"


def integer(value: int | float) -> str:
    """Formatiert eine ganze Zahl mit Tausendertrennzeichen.

    Args:
        value: Zu formatierender Wert.

    Returns:
        Die formatierte Zahl, etwa ``"10 000"``.

    Examples:
        >>> integer(10000) == "10" + THOUSANDS_SEPARATOR + "000"
        True
    """
    return decimal(value, 0)
