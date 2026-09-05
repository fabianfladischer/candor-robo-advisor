"""Anlageuniversum: acht Assetklassen, je ein real handelbares Instrument.

Auswahlkriterien, in dieser Reihenfolge angewandt:

1. **Handelbarkeit für einen deutschen Privatanleger.** Sieben der acht Instrumente
   notieren auf Xetra, das achte (Gold) an der LSE mit zusätzlicher Notierung in
   Stuttgart.
2. **Länge der verfügbaren Historie.** Ausschlaggebend war nicht die Fondsauflegung,
   sondern der Beginn der bei der Datenquelle tatsächlich vorhandenen Reihe. Für
   deutsche Notierungen beginnt sie einheitlich im Januar 2008.
3. **Physische Replikation vor Swap-Replikation.** Deshalb ``IQQW.DE`` statt
   ``DBXW.DE`` und ``IQQE.DE`` statt ``XMEM.DE``: Beide Alternativen hätten dieselbe
   Historie geliefert, tragen aber ein Kontrahentenrisiko, das in einem
   Anlagevorschlag für Privatkunden erklärungsbedürftig wäre.
4. **Notierungswährung Euro**, soweit erreichbar. Einzige Ausnahme ist Gold; die
   Reihe wird deshalb über ``EURUSD=X`` in Euro umgerechnet.

Zwei Einschränkungen sind ausdrücklich festzuhalten:

* **Gold ist kein UCITS-ETF und kann keiner sein.** Die OGAW-Richtlinie verlangt
  Risikostreuung; ein Fonds auf eine einzelne Ware ist danach nicht zulässig.
  Physisches Gold wird in Europa deshalb ausnahmslos als ETC (Exchange Traded
  Commodity) begeben — rechtlich eine besicherte Schuldverschreibung, kein
  Sondervermögen. Daraus folgt ein Emittentenrisiko, das die übrigen sieben
  Positionen nicht tragen.
* **``BD_SHORT`` bildet deutsche, nicht europaweite Staatsanleihen ab.** Eine
  euroweite Kurzläuferreihe war ab 2008 nicht verfügbar. Die Position ist damit
  frei von Peripherie-Spreads und als Liquiditätsanker konservativer, als der
  Klassenname ``Kurzläufer Euro`` vermuten lässt.

Zum Verifikationsstand der ISINs siehe ``IsinStatus``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

#: Basiswährung aller Auswertungen. Sicht eines deutschen Privatanlegers.
BASE_CURRENCY: Final = "EUR"

#: Wechselkursreihe zur Umrechnung fremdwährungsnotierter Instrumente.
#: Notierung: US-Dollar je Euro. Umrechnung daher ``kurs_eur = kurs_usd / eurusd``.
FX_SYMBOL: Final = "EURUSD=X"

#: Frühester Monat, für den die Datenquelle die deutschen Notierungen führt.
EARLIEST_AVAILABLE_MONTH: Final = "2008-01"


class IsinStatus(Enum):
    """Belegtiefe der ISIN-Zuordnung.

    Die ISINs wurden am 26.07.2026 über die Symbolsuche der Datenquelle geprüft:
    Eine Suche nach der ISIN liefert dort eine Notierung des Instruments zurück.

    Attributes:
        DIRECT: Die ISIN-Suche liefert genau das hier verwendete Symbol zurück.
        BY_NAME: Die ISIN-Suche liefert eine andere Notierung desselben Fonds
            (Amsterdam oder London). Die Zuordnung zum deutschen Symbol stützt
            sich auf die exakte Übereinstimmung des vollständigen Fondsnamens
            beider Notierungen.
        UNVERIFIED: Nicht abschließend geprüft – nicht ohne eigene
            Nachprüfung verwenden.
    """

    DIRECT = "verifiziert (Direkttreffer)"
    BY_NAME = "verifiziert (Namensidentität mit deutscher Notierung)"
    UNVERIFIED = "ZU_PRUEFEN"


class Wrapper(Enum):
    """Rechtliche Hülle des Instruments.

    Attributes:
        UCITS_ETF: OGAW-konformer Fonds, Sondervermögen, kein Emittentenrisiko.
        ETC: Besicherte Schuldverschreibung auf eine Ware. Kein Sondervermögen,
            daher Emittentenrisiko trotz Besicherung.
    """

    UCITS_ETF = "UCITS-ETF"
    ETC = "ETC"


@dataclass(frozen=True, slots=True)
class Instrument:
    """Ein Instrument des Anlageuniversums.

    Attributes:
        key: Kennung der Assetklasse, zugleich Spaltenname in allen Zeitreihen.
        label: Deutsche Bezeichnung der Assetklasse für Tabellen und Abbildungen.
        fund_name: Vollständiger Name laut Datenquelle.
        isin: Internationale Wertpapierkennnummer.
        isin_status: Belegtiefe der ISIN-Zuordnung.
        symbol: Symbol bei der Datenquelle.
        exchange: Handelsplatz der verwendeten Notierung.
        currency: Notierungswährung. Weicht sie von ``BASE_CURRENCY`` ab, wird
            die Reihe über ``FX_SYMBOL`` umgerechnet.
        wrapper: Rechtliche Hülle.
        distributing: ``True`` bei ausschüttender Anteilklasse. Für die
            Renditeberechnung ist deshalb der ausschüttungsbereinigte Kurs
            maßgeblich, nicht der Schlusskurs.
        note: Fachliche Einschränkung, die bei der Auswertung mitzudenken ist.
    """

    key: str
    label: str
    fund_name: str
    isin: str
    isin_status: IsinStatus
    symbol: str
    exchange: str
    currency: str
    wrapper: Wrapper
    distributing: bool
    note: str = ""

    @property
    def needs_fx_conversion(self) -> bool:
        """Gibt an, ob die Reihe in die Basiswährung umzurechnen ist.

        Returns:
            ``True``, wenn die Notierungswährung nicht ``BASE_CURRENCY`` ist.
        """
        return self.currency != BASE_CURRENCY


#: Das Anlageuniversum. Reihenfolge bestimmt die Spaltenreihenfolge aller
#: Zeitreihen, Tabellen und Korrelationsmatrizen: erst Aktien, dann Anleihen
#: nach fallendem Risiko, dann Sachwerte.
UNIVERSE: Final[tuple[Instrument, ...]] = (
    Instrument(
        key="EQ_DEV",
        label="Industrieländer-Aktien",
        fund_name="iShares MSCI World UCITS ETF USD (Dist)",
        isin="IE00B0M62Q58",
        isin_status=IsinStatus.BY_NAME,
        symbol="IQQW.DE",
        exchange="XETRA",
        currency="EUR",
        wrapper=Wrapper.UCITS_ETF,
        distributing=True,
    ),
    Instrument(
        key="EQ_EM",
        label="Schwellenländer-Aktien",
        fund_name="iShares MSCI EM UCITS ETF USD (Dist)",
        isin="IE00B0M63177",
        isin_status=IsinStatus.BY_NAME,
        symbol="IQQE.DE",
        exchange="XETRA",
        currency="EUR",
        wrapper=Wrapper.UCITS_ETF,
        distributing=True,
    ),
    Instrument(
        key="EQ_EUR",
        label="Europa-Aktien",
        fund_name="iShares STOXX Europe 600 UCITS ETF (DE) EUR (Dist)",
        isin="DE0002635307",
        isin_status=IsinStatus.DIRECT,
        symbol="EXSA.DE",
        exchange="XETRA",
        currency="EUR",
        wrapper=Wrapper.UCITS_ETF,
        distributing=True,
        note="Überschneidet sich konstruktionsbedingt mit EQ_DEV; der Europa-Anteil "
        "des MSCI World ist in beiden Reihen enthalten.",
    ),
    Instrument(
        key="BD_GOV",
        label="Euro-Staatsanleihen",
        fund_name="iShares Euro Government Bond Capped 1.5-10.5yr UCITS ETF (DE)",
        isin="DE000A0H0785",
        isin_status=IsinStatus.DIRECT,
        symbol="EXHF.DE",
        exchange="XETRA",
        currency="EUR",
        wrapper=Wrapper.UCITS_ETF,
        distributing=True,
    ),
    Instrument(
        key="BD_CORP",
        label="Euro-Unternehmensanleihen",
        fund_name="iShares € Corp Bond Large Cap UCITS ETF EUR (Dist)",
        isin="IE0032523478",
        isin_status=IsinStatus.BY_NAME,
        symbol="IBCS.DE",
        exchange="XETRA",
        currency="EUR",
        wrapper=Wrapper.UCITS_ETF,
        distributing=True,
    ),
    Instrument(
        key="BD_SHORT",
        label="Kurzläufer Euro",
        fund_name="iShares eb.rexx Government Germany 1.5-2.5yr UCITS ETF (DE)",
        isin="DE0006289473",
        isin_status=IsinStatus.DIRECT,
        symbol="EXHB.DE",
        exchange="XETRA",
        currency="EUR",
        wrapper=Wrapper.UCITS_ETF,
        distributing=True,
        note="Deutsche statt europaweite Staatsanleihen — eine euroweite "
        "Kurzläuferreihe war ab 2008 nicht verfügbar. Enthält damit keine "
        "Peripherie-Spreads.",
    ),
    Instrument(
        key="RE",
        label="Immobilienaktien Europa",
        fund_name="iShares STOXX Europe 600 Real Estate UCITS ETF (DE)",
        isin="DE000A0Q4R44",
        isin_status=IsinStatus.DIRECT,
        symbol="EXI5.DE",
        exchange="XETRA",
        currency="EUR",
        wrapper=Wrapper.UCITS_ETF,
        distributing=True,
        note="Immobilien-Aktiengesellschaften, keine Direktanlage in Immobilien. "
        "Die Reihe verhält sich in Krisen deutlich aktienähnlicher als der "
        "zugrunde liegende Immobilienmarkt.",
    ),
    Instrument(
        key="GOLD",
        label="Gold",
        fund_name="WisdomTree Physical Gold",
        isin="JE00B1VS3770",
        isin_status=IsinStatus.DIRECT,
        symbol="PHAU.L",
        exchange="LSE",
        currency="USD",
        wrapper=Wrapper.ETC,
        distributing=False,
        note="ETC, kein UCITS-ETF — für eine einzelne Ware nach OGAW nicht "
        "zulässig. Trägt Emittentenrisiko. In US-Dollar notiert, Umrechnung "
        "in Euro über EURUSD=X. Zusätzliche deutsche Notierung in Stuttgart "
        "(JE00B1VS3770.SG).",
    ),
)

#: Kennungen in Universumsreihenfolge.
ASSET_KEYS: Final[tuple[str, ...]] = tuple(instrument.key for instrument in UNIVERSE)


#: Bekannte Einschränkungen des Datenstands. Gemessen am 26.07.2026 gegen den
#: Ausschüttungsfeed der Quelle; die Werte sind Messergebnisse, keine Schätzungen.
#:
#: Befund: Die Quelle führt für die sieben ausschüttenden ETFs erst ab 2013 bzw.
#: 2014 Ausschüttungen. Der ausschüttungsbereinigte Kurs bereinigt davor deshalb
#: nichts — geprüft an ``EXHB.DE``, wo der Bereinigungsfaktor über 2009 konstant
#: bei 0,8541 liegt und die Renditen aus ``adjclose`` und ``close`` identisch
#: sind. Die Renditen der ersten fünf bis sechs Jahre sind damit reine
#: Kursrenditen und um die Ausschüttungsrendite **zu niedrig**.
KNOWN_LIMITATIONS: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "L-DIST",
        "measured_at": "2026-07-26",
        "summary": (
            "Ausschüttungen erst ab 2013/2014 im Feed der Quelle; Renditen der Jahre "
            "2008–2013 sind reine Kursrenditen und unterschätzen die Gesamtrendite."
        ),
        "affected": "alle Instrumente außer GOLD (ETC, schüttet nicht aus)",
        "first_distribution_year": (
            "IQQW.DE 2014 · IQQE.DE 2014 · EXSA.DE 2013 · EXHF.DE 2013 · "
            "IBCS.DE 2014 · EXHB.DE 2013 · EXI5.DE 2013 · PHAU.L keine"
        ),
    },
    {
        "id": "L-START",
        "measured_at": "2026-07-26",
        "summary": (
            "Historie beginnt 2008-02 statt der geforderten 2007-01. Der Markthöchststand "
            "vom Oktober 2007 liegt davor, weshalb die Drawdowns der Finanzkrise zu "
            "niedrig ausgewiesen werden."
        ),
        "affected": "alle Instrumente",
        "first_distribution_year": "",
    },
)


def get_instrument(key: str) -> Instrument:
    """Liefert das Instrument zu einer Assetklassen-Kennung.

    Args:
        key: Kennung der Assetklasse, etwa ``"EQ_DEV"``.

    Returns:
        Das zugehörige :class:`Instrument`.

    Raises:
        KeyError: Wenn die Kennung nicht im Universum vorkommt. Bewusst ein
            harter Fehler statt eines stillen ``None`` (Prinzip P5).
    """
    for instrument in UNIVERSE:
        if instrument.key == key:
            return instrument
    raise KeyError(f"Unbekannte Assetklasse: {key!r}. Bekannt sind: {', '.join(ASSET_KEYS)}")
