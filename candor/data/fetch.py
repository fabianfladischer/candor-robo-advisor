"""Abruf von Monatsrenditen — Datenquelle hinter einem Protocol gekapselt.

Der Rest des Systems kennt ausschließlich :class:`PriceSource`. Ein Wechsel der
Datenquelle — etwa auf einen kommerziellen Anbieter, falls die derzeitige Quelle
wegfällt (Risiko R-01) — berührt damit nur dieses Modul und lässt Snapshot,
Manifest, Loader und alle Auswertungen unverändert.

Der Live-Abruf ist ausdrücklich **nicht** der Primärpfad des Systems. Er dient
allein dazu, den eingefrorenen Snapshot zu erzeugen (Prinzip P6). Jede Berechnung
im Bericht liest den Snapshot.

Verarbeitungskette je Instrument:

1. Monatsschlusskurse abrufen, ausschüttungsbereinigt.
2. Zeitstempel in die **Börsenzeitzone** umrechnen, bevor der Monat bestimmt
   wird. Die Quelle stempelt Monatskerzen auf den Monatsbeginn in Ortszeit;
   in UTC gelesen fällt die Kerze während der Sommerzeit in den Vormonat.
   Ohne diesen Schritt sind sämtliche Monatsangaben um eins verschoben.
3. Unvollständigen laufenden Monat verwerfen — sonst enthielte die Reihe eine
   Rendite über einen angebrochenen Zeitraum.
4. Fremdwährungsnotierungen über :data:`~candor.data.universe_meta.FX_SYMBOL`
   in Euro umrechnen.
5. Einfache Monatsrenditen bilden.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any, Final, Protocol

import pandas as pd
import requests

from candor.data.universe_meta import (
    UNIVERSE,
    FX_SYMBOL,
    Instrument,
)

#: Zeitüberschreitung je HTTP-Anfrage in Sekunden.
REQUEST_TIMEOUT: Final = 30

#: Wartezeit zwischen zwei Anfragen. Rücksicht auf die Datenquelle; der Abruf
#: läuft ohnehin nur beim Erzeugen des Snapshots.
REQUEST_PAUSE_SECONDS: Final = 1.5

#: Grundwartezeit nach einer Ratenbegrenzung (HTTP 429). Deutlich länger als die
#: reguläre Pause: Ein 429 ist keine Störung, sondern die ausdrückliche Bitte der
#: Quelle, langsamer zu werden. Sie wird respektiert, nicht umgangen.
RATE_LIMIT_PAUSE_SECONDS: Final = 20.0

#: Anzahl der Versuche je Symbol bei Netz- oder Serverfehlern.
MAX_ATTEMPTS: Final = 4

#: HTTP-Status der Ratenbegrenzung.
HTTP_TOO_MANY_REQUESTS: Final = 429

#: Kennung des Clients gegenüber der Quelle.
USER_AGENT: Final = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


class DataSourceError(RuntimeError):
    """Der Abruf ist fehlgeschlagen.

    Wird bewusst als harter Fehler geworfen und nicht durch Ersatzwerte
    überdeckt: Synthetische Daten als stiller Ersatz für einen gescheiterten
    Abruf würden jede spätere Auswertung entwerten (Prinzip P5).
    """


class PriceSource(Protocol):
    """Schnittstelle einer Kursdatenquelle.

    Implementierungen liefern ausschüttungsbereinigte Monatsschlusskurse in der
    Notierungswährung des Instruments.
    """

    @property
    def label(self) -> str:
        """Bezeichnung der Quelle, wie sie ins Manifest geschrieben wird."""
        ...

    def monthly_closes(self, symbol: str) -> pd.Series:
        """Ruft ausschüttungsbereinigte Monatsschlusskurse ab.

        Args:
            symbol: Symbol des Instruments bei der Quelle.

        Returns:
            Reihe der Kurse, indiziert auf Monatsenden, aufsteigend sortiert,
            ohne den unvollständigen laufenden Monat.

        Raises:
            DataSourceError: Wenn der Abruf endgültig fehlschlägt oder die
                Antwort keine verwertbare Zeitreihe enthält.
        """
        ...


class YahooFinanceSource:
    """Kursdatenquelle auf Basis der Chart-Schnittstelle von Yahoo Finance.

    Die Schnittstelle liefert neben dem Schlusskurs eine bereinigte Reihe
    (``adjclose``), in der Ausschüttungen als reinvestiert unterstellt sind.
    Sieben der acht Instrumente des Universums sind ausschüttend; ohne diese
    Bereinigung wäre die gemessene Rendite systematisch zu niedrig.
    """

    BASE_URL: Final = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def __init__(self, session: requests.Session | None = None) -> None:
        """Legt die Quelle an.

        Args:
            session: Optionale, bereits konfigurierte HTTP-Sitzung. Wird keine
                übergeben, legt die Quelle eine eigene an und setzt eine
                aussagekräftige Kennung.

        Note:
            Die Kennung wird bewusst zugewiesen und nicht über ``setdefault``
            gesetzt: ``requests.Session`` bringt bereits einen ``User-Agent``
            mit, sodass ``setdefault`` wirkungslos bliebe. Die Quelle begrenzt
            Anfragen mit der Standardkennung von ``requests`` sehr eng
            (HTTP 429), was ohne diesen Hinweis schwer zu diagnostizieren ist.
        """
        if session is not None:
            self._session = session
        else:
            self._session = requests.Session()
            self._session.headers["User-Agent"] = USER_AGENT

    @property
    def label(self) -> str:
        """Bezeichnung der Quelle für das Manifest."""
        return "Yahoo Finance (chart v8), ausschüttungsbereinigte Monatsschlusskurse"

    def monthly_closes(self, symbol: str) -> pd.Series:
        """Ruft ausschüttungsbereinigte Monatsschlusskurse ab.

        Args:
            symbol: Symbol des Instruments bei der Quelle.

        Returns:
            Kursreihe auf Monatsenden ohne den unvollständigen laufenden Monat.

        Raises:
            DataSourceError: Bei endgültigem Fehlschlag des Abrufs.
        """
        payload = self._request(symbol)
        # Pause nach jedem Netzzugriff, nicht in der Orchestrierung: Nur diese
        # Klasse spricht mit der Quelle. Eine Testquelle wartet dadurch nicht.
        time.sleep(REQUEST_PAUSE_SECONDS)
        return _to_month_end_series(payload, symbol)

    def _request(self, symbol: str) -> dict[str, Any]:
        """Führt die HTTP-Anfrage mit Wiederholungen aus.

        Args:
            symbol: Abzurufendes Symbol.

        Returns:
            Der ``result``-Block der Antwort.

        Raises:
            DataSourceError: Wenn alle Versuche scheitern.
        """
        letzter_fehler = ""
        for versuch in range(1, MAX_ATTEMPTS + 1):
            wartezeit = REQUEST_PAUSE_SECONDS * versuch
            try:
                antwort = self._session.get(
                    f"{self.BASE_URL}{symbol}",
                    params={"range": "max", "interval": "1mo"},
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.RequestException as fehler:
                letzter_fehler = f"Netzfehler: {fehler}"
            else:
                if antwort.status_code == 200:
                    inhalt = antwort.json().get("chart", {})
                    ergebnis = inhalt.get("result")
                    if ergebnis:
                        return dict(ergebnis[0])
                    letzter_fehler = f"Antwort ohne Zeitreihe: {inhalt.get('error')}"
                elif antwort.status_code == HTTP_TOO_MANY_REQUESTS:
                    letzter_fehler = "HTTP 429 (Ratenbegrenzung der Quelle)"
                    wartezeit = _retry_after(antwort) or RATE_LIMIT_PAUSE_SECONDS * versuch
                else:
                    letzter_fehler = f"HTTP {antwort.status_code}"
            if versuch < MAX_ATTEMPTS:
                time.sleep(wartezeit)
        raise DataSourceError(
            f"Abruf für {symbol!r} nach {MAX_ATTEMPTS} Versuchen fehlgeschlagen. "
            f"Letzter Fehler: {letzter_fehler}"
        )


def _retry_after(response: requests.Response) -> float | None:
    """Liest die von der Quelle gewünschte Wartezeit aus dem Antwortkopf.

    Args:
        response: Antwort mit Statuscode 429.

    Returns:
        Gewünschte Wartezeit in Sekunden, oder ``None``, wenn der Kopf fehlt
        oder unlesbar ist.
    """
    kopfwert = response.headers.get("Retry-After")
    if not kopfwert:
        return None
    try:
        return float(kopfwert)
    except ValueError:
        return None


def _to_month_end_series(payload: dict[str, Any], symbol: str) -> pd.Series:
    """Formt die Antwort in eine Kursreihe auf Monatsenden um.

    Args:
        payload: ``result``-Block der Antwort.
        symbol: Symbol, nur für Fehlermeldungen.

    Returns:
        Aufsteigend sortierte Kursreihe ohne unvollständigen laufenden Monat.

    Raises:
        DataSourceError: Wenn Zeitstempel oder bereinigte Kurse fehlen.
    """
    stamps = payload.get("timestamp")
    indicators = payload.get("indicators", {})
    adjusted = indicators.get("adjclose")
    if not stamps or not adjusted:
        raise DataSourceError(
            f"Antwort für {symbol!r} enthält keine ausschüttungsbereinigte Reihe. "
            "Ohne Bereinigung wäre die Rendite ausschüttender Anteilklassen zu niedrig."
        )

    werte = adjusted[0].get("adjclose")
    if not werte:
        raise DataSourceError(f"Antwort für {symbol!r} enthält keine Kurswerte.")

    # Die Quelle stempelt Monatskerzen auf den Monatsbeginn in BOERSENZEIT.
    # Ohne Umrechnung in diese Zeitzone landet die Kerze waehrend der Sommerzeit
    # im Vormonat: 2008-09-01 00:00 BST ist 2008-08-31 23:00 UTC. Fuer Xetra
    # (immer >= +1 h) verschoebe das jede Kerze um einen Monat; fuer London
    # (+0 h im Winter, +1 h im Sommer) entstuenden zusaetzlich Dubletten im
    # Maerz und Luecken im Oktober und November jedes Jahres.
    zeitzone = str(payload.get("meta", {}).get("exchangeTimezoneName") or "UTC")
    lokal = (
        pd.to_datetime(pd.Series(stamps, dtype="int64"), unit="s", utc=True)
        .dt.tz_convert(zeitzone)
        .dt.tz_localize(None)
    )
    reihe = pd.Series(werte, index=pd.DatetimeIndex(lokal), dtype="float64", name=symbol)
    reihe = reihe.dropna().sort_index()

    reihe.index = pd.DatetimeIndex(reihe.index).to_period("M").to_timestamp("M")
    reihe = reihe[~reihe.index.duplicated(keep="last")]
    laufender_monat = (
        pd.Timestamp(dt.datetime.now(tz=dt.UTC).date()).to_period("M").to_timestamp("M")
    )
    return reihe[reihe.index < laufender_monat]


def fetch_prices_eur(
    source: PriceSource,
    universe: tuple[Instrument, ...] = UNIVERSE,
) -> pd.DataFrame:
    """Ruft alle Instrumente ab und rechnet sie in Euro um.

    Args:
        source: Datenquelle.
        universe: Abzurufende Instrumente. Vorgabe ist das Projektuniversum.

    Returns:
        Kurstabelle in Euro. Zeilen sind Monatsenden, Spalten die
        Assetklassen-Kennungen in Universumsreihenfolge.

    Raises:
        DataSourceError: Wenn ein Abruf fehlschlägt oder für ein
            fremdwährungsnotiertes Instrument kein Wechselkurs vorliegt.
    """
    spalten: dict[str, pd.Series] = {}
    fx: pd.Series | None = None

    if any(instrument.needs_fx_conversion for instrument in universe):
        fx = source.monthly_closes(FX_SYMBOL)
        if fx.empty:
            raise DataSourceError(f"Wechselkursreihe {FX_SYMBOL!r} ist leer.")

    for instrument in universe:
        reihe = source.monthly_closes(instrument.symbol)
        if reihe.empty:
            raise DataSourceError(f"Kursreihe für {instrument.symbol!r} ist leer.")
        if instrument.needs_fx_conversion:
            if fx is None:  # pragma: no cover — durch die Prüfung oben ausgeschlossen
                raise DataSourceError("Wechselkursreihe fehlt.")
            # EURUSD=X notiert US-Dollar je Euro.
            reihe = reihe / fx.reindex(reihe.index)
            reihe = reihe.dropna()
        spalten[instrument.key] = reihe.rename(instrument.key)

    tabelle = pd.DataFrame(spalten)
    tabelle.index.name = "month_end"
    return tabelle[list(spalten)]


def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Bildet einfache Monatsrenditen aus einer Kurstabelle.

    Args:
        prices: Kurstabelle mit Monatsenden als Index.

    Returns:
        Renditetabelle. Die erste Zeile der Kurstabelle entfällt, da für sie
        keine Vorperiode existiert.
    """
    returns = prices.sort_index().pct_change()
    returns = returns.iloc[1:]
    returns.index.name = "month_end"
    return returns
