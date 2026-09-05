"""Einfrieren der Zeitreihen als Parquet unter ``data/snapshot/``.

Der Snapshot ist die **Primärdatenquelle** des Projekts. Jede ausgewiesene Zahl
entsteht aus ihm, nicht aus einem Live-Abruf (Prinzip P6). Die verwendete Quelle
ist unentgeltlich, ohne Nutzungsvertrag und ohne Verfügbarkeitszusage; sie kann
jederzeit ihr Schema ändern oder wegfallen. Nur ein eingefrorener Datenstand
macht die Zahlen zu einem späteren Zeitpunkt nachrechenbar.

Erzeugte Dateien:
    prices_eur_monthly.parquet    ausschüttungsbereinigte Monatsschlusskurse in Euro
    returns_eur_monthly.parquet   einfache Monatsrenditen
    manifest.json                 SHA-256-Hashes, Zeitraum, Herkunft, Paketversionen

Ein vorhandener Snapshot wird nur mit ``--force`` überschrieben. Das ist kein
Komfortmerkmal: Ein versehentlich überschriebener Snapshot ändert rückwirkend
jede daraus abgeleitete Zahl, ohne dass es auffiele.

Aufruf:
    python -m candor.data.snapshot [--force] [--snapshot-dir PFAD]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import pandas as pd

from candor.data.fetch import (
    DataSourceError,
    PriceSource,
    YahooFinanceSource,
    fetch_prices_eur,
    to_monthly_returns,
)
from candor.data.manifest import FileEntry, default_snapshot_dir, sha256_of, write_manifest
from candor.data.universe_meta import KNOWN_LIMITATIONS, UNIVERSE, Instrument

#: Dateiname der Kursreihen im Snapshot.
PRICES_FILE: Final = "prices_eur_monthly.parquet"

#: Dateiname der Renditereihen im Snapshot.
RETURNS_FILE: Final = "returns_eur_monthly.parquet"

#: Krisenphasen, die der Snapshot abdecken muss, damit die Stresstests
#: überhaupt rechenbar sind.
REQUIRED_CRISIS_YEARS: Final = (2008, 2020, 2022)


class SnapshotError(RuntimeError):
    """Der Snapshot konnte nicht erzeugt werden oder ist unbrauchbar."""


def _period_bounds(frame: pd.DataFrame) -> tuple[str, str]:
    """Ermittelt ersten und letzten abgedeckten Monat.

    Args:
        frame: Tabelle mit Monatsenden als Index.

    Returns:
        Paar aus Anfangs- und Endmonat im Format ``YYYY-MM``.
    """
    index = pd.DatetimeIndex(frame.index)
    return index.min().strftime("%Y-%m"), index.max().strftime("%Y-%m")


def find_internal_gaps(frame: pd.DataFrame) -> list[str]:
    """Sucht Lücken innerhalb einer Reihe, also fehlende Monate nach Reihenbeginn.

    Ein fehlender Monat am Anfang ist zulässig — die Instrumente starten zu
    unterschiedlichen Zeitpunkten. Eine Lücke *innerhalb* einer laufenden Reihe
    ist dagegen immer ein Datenfehler und war hier der Nachweis dafür, dass die
    Monatsstempel falsch in die Zeitzone umgerechnet wurden.

    Args:
        frame: Kurs- oder Renditetabelle mit Monatsenden als Index.

    Returns:
        Befunde im Klartext, leer wenn keine Lücke vorliegt.
    """
    befunde: list[str] = []
    index = pd.DatetimeIndex(frame.index)
    for spalte in frame.columns:
        belegt = frame[spalte].notna()
        if not bool(belegt.any()):
            befunde.append(f"{spalte}: Reihe ist vollständig leer")
            continue
        gueltige = index[belegt.to_numpy()]
        erwartet = pd.period_range(
            gueltige.min().to_period("M"), gueltige.max().to_period("M"), freq="M"
        )
        fehlend = erwartet.difference(gueltige.to_period("M"))
        if len(fehlend) > 0:
            beispiele = ", ".join(str(monat) for monat in fehlend[:6])
            befunde.append(
                f"{spalte}: {len(fehlend)} fehlende(r) Monat(e) innerhalb der Reihe ({beispiele})"
            )
    return befunde


def check_crisis_coverage(returns: pd.DataFrame) -> list[str]:
    """Prüft die Abdeckung der geforderten Krisenjahre je Assetklasse.

    Unterschieden wird zwischen einer bloßen Einschränkung — das Krisenjahr
    liegt teilweise vor dem Reihenbeginn — und einem Defekt, also einer Lücke
    innerhalb der laufenden Reihe.

    Args:
        returns: Renditetabelle.

    Returns:
        Befunde im Klartext. Leer, wenn jedes Krisenjahr vollständig belegt ist.
    """
    befunde: list[str] = []
    index = pd.DatetimeIndex(returns.index)
    jahre = index.year
    for spalte in returns.columns:
        belegt = returns[spalte].notna()
        if not bool(belegt.any()):
            continue
        beginn = index[belegt.to_numpy()].min()
        for jahr in REQUIRED_CRISIS_YEARS:
            monate = int((belegt & (jahre == jahr)).sum())
            if monate == 12:
                continue
            if beginn.year == jahr:
                befunde.append(
                    f"{spalte}: {jahr} mit {monate} von 12 Monaten belegt "
                    f"(Reihe beginnt erst {beginn:%Y-%m} — Einschränkung, kein Defekt)"
                )
            else:
                befunde.append(f"{spalte}: {jahr} nur {monate} von 12 Monaten belegt — DEFEKT")
    return befunde


def build_snapshot(
    snapshot_dir: Path | None = None,
    source: PriceSource | None = None,
    universe: tuple[Instrument, ...] = UNIVERSE,
    *,
    force: bool = False,
) -> Path:
    """Ruft die Zeitreihen ab, friert sie ein und schreibt das Manifest.

    Args:
        snapshot_dir: Zielverzeichnis. Vorgabe ist ``data/snapshot``.
        source: Datenquelle. Vorgabe ist :class:`~candor.data.fetch.YahooFinanceSource`.
        universe: Einzufrierende Instrumente.
        force: Überschreibt einen vorhandenen Snapshot.

    Returns:
        Pfad des geschriebenen Manifests.

    Raises:
        SnapshotError: Wenn bereits ein Snapshot vorliegt und ``force`` nicht
            gesetzt ist, oder wenn die abgerufenen Reihen leer sind.
        DataSourceError: Wenn der Abruf fehlschlägt. Der Fehler wird bewusst
            nach oben durchgereicht — ein Ersatz durch synthetische Daten fände
            hier nicht statt.
    """
    ziel = snapshot_dir if snapshot_dir is not None else default_snapshot_dir()
    ziel.mkdir(parents=True, exist_ok=True)

    vorhanden = [pfad for pfad in (ziel / PRICES_FILE, ziel / RETURNS_FILE) if pfad.is_file()]
    if vorhanden and not force:
        raise SnapshotError(
            f"Snapshot besteht bereits in {ziel}. Ein Überschreiben ändert rückwirkend "
            "jede daraus abgeleitete Zahl. Bewusst erzwingen mit --force."
        )

    quelle = source if source is not None else YahooFinanceSource()
    prices = fetch_prices_eur(quelle, universe)
    if prices.empty:
        raise SnapshotError("Abruf lieferte eine leere Kurstabelle.")

    luecken = find_internal_gaps(prices)
    if luecken:
        raise SnapshotError(
            "Die abgerufenen Kursreihen weisen Lücken innerhalb der Reihe auf. "
            "Der Snapshot wird nicht geschrieben, weil alle darauf beruhenden "
            "Kennzahlen sonst still verfälscht wären:\n  - " + "\n  - ".join(luecken)
        )

    returns = to_monthly_returns(prices)
    if returns.empty:
        raise SnapshotError("Aus den Kursen ließen sich keine Renditen bilden.")

    prices.to_parquet(ziel / PRICES_FILE, engine="pyarrow", index=True)
    returns.to_parquet(ziel / RETURNS_FILE, engine="pyarrow", index=True)

    entries: list[FileEntry] = []
    for name, frame in ((PRICES_FILE, prices), (RETURNS_FILE, returns)):
        start, ende = _period_bounds(frame)
        entries.append(
            FileEntry(
                filename=name,
                sha256=sha256_of(ziel / name),
                rows=int(len(frame)),
                columns=[str(spalte) for spalte in frame.columns],
                period_start=start,
                period_end=ende,
            )
        )

    beschreibung = [
        {
            "key": instrument.key,
            "label": instrument.label,
            "fund_name": instrument.fund_name,
            "isin": instrument.isin,
            "isin_status": instrument.isin_status.value,
            "symbol": instrument.symbol,
            "exchange": instrument.exchange,
            "currency": instrument.currency,
            "wrapper": instrument.wrapper.value,
        }
        for instrument in universe
    ]
    return write_manifest(ziel, entries, quelle.label, beschreibung, KNOWN_LIMITATIONS)


def main(argv: list[str] | None = None) -> int:
    """Einstiegspunkt der Kommandozeile.

    Args:
        argv: Argumente ohne Programmnamen.

    Returns:
        ``0`` bei Erfolg, ``1`` bei Abbruch mit Fehler.
    """
    parser = argparse.ArgumentParser(description="Marktdaten abrufen und als Snapshot einfrieren.")
    parser.add_argument("--force", action="store_true", help="vorhandenen Snapshot überschreiben")
    parser.add_argument("--snapshot-dir", type=Path, default=None, help="abweichendes Verzeichnis")
    args = parser.parse_args(argv)

    try:
        manifest = build_snapshot(args.snapshot_dir, force=args.force)
    except (SnapshotError, DataSourceError) as fehler:
        print(f"FEHLER — {fehler}", file=sys.stderr)
        return 1

    ziel = manifest.parent
    returns = pd.read_parquet(ziel / RETURNS_FILE)
    start, ende = _period_bounds(returns)
    print(f"Snapshot geschrieben nach {ziel}")
    print(f"  Zeitraum:      {start} bis {ende} ({len(returns)} Monatsrenditen)")
    print(f"  Assetklassen:  {', '.join(str(spalte) for spalte in returns.columns)}")
    print(f"  Manifest:      {manifest.name}")

    befunde = check_crisis_coverage(returns)
    if befunde:
        print("\nHinweis zur Krisenabdeckung:")
        for befund in befunde:
            print(f"  - {befund}")
    else:
        print(f"  Krisenjahre {REQUIRED_CRISIS_YEARS}: vollständig abgedeckt")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
