"""Zugriff auf die Marktdaten — standardmäßig aus dem eingefrorenen Snapshot.

Dies ist die einzige Stelle, über die der Rest des Systems Marktdaten bezieht.
Der Vorgabepfad liest den Snapshot. Ein Live-Abruf findet **nur** statt, wenn er
über ``live=True`` ausdrücklich verlangt wird, und selbst dann wird der Snapshot
nicht verändert.

Warum diese Asymmetrie: Ein Robo-Advisor, dessen Empfehlung sich zwischen zwei
Aufrufen ändert, weil sich die Marktdaten geändert haben, ist nicht prüfbar.
Wo eine Zahl Monate später nachrechenbar bleiben soll, ist der eingefrorene
Stand die einzig tragfähige Grundlage (Prinzip P6).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from candor.data.fetch import PriceSource, YahooFinanceSource, fetch_prices_eur, to_monthly_returns
from candor.data.manifest import default_snapshot_dir, read_manifest
from candor.data.snapshot import PRICES_FILE, RETURNS_FILE


class SnapshotMissingError(FileNotFoundError):
    """Der Snapshot fehlt oder ist unvollständig.

    Bewusst ein harter Fehler: Ein stiller Rückfall auf einen Live-Abruf würde
    die Reproduzierbarkeit unbemerkt aufheben (Prinzip P5 und Prinzip P6).
    """


def _read(path: Path) -> pd.DataFrame:
    """Liest eine Parquet-Datei des Snapshots.

    Args:
        path: Pfad der Datei.

    Returns:
        Die eingelesene Tabelle mit Monatsenden als Index.

    Raises:
        SnapshotMissingError: Wenn die Datei nicht vorhanden ist.
    """
    if not path.is_file():
        raise SnapshotMissingError(
            f"Snapshot-Datei fehlt: {path}. Mit 'make snapshot' erzeugen. "
            "Ein Live-Abruf tritt nicht automatisch an ihre Stelle."
        )
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index.name = "month_end"
    return frame


def load_returns(
    *,
    live: bool = False,
    snapshot_dir: Path | None = None,
    source: PriceSource | None = None,
) -> pd.DataFrame:
    """Lädt die Monatsrenditen aller Assetklassen.

    Args:
        live: Ruft die Daten frisch ab, statt den Snapshot zu lesen. Ausdrücklich
            zu setzen; der Snapshot bleibt dabei unverändert.
        snapshot_dir: Abweichendes Snapshot-Verzeichnis.
        source: Datenquelle für den Live-Abruf. Nur wirksam bei ``live=True``.

    Returns:
        Renditetabelle in Euro. Zeilen sind Monatsenden, Spalten die
        Assetklassen-Kennungen.

    Raises:
        SnapshotMissingError: Wenn ``live=False`` und der Snapshot fehlt.
    """
    if live:
        quelle = source if source is not None else YahooFinanceSource()
        return to_monthly_returns(fetch_prices_eur(quelle))
    verzeichnis = snapshot_dir if snapshot_dir is not None else default_snapshot_dir()
    return _read(verzeichnis / RETURNS_FILE)


def load_prices(
    *,
    live: bool = False,
    snapshot_dir: Path | None = None,
    source: PriceSource | None = None,
) -> pd.DataFrame:
    """Lädt die ausschüttungsbereinigten Monatsschlusskurse in Euro.

    Args:
        live: Ruft die Daten frisch ab, statt den Snapshot zu lesen.
        snapshot_dir: Abweichendes Snapshot-Verzeichnis.
        source: Datenquelle für den Live-Abruf. Nur wirksam bei ``live=True``.

    Returns:
        Kurstabelle in Euro mit Monatsenden als Index.

    Raises:
        SnapshotMissingError: Wenn ``live=False`` und der Snapshot fehlt.
    """
    if live:
        quelle = source if source is not None else YahooFinanceSource()
        return fetch_prices_eur(quelle)
    verzeichnis = snapshot_dir if snapshot_dir is not None else default_snapshot_dir()
    return _read(verzeichnis / PRICES_FILE)


def snapshot_info(snapshot_dir: Path | None = None) -> dict[str, str]:
    """Liefert Herkunftsangaben des Snapshots für Traces und Berichte.

    Jede Geeignetheitserklärung soll ausweisen können, auf welchem Datenstand
    sie beruht. Der Kurzhash der Renditedatei dient dabei als Datenstand-Kennung.

    Args:
        snapshot_dir: Abweichendes Snapshot-Verzeichnis.

    Returns:
        Zuordnung mit Abrufdatum, Quellenbezeichnung, Zeitraum und Kurzhash.

    Raises:
        ManifestError: Wenn das Manifest fehlt oder unlesbar ist.
    """
    verzeichnis = snapshot_dir if snapshot_dir is not None else default_snapshot_dir()
    manifest = read_manifest(verzeichnis)
    dateien = {eintrag["filename"]: eintrag for eintrag in manifest.get("files", [])}
    renditen = dateien.get(RETURNS_FILE, {})
    return {
        "fetched_at": str(manifest.get("fetched_at", "?")),
        "source": str(manifest.get("source", "?")),
        "period_start": str(renditen.get("period_start", "?")),
        "period_end": str(renditen.get("period_end", "?")),
        "rows": str(renditen.get("rows", "?")),
        "sha256_short": str(renditen.get("sha256", ""))[:12],
    }
