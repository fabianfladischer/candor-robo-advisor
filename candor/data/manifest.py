"""SHA-256-Manifest des Snapshots: Erzeugung und Verifikation.

Das Manifest ist der Reproduzierbarkeitsnachweis des Projekts (Prinzip P6). Es
hält je Snapshot-Datei den SHA-256-Hash, die Zeilenzahl und den abgedeckten
Zeitraum fest, dazu Abrufdatum, Quellenbezeichnung und die Versionen der
Pakete, mit denen der Snapshot entstanden ist.

Zweck: Wer die Ergebnisse nachrechnen will, führt `make verify-snapshot` aus und
erhält eine belastbare Aussage darüber, ob die Daten, auf denen alle Zahlen des
Berichts beruhen, unverändert sind.

Aufruf:
    python -m candor.data.manifest --verify
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: Dateiname des Manifests im Snapshot-Verzeichnis.
MANIFEST_NAME: Final = "manifest.json"

#: Fassung des Manifestformats. Ändert sich das Schema, steigt diese Zahl.
MANIFEST_SCHEMA_VERSION: Final = 1

#: Pakete, deren Version den Zahlenstand beeinflussen kann.
TRACKED_PACKAGES: Final = ("candor", "numpy", "pandas", "pyarrow", "scipy")

#: Blockgröße beim Hashen. Snapshot-Dateien sind klein; der Wert hält den
#: Speicherbedarf dennoch konstant.
CHUNK_SIZE: Final = 65_536


class ManifestError(RuntimeError):
    """Das Manifest fehlt, ist unlesbar oder unvollständig."""


@dataclass(frozen=True, slots=True)
class FileEntry:
    """Manifesteintrag einer einzelnen Snapshot-Datei.

    Attributes:
        filename: Dateiname relativ zum Snapshot-Verzeichnis.
        sha256: Hexadezimaler SHA-256-Hash des Dateiinhalts.
        rows: Zeilenzahl der enthaltenen Tabelle.
        columns: Spaltennamen der enthaltenen Tabelle.
        period_start: Erster abgedeckter Monat im Format ``YYYY-MM``.
        period_end: Letzter abgedeckter Monat im Format ``YYYY-MM``.
    """

    filename: str
    sha256: str
    rows: int
    columns: list[str]
    period_start: str
    period_end: str


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Ergebnis einer Snapshot-Prüfung.

    Attributes:
        ok: ``True``, wenn jede Datei vorhanden ist und ihr Hash übereinstimmt.
        checked: Zahl der geprüften Dateien.
        problems: Beanstandungen im Klartext, leer wenn ``ok``.
    """

    ok: bool
    checked: int
    problems: tuple[str, ...]

    def render(self) -> str:
        """Formt das Ergebnis in eine Konsolenausgabe um.

        Returns:
            Mehrzeiliger Text, dessen erste Zeile mit ``OK`` oder ``FEHLER``
            beginnt.
        """
        if self.ok:
            return f"OK — {self.checked} Datei(en) geprüft, alle Hashes stimmen überein."
        zeilen = [f"FEHLER — {len(self.problems)} Beanstandung(en) bei {self.checked} Datei(en):"]
        zeilen.extend(f"  - {problem}" for problem in self.problems)
        zeilen.append("")
        zeilen.append("Der Snapshot weicht vom Manifest ab. Alle darauf beruhenden Zahlen")
        zeilen.append("sind bis zur Klärung nicht belastbar (Prinzip P6).")
        return "\n".join(zeilen)


def sha256_of(path: Path) -> str:
    """Berechnet den SHA-256-Hash einer Datei.

    Args:
        path: Pfad der Datei.

    Returns:
        Hexadezimale Darstellung des Hashwerts.
    """
    digest = hashlib.sha256()
    with path.open("rb") as datei:
        while chunk := datei.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    """Ermittelt die Versionen der zahlenrelevanten Pakete.

    Returns:
        Zuordnung Paketname auf Version. Nicht installierte Pakete erhalten den
        Wert ``"nicht installiert"``.
    """
    versionen: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versionen[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            versionen[name] = "nicht installiert"
    versionen["python"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    return versionen


def write_manifest(
    snapshot_dir: Path,
    entries: list[FileEntry],
    source_label: str,
    universe: list[dict[str, str]],
    limitations: tuple[dict[str, str], ...] = (),
) -> Path:
    """Schreibt das Manifest in das Snapshot-Verzeichnis.

    Args:
        snapshot_dir: Zielverzeichnis des Snapshots.
        entries: Einträge der eingefrorenen Dateien.
        source_label: Bezeichnung der Datenquelle.
        universe: Beschreibung des Universums für den Nachweis, welches
            Instrument hinter welcher Spalte steht.
        limitations: Gemessene Einschränkungen des Datenstands. Sie wandern ins
            Manifest, damit eine spätere Auswertung sie nicht übersehen kann.

    Returns:
        Pfad des geschriebenen Manifests.
    """
    dokument: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "fetched_at": dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        "source": source_label,
        "package_versions": package_versions(),
        "universe": universe,
        "known_limitations": list(limitations),
        "files": [
            {
                "filename": entry.filename,
                "sha256": entry.sha256,
                "rows": entry.rows,
                "columns": entry.columns,
                "period_start": entry.period_start,
                "period_end": entry.period_end,
            }
            for entry in entries
        ],
    }
    ziel = snapshot_dir / MANIFEST_NAME
    ziel.write_text(json.dumps(dokument, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ziel


def read_manifest(snapshot_dir: Path) -> dict[str, Any]:
    """Liest das Manifest ein.

    Args:
        snapshot_dir: Snapshot-Verzeichnis.

    Returns:
        Der Manifestinhalt.

    Raises:
        ManifestError: Wenn das Manifest fehlt oder kein gültiges JSON enthält.
    """
    pfad = snapshot_dir / MANIFEST_NAME
    if not pfad.is_file():
        raise ManifestError(
            f"Kein Manifest unter {pfad}. Snapshot zuerst mit 'make snapshot' erzeugen."
        )
    try:
        inhalt = json.loads(pfad.read_text(encoding="utf-8"))
    except json.JSONDecodeError as fehler:
        raise ManifestError(f"Manifest {pfad} ist kein gültiges JSON: {fehler}") from fehler
    if not isinstance(inhalt, dict):
        raise ManifestError(f"Manifest {pfad} hat ein unerwartetes Format.")
    return inhalt


def verify_snapshot(snapshot_dir: Path | None = None) -> VerificationResult:
    """Prüft alle im Manifest verzeichneten Dateien gegen ihren Hash.

    Args:
        snapshot_dir: Snapshot-Verzeichnis. Vorgabe ist ``data/snapshot`` im
            Projektwurzelverzeichnis.

    Returns:
        Das Prüfergebnis. Fehlende Dateien und abweichende Hashes werden
        einzeln benannt.

    Raises:
        ManifestError: Wenn das Manifest selbst fehlt oder unlesbar ist.
    """
    verzeichnis = snapshot_dir if snapshot_dir is not None else default_snapshot_dir()
    manifest = read_manifest(verzeichnis)
    dateien = manifest.get("files", [])
    if not dateien:
        raise ManifestError("Manifest enthält keine Dateieinträge.")

    beanstandungen: list[str] = []
    for eintrag in dateien:
        name = str(eintrag.get("filename", "?"))
        pfad = verzeichnis / name
        if not pfad.is_file():
            beanstandungen.append(f"{name}: Datei fehlt")
            continue
        tatsaechlich = sha256_of(pfad)
        erwartet = str(eintrag.get("sha256", ""))
        if tatsaechlich != erwartet:
            beanstandungen.append(
                f"{name}: Hash weicht ab (erwartet {erwartet[:12]}…, "
                f"berechnet {tatsaechlich[:12]}…)"
            )

    return VerificationResult(
        ok=not beanstandungen,
        checked=len(dateien),
        problems=tuple(beanstandungen),
    )


def default_snapshot_dir() -> Path:
    """Liefert das Snapshot-Verzeichnis des Projekts.

    Returns:
        Pfad ``data/snapshot`` relativ zum Projektwurzelverzeichnis.
    """
    return Path(__file__).resolve().parents[2] / "data" / "snapshot"


def main(argv: list[str] | None = None) -> int:
    """Einstiegspunkt der Kommandozeile.

    Args:
        argv: Argumente ohne Programmnamen. Vorgabe ist ``sys.argv[1:]``.

    Returns:
        ``0`` bei erfolgreicher Prüfung, ``1`` bei Abweichungen, ``2`` wenn das
        Manifest fehlt oder unlesbar ist.
    """
    parser = argparse.ArgumentParser(description="Snapshot gegen das SHA-256-Manifest prüfen.")
    parser.add_argument("--verify", action="store_true", help="Prüfung ausführen")
    parser.add_argument("--snapshot-dir", type=Path, default=None, help="abweichendes Verzeichnis")
    args = parser.parse_args(argv)

    if not args.verify:
        parser.print_help()
        return 0

    try:
        ergebnis = verify_snapshot(args.snapshot_dir)
    except ManifestError as fehler:
        print(f"FEHLER — {fehler}")
        return 2

    print(ergebnis.render())
    return 0 if ergebnis.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
