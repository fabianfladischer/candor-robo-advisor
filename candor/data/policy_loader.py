"""Einlesen und Prüfen des versionierten Regelwerks aus ``policies/*.yaml``.

Das Modul liegt in der Datenschicht und nicht im Domänenkern, weil es das
Dateisystem berührt (Prinzip P4). Der Domänenkern erhält ausschließlich fertig
geprüfte Objekte aus :mod:`candor.domain.policy`.

Geprüft wird in drei Stufen, jede mit eigener Fehlermeldung:

1. **Lesbarkeit** — existiert die Datei, ist sie gültiges YAML?
2. **Schema** — sind alle Pflichtschlüssel vorhanden und keine unbekannten?
   Unbekannte Schlüssel führen zum Abbruch. Ein vertippter Schlüssel bliebe
   sonst wirkungslos, und die Regel behielte unbemerkt ihren Vorgabewert.
3. **Innere Stimmigkeit** — addieren sich Gewichte zu eins, steigt die
   Aktienquote monoton, ist jede Beschäftigungsart bewertet? Diese Prüfungen
   stecken in den Modellen selbst (siehe :mod:`candor.domain.policy`).

Aufruf:
    python -m candor.data.policy_loader
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final, TypeVar

import yaml
from pydantic import ValidationError

from candor.domain.errors import PolicyError
from candor.domain.policy import (
    CapacityPolicy,
    ClassesPolicy,
    GlidepathPolicy,
    GoalPolicy,
    KnowledgePolicy,
    PolicyHeader,
    PolicySet,
    PortfolioPolicy,
    SimulationPolicy,
    TolerancePolicy,
)

T = TypeVar("T", bound=PolicyHeader)

#: Zuordnung Policy-Name auf Dateiname und Modell. Die Namen entsprechen den
#: Feldern von :class:`~candor.domain.policy.PolicySet`.
POLICY_FILES: Final[dict[str, tuple[str, type[PolicyHeader]]]] = {
    "capacity": ("capacity_v1.yaml", CapacityPolicy),
    "tolerance": ("tolerance_v1.yaml", TolerancePolicy),
    "knowledge": ("knowledge_v1.yaml", KnowledgePolicy),
    "classes": ("classes_v1.yaml", ClassesPolicy),
    "glidepath": ("glidepath_v1.yaml", GlidepathPolicy),
    "goals": ("goals_v1.yaml", GoalPolicy),
    "simulation": ("simulation_v1.yaml", SimulationPolicy),
    "portfolio": ("portfolio_v1.yaml", PortfolioPolicy),
}


def default_policy_dir() -> Path:
    """Liefert das Regelwerksverzeichnis des Projekts.

    Returns:
        Pfad ``policies`` relativ zum Projektwurzelverzeichnis.
    """
    return Path(__file__).resolve().parents[2] / "policies"


def read_yaml(path: Path) -> dict[str, object]:
    """Liest eine YAML-Datei als Zuordnung ein.

    Args:
        path: Pfad der Datei.

    Returns:
        Der Inhalt als Zuordnung.

    Raises:
        PolicyError: Wenn die Datei fehlt, kein gültiges YAML enthält oder auf
            oberster Ebene keine Zuordnung ist.
    """
    if not path.is_file():
        raise PolicyError(f"Policy-Datei fehlt: {path}")
    try:
        inhalt = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as fehler:
        raise PolicyError(f"{path.name}: kein gültiges YAML — {fehler}") from fehler
    if inhalt is None:
        raise PolicyError(f"{path.name}: Die Datei ist leer.")
    if not isinstance(inhalt, dict):
        raise PolicyError(
            f"{path.name}: Auf oberster Ebene wird eine Zuordnung erwartet, "
            f"gefunden wurde {type(inhalt).__name__}."
        )
    return inhalt


def load_policy(path: Path, model: type[T]) -> T:
    """Liest eine einzelne Policy ein und prüft sie gegen ihr Schema.

    Args:
        path: Pfad der YAML-Datei.
        model: Erwartetes Modell.

    Returns:
        Die geprüfte Policy.

    Raises:
        PolicyError: Bei Lese-, Schema- oder Stimmigkeitsfehlern. Die Meldung
            nennt Datei und Schlüsselpfad.
    """
    rohdaten = read_yaml(path)
    try:
        return model.model_validate(rohdaten)
    except ValidationError as fehler:
        raise PolicyError(f"{path.name}: {_beschreibe_schemafehler(fehler)}") from fehler


def load_policy_set(policy_dir: Path | None = None) -> PolicySet:
    """Liest das vollständige Regelwerk ein.

    Args:
        policy_dir: Abweichendes Verzeichnis. Vorgabe ist ``policies``.

    Returns:
        Das geprüfte Regelwerk.

    Raises:
        PolicyError: Wenn eine Datei fehlt oder fehlerhaft ist.
    """
    verzeichnis = policy_dir if policy_dir is not None else default_policy_dir()
    geladen: dict[str, PolicyHeader] = {}
    for name, (dateiname, modell) in POLICY_FILES.items():
        geladen[name] = load_policy(verzeichnis / dateiname, modell)
    return PolicySet.model_validate(geladen)


def _beschreibe_schemafehler(fehler: ValidationError) -> str:
    """Formt einen pydantic-Fehler in eine lesbare Meldung um.

    Args:
        fehler: Der aufgetretene Fehler.

    Returns:
        Beschreibung mit Schlüsselpfad und Ursache je Beanstandung.
    """
    teile: list[str] = []
    for eintrag in fehler.errors()[:5]:
        pfad = ".".join(str(stufe) for stufe in eintrag["loc"]) or "<Wurzel>"
        art = eintrag.get("type", "")
        meldung = eintrag.get("msg", "unzulässiger Wert")
        if art == "extra_forbidden":
            meldung = (
                "unbekannter Schlüssel — vermutlich ein Tippfehler. Unbekannte Schlüssel "
                "werden abgewiesen, damit eine Regel nicht unbemerkt ihren Vorgabewert behält"
            )
        teile.append(f"{pfad}: {meldung}")
    weitere = len(fehler.errors()) - len(teile)
    if weitere > 0:
        teile.append(f"… und {weitere} weitere Beanstandung(en)")
    return " | ".join(teile)


def main(argv: list[str] | None = None) -> int:
    """Prüft das Regelwerk und meldet den Befund.

    Args:
        argv: Argumente ohne Programmnamen. Derzeit ohne Optionen.

    Returns:
        ``0`` bei gültigem Regelwerk, ``1`` sonst.
    """
    verzeichnis = default_policy_dir() if not argv else Path(argv[0])
    try:
        regelwerk = load_policy_set(verzeichnis)
    except PolicyError as fehler:
        sys.stdout.write(f"FEHLER — {fehler}\n")
        return 1
    sys.stdout.write(f"OK — {len(POLICY_FILES)} Policies geprüft.\n")
    for name, version in sorted(regelwerk.versions.items()):
        sys.stdout.write(f"  {name:10s} {version}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
