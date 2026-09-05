"""Smoke-Test: Nachweis, dass das Projektgerüst trägt.

Der Test belegt dreierlei:

1. Das Paket ``candor`` ist installiert und importierbar.
2. Alle vorgesehenen Teilpakete existieren und lassen sich einzeln
   importieren — ein fehlendes ``__init__.py`` fällt hier auf.
3. Die Versionsangabe in ``pyproject.toml`` und ``candor.__version__`` stimmen
   überein. Das ist keine Formalie: Die Paketversion wandert in jeden Trace und
   in jede Geeignetheitserklärung und muss deshalb eindeutig sein.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

#: Teilpakete laut Architekturplan.
SUBPAKETE = [
    "candor.domain",
    "candor.profiling",
    "candor.goals",
    "candor.engine",
    "candor.portfolio",
    "candor.trace",
    "candor.report",
    "candor.data",
]

#: Verzeichnisse, die das Gerüst bereitstellen muss.
PFLICHTVERZEICHNISSE = [
    "policies",
    "candor",
    "app",
    "data/snapshot",
    "tests",
    "figures",
]


def test_paket_importierbar() -> None:
    """Das Paket ``candor`` lässt sich importieren und meldet eine Version.

    Der Import erfolgt bewusst zur Laufzeit und nicht auf Modulebene: Schlägt er
    fehl, meldet pytest einen fehlgeschlagenen Test mit klarer Ursache statt
    eines Sammelfehlers beim Einlesen der Testdatei.
    """
    candor = importlib.import_module("candor")

    assert isinstance(candor.__version__, str)
    assert candor.__version__


@pytest.mark.parametrize("modulname", SUBPAKETE)
def test_teilpaket_importierbar(modulname: str) -> None:
    """Jedes geplante Teilpaket ist ein importierbares Python-Paket.

    Args:
        modulname: Vollqualifizierter Name des Teilpakets.
    """
    modul = importlib.import_module(modulname)

    assert modul.__doc__, f"{modulname} hat keinen Modul-Docstring"


def test_version_stimmt_mit_pyproject_ueberein(project_root: Path) -> None:
    """``candor.__version__`` und die Version in ``pyproject.toml`` sind identisch.

    Args:
        project_root: Wurzelverzeichnis des Projekts.
    """
    candor = importlib.import_module("candor")

    with (project_root / "pyproject.toml").open("rb") as datei:
        pyproject = tomllib.load(datei)

    assert pyproject["project"]["version"] == candor.__version__


@pytest.mark.parametrize("verzeichnis", PFLICHTVERZEICHNISSE)
def test_projektstruktur_vollstaendig(project_root: Path, verzeichnis: str) -> None:
    """Die festgelegten Verzeichnisse existieren.

    Args:
        project_root: Wurzelverzeichnis des Projekts.
        verzeichnis: Relativer Pfad des erwarteten Verzeichnisses.
    """
    assert (project_root / verzeichnis).is_dir()


def test_paket_ist_als_typisiert_markiert(project_root: Path) -> None:
    """``candor/py.typed`` existiert.

    Ohne diese Markierung ignorieren mypy-Läufe außerhalb des Projekts die
    Typinformationen des Pakets. Die strikte Typprüfung wäre dann nur intern
    wirksam.

    Args:
        project_root: Wurzelverzeichnis des Projekts.
    """
    assert (project_root / "candor" / "py.typed").is_file()
