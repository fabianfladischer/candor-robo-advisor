"""Basis-Fixtures und Testkonfiguration für die gesamte Suite.

Zwei Festlegungen gelten projektweit:

1. **Determinismus (Prinzip P1).** Jede Zufallsquelle erhält einen explizit
   übergebenen Seed. Tests ziehen ihren Generator ausschließlich über die
   Fixture ``rng``; ein globales ``numpy.random.seed()`` kommt nicht vor.
2. **Pfade statt Konstanten.** Verzeichnisse werden über Fixtures bezogen,
   damit kein Test relative Pfade gegen das Arbeitsverzeichnis auflöst.

Das hypothesis-Profil lässt sich über die Umgebungsvariable ``HYPOTHESIS_PROFILE``
umschalten (``schnell`` | ``standard`` | ``gruendlich``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, settings

#: Basis-Seed des Projekts. Geht in Simulation und Traces ein.
PROJECT_SEED = 20260726

# --- hypothesis-Profile ----------------------------------------------------
# `deadline=None`, weil numerische Verfahren (Bootstrap, Solver) einzelne
# Beispiele stark unterschiedlich lange rechnen; eine Zeitschranke pro Beispiel
# würde sonst zu Fehlalarmen führen.

settings.register_profile("schnell", max_examples=25, deadline=None)
settings.register_profile("standard", max_examples=100, deadline=None)
settings.register_profile(
    "gruendlich",
    max_examples=1000,
    deadline=None,
    phases=tuple(Phase),
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "standard"))


# --- Verzeichnisse ---------------------------------------------------------


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Wurzelverzeichnis des Projekts (Elternverzeichnis von ``tests/``)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def policies_dir(project_root: Path) -> Path:
    """Verzeichnis der versionierten YAML-Policies (Prinzip P3)."""
    return project_root / "policies"


@pytest.fixture(scope="session")
def snapshot_dir(project_root: Path) -> Path:
    """Verzeichnis des eingefrorenen Marktdaten-Snapshots (Prinzip P6)."""
    return project_root / "data" / "snapshot"


@pytest.fixture(scope="session")
def figures_dir(project_root: Path) -> Path:
    """Ausgabeverzeichnis der Abbildungen (PNG, 300 dpi, je mit .caption.txt)."""
    return project_root / "figures"


@pytest.fixture(scope="session")
def docs_dir(project_root: Path) -> Path:
    """Verzeichnis der Projektdokumentation."""
    return project_root / "docs"


# --- Determinismus ---------------------------------------------------------


@pytest.fixture(scope="session")
def seed() -> int:
    """Basis-Seed des Projekts.

    Returns:
        Der projektweit festgelegte Seed 20260726.
    """
    return PROJECT_SEED


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    """Frischer Zufallsgenerator mit dem Projekt-Seed.

    Funktionsbezogen (nicht sitzungsweit), damit jeder Test denselben
    Ausgangszustand sieht und die Reihenfolge der Tests das Ergebnis nicht
    beeinflusst.

    Args:
        seed: Basis-Seed des Projekts.

    Returns:
        Ein ``numpy.random.Generator`` gemäß Prinzip P1.
    """
    return np.random.default_rng(seed)


# --- Arbeitsverzeichnisse für schreibende Tests ----------------------------


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """Temporäres Ausgabeverzeichnis für Tests, die Dateien schreiben.

    Args:
        tmp_path: Von pytest bereitgestelltes temporäres Verzeichnis.

    Returns:
        Ein existierendes, leeres Verzeichnis außerhalb des Projektbaums.
    """
    target = tmp_path / "out"
    target.mkdir()
    return target
