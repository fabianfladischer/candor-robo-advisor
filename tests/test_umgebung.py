"""Wächter gegen Drift zwischen Abhängigkeitsdeklaration und Umgebungsschnappschuss.

**Anlass.** `requirements.lock` entsteht per `pip freeze` zu einem Zeitpunkt
und friert die Umgebung ein. `pyproject.toml` dagegen wird fortgeschrieben,
sobald ein Modul eine neue Abhängigkeit braucht. Beide Dateien laufen still
auseinander: Zuerst kam `pypdf` hinzu, später `python-docx` – der Lock wurde
beide Male nicht neu erzeugt.

Bemerkt wurde das lange nicht, weil in der Arbeitsumgebung immer schon alles
installiert war. Erst ein venv, das **ausschließlich** aus dem Lock
gebaut wird, zeigt die Lücke – und genau diesen Weg geht eine fremde Maschine.

Der Test hier ist die billige Absicherung: Er läuft bei jedem `make verify` mit
und braucht keine Netzverbindung. Die teure Absicherung ist `make verify-lock`,
das ein echtes Wegwerf-venv aus dem Lock baut und darin die Testsammlung
startet. Der billige Test fängt den Regelfall (neue Abhängigkeit deklariert,
Lock vergessen); das Target fängt zusätzlich fehlende **transitive**
Abhängigkeiten — `lxml` fehlte aus genau diesem Grund.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Final

PROJEKTWURZEL: Final = Path(__file__).resolve().parent.parent
PYPROJECT: Final = PROJEKTWURZEL / "pyproject.toml"
LOCKDATEI: Final = PROJEKTWURZEL / "requirements.lock"

# PEP 508: Name, danach optional Extras, Versionsbedingung, Marker.
_NAME_MUSTER: Final = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalisiere(name: str) -> str:
    """Normalisiert einen Paketnamen nach PEP 503.

    Kleinschreibung, und jede Folge aus ``-``, ``_`` oder ``.`` wird zu einem
    einzelnen ``-``. Nötig, weil dieselbe Abhängigkeit als ``python-docx``,
    ``python_docx`` oder ``Python.Docx`` geschrieben sein kann.

    :param name: Paketname in beliebiger Schreibweise.
    :return: Normalisierter Name.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _deklarierte_pakete() -> set[str]:
    """Liest alle in ``pyproject.toml`` deklarierten Abhängigkeiten.

    Berücksichtigt sowohl die Laufzeitabhängigkeiten als auch **alle**
    optionalen Gruppen. Die Gruppe ``dev`` zählt mit, weil die Testsuite ohne
    sie nicht einmal eingesammelt werden kann — und die Reproduzierbarkeit der
    Testergebnisse ist der Kern von Prinzip P6.

    :return: Menge normalisierter Paketnamen.
    """
    inhalt = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    projekt = inhalt["project"]

    roh: list[str] = list(projekt.get("dependencies", []))
    for gruppe in projekt.get("optional-dependencies", {}).values():
        roh.extend(gruppe)

    pakete: set[str] = set()
    for eintrag in roh:
        treffer = _NAME_MUSTER.match(eintrag)
        if treffer:
            pakete.add(_normalisiere(treffer.group(1)))
    return pakete


def _gesperrte_pakete() -> dict[str, str]:
    """Liest ``requirements.lock`` als Abbildung Name auf Fassung.

    :return: Normalisierter Name auf Versionsangabe.
    """
    eintraege: dict[str, str] = {}
    for rohzeile in LOCKDATEI.read_text(encoding="utf-8").splitlines():
        zeile = rohzeile.strip()
        if not zeile or zeile.startswith("#"):
            continue
        name, _, fassung = zeile.partition("==")
        eintraege[_normalisiere(name)] = fassung
    return eintraege


def test_jede_deklarierte_abhaengigkeit_steht_im_lock() -> None:
    """Jede in ``pyproject.toml`` deklarierte Abhängigkeit muss im Lock stehen.

    Das ist die Richtung, die den Anlass gab: deklariert, aber nicht
    eingefroren. Die Gegenrichtung wird bewusst **nicht** geprüft — der Lock
    enthält naturgemäß transitive Abhängigkeiten, die nirgends deklariert sind.
    """
    fehlend = sorted(_deklarierte_pakete() - set(_gesperrte_pakete()))
    assert not fehlend, (
        "In pyproject.toml deklariert, in requirements.lock nicht eingefroren: "
        f"{', '.join(fehlend)}. "
        "Ein aus dem Lock allein gebautes venv kann das Projekt nicht ausführen. "
        "Behebung: 'make setup' in einer vollständigen Umgebung, oder die "
        "fehlenden Zeilen einzeln nachtragen."
    )


def test_lock_ist_vollstaendig_gepinnt() -> None:
    """Jede Zeile des Locks muss eine exakte Fassung nennen.

    Ein Lock mit ``>=`` statt ``==`` friert nichts ein und verfehlt seinen
    Zweck. Prinzip P6 verlangt, dass eine spätere Rechnung dieselben Zahlen
    erhält wie der Verfasser im Juli.
    """
    ohne_pin = sorted(name for name, fassung in _gesperrte_pakete().items() if not fassung)
    assert not ohne_pin, (
        f"Zeilen in requirements.lock ohne exakte Fassung (==): {', '.join(ohne_pin)}"
    )


def test_werkzeuge_der_testsuite_sind_eingefroren() -> None:
    """Die Pakete, deren Fehlen den Anlass gab, sind namentlich abgesichert.

    Redundant zum allgemeinen Test – mit Absicht. ``pypdf`` steht
    stellvertretend für die Klasse von Abhängigkeiten, die nur in Tests
    vorkommen und deshalb im Alltag nie auffallen: Der Golden-Master-Test liest
    damit den Text aus den erzeugten Dokumenten zurück. Fehlte das Paket in der
    Sperrdatei, liefe die Testsuite auf einer fremden Maschine nicht an.
    """
    gesperrt = _gesperrte_pakete()
    for paket in ("pypdf",):
        assert paket in gesperrt, (
            f"{paket} fehlt in requirements.lock – nur in Tests und Hilfsskripten "
            "verwendete Abhängigkeiten fallen im Alltag nicht auf."
        )
