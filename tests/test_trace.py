"""Tests des Decision-Trace.

Der zentrale Nachweis ist die **deterministische Serialisierung**: Zwei Läufe
mit gleichem Inhalt, aber unterschiedlicher Uhrzeit müssen denselben
Inhaltshash ergeben (Invariante I4). Ebenso wichtig ist die Gegenprobe — ändert
sich ein Eingabewert, muss sich der Hash ändern, sonst wäre er wertlos.

Zusätzlich wird geprüft, was der Anspruch „ein Trace muss ohne den Code genügen"
konkret bedeutet: Jedes Ereignis führt Regelbeschreibung, sämtliche Eingaben,
das Ergebnis und den Regelstand mit.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from decimal import Decimal

import pytest

from candor.domain.client import EmploymentType
from candor.trace.events import Rule, TraceError, TraceEvent, canonical_json, encode_value
from candor.trace.recorder import TraceMetadata, TraceRecorder

REGEL_LIQ = Rule(
    rule_id="CAP-LIQ",
    description="Liquiditätsreserve in Monatsausgaben wird auf Punkte abgebildet.",
    legal_basis="Art. 54 Abs. 4 DelVO (EU) 2017/565",
)
REGEL_MIN = Rule(
    rule_id="CLS-MIN",
    description="Die Risikoklasse ist das Minimum aus Tragfähigkeit und Bereitschaft.",
)


def feste_uhr(zeitpunkt: dt.datetime) -> Callable[[], dt.datetime]:
    """Erzeugt eine Zeitquelle, die stets denselben Zeitpunkt liefert.

    Args:
        zeitpunkt: Der zurückzugebende Zeitpunkt.

    Returns:
        Die Zeitquelle.
    """
    return lambda: zeitpunkt


def kopfdaten(**abweichungen: object) -> TraceMetadata:
    """Baut Kopfdaten mit optionalen Abweichungen.

    Args:
        **abweichungen: Felder, die vom Grundfall abweichen sollen.

    Returns:
        Die erzeugten Kopfdaten.
    """
    felder: dict[str, object] = {
        "client_id": "K-0001",
        "goal_id": "Z-0001",
        "policy_versions": {"capacity": "1.0.0", "classes": "1.0.0"},
        "package_version": "0.1.0",
        "seed": 20260726,
        "snapshot_hash": "365e8354e9b6",
    }
    felder.update(abweichungen)
    return TraceMetadata(**felder)  # type: ignore[arg-type]


def befuellter_recorder(clock: Callable[[], dt.datetime] | None = None) -> TraceRecorder:
    """Baut einen Recorder mit zwei aufgezeichneten Schritten.

    Args:
        clock: Optionale Zeitquelle.

    Returns:
        Der befüllte Recorder.
    """
    recorder = TraceRecorder(kopfdaten(), clock=clock)
    recorder.record(
        REGEL_LIQ,
        {"liquiditaetsreserve_monate": Decimal("4.0"), "band": "bis 6,0"},
        70,
        "1.0.0",
    )
    recorder.record(
        REGEL_MIN,
        {"tragfaehigkeit": 4, "bereitschaft": 3, "kenntnisdeckel": 4},
        3,
        "1.0.0",
    )
    return recorder


# --- Aufzeichnung ----------------------------------------------------------


def test_kennungen_sind_fortlaufend_und_deterministisch() -> None:
    """Ereigniskennungen leiten sich aus Nummer und Regel ab, nicht aus Zufall."""
    recorder = befuellter_recorder()

    assert [ereignis.event_id for ereignis in recorder.events] == [
        "0001-CAP-LIQ",
        "0002-CLS-MIN",
    ]
    assert [ereignis.sequence for ereignis in recorder.events] == [1, 2]


def test_recorder_ist_append_only() -> None:
    """Die Ereignisliste lässt sich von außen nicht verändern."""
    recorder = befuellter_recorder()
    ereignisse = recorder.events

    assert isinstance(ereignisse, tuple)
    assert not hasattr(recorder, "remove")
    assert not hasattr(recorder, "clear")
    assert len(recorder) == 2


def test_ereignis_ist_unveraenderlich() -> None:
    """Ein aufgezeichnetes Ereignis lässt sich nicht nachträglich ändern."""
    ereignis = befuellter_recorder().events[0]

    with pytest.raises(AttributeError):
        ereignis.output = 99  # type: ignore[misc]


def test_rechtsgrundlage_wandert_ins_ereignis() -> None:
    """Die Rechtsgrundlage der Regel steht im aufgezeichneten Ereignis."""
    ereignisse = befuellter_recorder().events

    assert ereignisse[0].legal_basis == "Art. 54 Abs. 4 DelVO (EU) 2017/565"
    assert ereignisse[1].legal_basis is None


def test_trace_genuegt_ohne_den_code() -> None:
    """Jedes Ereignis trägt alles, was zum Nachvollziehen nötig ist."""
    abgelegt = json.loads(befuellter_recorder().to_json())

    for ereignis in abgelegt["events"]:
        assert ereignis["rule_id"]
        assert len(ereignis["rule_description"]) > 20
        assert ereignis["inputs"]
        assert "output" in ereignis
        assert ereignis["policy_version"]
        assert ereignis["timestamp"]
    assert abgelegt["metadata"]["policy_versions"]
    assert abgelegt["metadata"]["seed"] == 20260726


# --- Determinismus ---------------------------------------------------------


def test_inhaltshash_ist_unabhaengig_von_der_uhrzeit() -> None:
    """Zwei Läufe zu verschiedenen Zeiten ergeben denselben Inhaltshash.

    Das ist die prüfbare Fassung von Invariante I4.
    """
    frueh = befuellter_recorder(feste_uhr(dt.datetime(2026, 7, 26, 8, 0, tzinfo=dt.UTC)))
    spaet = befuellter_recorder(feste_uhr(dt.datetime(2026, 12, 24, 23, 59, tzinfo=dt.UTC)))

    assert frueh.content_hash() == spaet.content_hash()
    assert frueh.canonical_json() == spaet.canonical_json()


def test_inhaltshash_aendert_sich_bei_anderer_eingabe() -> None:
    """Die Gegenprobe: Ein anderer Eingabewert ergibt einen anderen Hash."""
    grundfall = befuellter_recorder()
    abweichend = TraceRecorder(kopfdaten())
    abweichend.record(REGEL_LIQ, {"liquiditaetsreserve_monate": Decimal("9.0")}, 90, "1.0.0")

    assert grundfall.content_hash() != abweichend.content_hash()


def test_inhaltshash_aendert_sich_bei_anderem_regelstand() -> None:
    """Ein anderer Policy-Stand ergibt einen anderen Hash (Prinzip P3)."""
    grundfall = befuellter_recorder()
    andere_version = TraceRecorder(kopfdaten(policy_versions={"capacity": "2.0.0"}))
    andere_version.record(REGEL_LIQ, {"liquiditaetsreserve_monate": Decimal("4.0")}, 70, "2.0.0")

    assert grundfall.content_hash() != andere_version.content_hash()


def test_kanonische_fassung_enthaelt_keine_zeitstempel() -> None:
    """Die kanonische Fassung ist frei von Zeitangaben."""
    recorder = befuellter_recorder(feste_uhr(dt.datetime(2026, 7, 26, 8, 0, tzinfo=dt.UTC)))

    assert "timestamp" not in recorder.canonical_json()
    assert "2026-07-26T08:00" in recorder.to_json()


def test_serialisierung_ist_wiederholbar() -> None:
    """Zweimaliges Serialisieren desselben Trace liefert dieselbe Bytefolge."""
    recorder = befuellter_recorder(feste_uhr(dt.datetime(2026, 7, 26, 8, 0, tzinfo=dt.UTC)))

    assert recorder.to_json() == recorder.to_json()
    assert recorder.canonical_json().encode("utf-8") == recorder.canonical_json().encode("utf-8")


def test_schluessel_sind_sortiert() -> None:
    """Die kanonische Fassung sortiert Schlüssel, unabhängig von der Eingabereihenfolge."""
    erster = TraceRecorder(kopfdaten())
    erster.record(REGEL_MIN, {"b": 2, "a": 1}, 3, "1.0.0")
    zweiter = TraceRecorder(kopfdaten())
    zweiter.record(REGEL_MIN, {"a": 1, "b": 2}, 3, "1.0.0")

    assert erster.canonical_json() == zweiter.canonical_json()


def test_abgelegter_trace_ist_gueltiges_json() -> None:
    """Der abgelegte Trace lässt sich vollständig zurücklesen."""
    recorder = befuellter_recorder()

    zurueckgelesen = json.loads(recorder.to_json())

    assert zurueckgelesen["content_hash"] == recorder.content_hash()
    assert len(zurueckgelesen["events"]) == 2


# --- Wertabbildung ---------------------------------------------------------


def test_decimal_wird_als_zeichenkette_abgelegt() -> None:
    """Beträge werden verlustfrei als Zeichenkette abgelegt, nicht als Gleitkommazahl."""
    assert encode_value(Decimal("0.1")) == "0.1"
    assert encode_value(Decimal("1234.56")) == "1234.56"
    assert json.loads(canonical_json({"betrag": Decimal("0.1")}))["betrag"] == "0.1"


def test_aufzaehlung_wird_ueber_den_namen_abgelegt() -> None:
    """Aufzählungen erscheinen mit ihrem Namen, nicht mit ihrem Anzeigetext."""
    assert encode_value(EmploymentType.VERBEAMTET) == "VERBEAMTET"


def test_datum_wird_nach_iso_abgebildet() -> None:
    """Datums- und Zeitangaben werden nach ISO 8601 abgebildet."""
    assert encode_value(dt.date(2026, 7, 26)) == "2026-07-26"
    assert encode_value(dt.datetime(2026, 7, 26, 8, 0, tzinfo=dt.UTC)).startswith(
        "2026-07-26T08:00"
    )


def test_nicht_abbildbarer_wert_wird_hart_abgewiesen() -> None:
    """Ein nicht abbildbarer Wert führt zum Fehler statt zu ``str(objekt)``."""

    class Beliebig:
        pass

    with pytest.raises(TraceError, match="nicht verlustfrei"):
        encode_value(Beliebig())


def test_nicht_abbildbare_eingabe_faellt_beim_aufzeichnen_auf() -> None:
    """Der Fehler tritt beim Aufzeichnen auf, nicht erst beim Speichern."""
    recorder = TraceRecorder(kopfdaten())

    with pytest.raises(TraceError):
        recorder.record(REGEL_MIN, {"objekt": object()}, 1, "1.0.0")


# --- Pflichtangaben --------------------------------------------------------


def test_regel_ohne_beschreibung_wird_abgewiesen() -> None:
    """Eine Regel ohne Klartextbeschreibung ist unzulässig."""
    with pytest.raises(TraceError, match="Beschreibung"):
        Rule(rule_id="X-1", description="   ")


def test_regel_ohne_kennung_wird_abgewiesen() -> None:
    """Eine Regel ohne Kennung ist unzulässig."""
    with pytest.raises(TraceError, match="ohne Kennung"):
        Rule(rule_id="", description="Irgendeine Regel.")


def test_kopf_ohne_regelstand_wird_abgewiesen() -> None:
    """Ein Trace ohne Policy-Versionen ist später nicht zuordenbar."""
    with pytest.raises(TraceError, match="ohne Regelstand"):
        kopfdaten(policy_versions={})


def test_kopf_ohne_kundenkennung_wird_abgewiesen() -> None:
    """Ein Trace ohne Kundenkennung ist nicht zuordenbar."""
    with pytest.raises(TraceError, match="Kundenkennung"):
        kopfdaten(client_id="  ")


def test_zeitzonenloser_zeitstempel_wird_abgewiesen() -> None:
    """Ein Ereignis ohne Zeitzonenbezug ist unzulässig."""
    with pytest.raises(TraceError, match="zeitzonenlos"):
        TraceEvent(
            event_id="0001-X",
            sequence=1,
            timestamp=dt.datetime(2026, 7, 26, 8, 0),  # noqa: DTZ001
            rule_id="X",
            rule_description="Eine Regel.",
            inputs={},
            output=None,
            policy_version="1.0.0",
        )


def test_unpassende_kennung_wird_abgewiesen() -> None:
    """Eine von Hand gesetzte, unpassende Ereigniskennung wird erkannt."""
    with pytest.raises(TraceError, match="passt nicht"):
        TraceEvent(
            event_id="9999-FALSCH",
            sequence=1,
            timestamp=dt.datetime(2026, 7, 26, 8, 0, tzinfo=dt.UTC),
            rule_id="X",
            rule_description="Eine Regel.",
            inputs={},
            output=None,
            policy_version="1.0.0",
        )
