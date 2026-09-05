"""Aufzeichnung eines vollständigen Entscheidungspfads.

Der Recorder ist **append-only**: Er kennt keine Methode zum Ändern oder
Entfernen eines bereits aufgezeichneten Ereignisses. Das ist kein technischer
Selbstzweck. Ein Beratungsprotokoll, aus dem sich nachträglich einzelne Schritte
entfernen lassen, ist als Nachweis wertlos — und die bloße Möglichkeit dazu
genügt, um den Nachweiswert zu entwerten.

Der aufgezeichnete Trace besteht aus zwei Teilen:

* dem **Kopf** (:class:`TraceMetadata`) mit Kunde, Ziel, Regelstand, Seed und
  Datenstand — alles, was das Ergebnis bestimmt, aber keine einzelne Regel ist,
* der **Ereignisfolge** in Aufzeichnungsreihenfolge.

Der Inhaltshash (:meth:`TraceRecorder.content_hash`) deckt beide Teile ab,
klammert aber Zeitstempel aus. Er ist die prüfbare Fassung von Invariante I4:
Gleiches Profil, gleicher Seed und gleicher Policy-Stand ergeben denselben Hash.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

from candor.trace.events import (
    JsonValue,
    Rule,
    TraceError,
    TraceEvent,
    as_input_mapping,
    build_event_id,
    canonical_json,
    encode_value,
)

#: Fassung des Trace-Formats. Ändert sich der Aufbau, steigt diese Zahl.
TRACE_SCHEMA_VERSION: Final = 1


def utc_now() -> dt.datetime:
    """Liefert den aktuellen Zeitpunkt in UTC.

    Als eigene Funktion geführt, damit Tests eine feste Uhr einsetzen können,
    ohne die Systemzeit zu verändern.

    Returns:
        Der aktuelle zeitzonenbehaftete Zeitpunkt.
    """
    return dt.datetime.now(tz=dt.UTC)


@dataclass(frozen=True, slots=True)
class TraceMetadata:
    """Kopfdaten eines Trace.

    Attributes:
        client_id: Kennung des Kunden.
        goal_id: Kennung des Anlageziels.
        policy_versions: Fassungen aller Teilpolicies (Prinzip P3).
        package_version: Version des Pakets ``candor``.
        seed: Verwendeter Seed, sofern eine Zufallsquelle beteiligt ist (Prinzip P1).
        snapshot_hash: Kurzhash des Marktdaten-Snapshots, sofern verwendet (Prinzip P6).
        context: Weitere Angaben, die das Ergebnis bestimmen.
    """

    client_id: str
    goal_id: str
    policy_versions: dict[str, str]
    package_version: str
    seed: int | None = None
    snapshot_hash: str | None = None
    context: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Prüft die Pflichtangaben des Kopfes.

        Raises:
            TraceError: Wenn Kunde, Ziel oder Regelstand fehlen.
        """
        if not self.client_id.strip():
            raise TraceError("Ein Trace ohne Kundenkennung ist nicht zuordenbar.")
        if not self.goal_id.strip():
            raise TraceError("Ein Trace ohne Zielkennung ist nicht zuordenbar.")
        if not self.policy_versions:
            raise TraceError(
                "Ein Trace ohne Regelstand lässt sich später keinem Regelwerk zuordnen "
                "(Prinzip P3)."
            )

    def as_canonical(self) -> dict[str, JsonValue]:
        """Liefert die kanonische Fassung des Kopfes.

        Returns:
            Die inhaltliche Darstellung ohne zeitabhängige Angaben.
        """
        kopf: dict[str, JsonValue] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "client_id": self.client_id,
            "goal_id": self.goal_id,
            "policy_versions": encode_value(dict(sorted(self.policy_versions.items()))),
            "package_version": self.package_version,
        }
        if self.seed is not None:
            kopf["seed"] = self.seed
        if self.snapshot_hash is not None:
            kopf["snapshot_hash"] = self.snapshot_hash
        if self.context:
            kopf["context"] = encode_value(dict(sorted(self.context.items())))
        return kopf


class TraceRecorder:
    """Sammelt Entscheidungsschritte in Aufzeichnungsreihenfolge.

    Der Recorder vergibt laufende Nummern und Kennungen selbst. Aufrufer können
    weder eine Nummer setzen noch ein Ereignis nachträglich verändern.
    """

    def __init__(
        self,
        metadata: TraceMetadata,
        clock: Callable[[], dt.datetime] | None = None,
    ) -> None:
        """Legt einen leeren Recorder an.

        Args:
            metadata: Kopfdaten des Trace.
            clock: Zeitquelle. Vorgabe ist :func:`utc_now`. Tests setzen hier
                eine feste Uhr ein.
        """
        self._metadata = metadata
        self._clock = clock if clock is not None else utc_now
        self._events: list[TraceEvent] = []

    @property
    def metadata(self) -> TraceMetadata:
        """Kopfdaten des Trace."""
        return self._metadata

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """Aufgezeichnete Ereignisse in Aufzeichnungsreihenfolge.

        Returns:
            Ein unveränderliches Tupel. Änderungen daran wirken nicht auf den
            Recorder zurück.
        """
        return tuple(self._events)

    def __len__(self) -> int:
        """Zahl der aufgezeichneten Ereignisse."""
        return len(self._events)

    def record(
        self,
        rule: Rule,
        inputs: dict[str, Any],
        output: object,
        policy_version: str,
    ) -> TraceEvent:
        """Zeichnet einen Entscheidungsschritt auf.

        Args:
            rule: Angewandte Regel mit Kennung, Beschreibung und Rechtsgrundlage.
            inputs: Sämtliche Eingabewerte der Regel.
            output: Ergebnis der Regel.
            policy_version: Fassung des Regelwerks, unter der die Regel wirkte.

        Returns:
            Das aufgezeichnete Ereignis.

        Raises:
            TraceError: Wenn Pflichtangaben fehlen oder ein Wert sich nicht
                verlustfrei nach JSON abbilden lässt. Der Fehler tritt beim
                Aufzeichnen auf und nicht erst beim Speichern — sonst fiele er
                erst auf, wenn die Entscheidung längst getroffen ist.
        """
        sequence = len(self._events) + 1
        ereignis = TraceEvent(
            event_id=build_event_id(sequence, rule.rule_id),
            sequence=sequence,
            timestamp=self._clock(),
            rule_id=rule.rule_id,
            rule_description=rule.description,
            inputs=as_input_mapping(inputs),
            output=encode_value(output),
            policy_version=policy_version,
            legal_basis=rule.legal_basis,
        )
        self._events.append(ereignis)
        return ereignis

    def as_canonical(self) -> dict[str, JsonValue]:
        """Liefert den vollständigen Trace ohne Zeitstempel.

        Returns:
            Kopf und Ereignisfolge in kanonischer Form.
        """
        return {
            "metadata": self._metadata.as_canonical(),
            "events": [ereignis.as_canonical() for ereignis in self._events],
        }

    def as_record(self) -> dict[str, JsonValue]:
        """Liefert den vollständigen Trace einschließlich Zeitstempel.

        Returns:
            Kopf, Inhaltshash und Ereignisfolge.
        """
        return {
            "metadata": self._metadata.as_canonical(),
            "content_hash": self.content_hash(),
            "events": [ereignis.as_record() for ereignis in self._events],
        }

    def canonical_json(self) -> str:
        """Serialisiert die kanonische Fassung deterministisch.

        Returns:
            Die kanonische JSON-Darstellung.
        """
        return canonical_json(self.as_canonical())

    def to_json(self, indent: int | None = 2) -> str:
        """Serialisiert den vollständigen Trace zur Ablage.

        Args:
            indent: Einrückung. ``None`` erzeugt eine einzeilige Darstellung.

        Returns:
            Die JSON-Darstellung mit Zeitstempeln und Inhaltshash.
        """
        return json.dumps(self.as_record(), sort_keys=True, ensure_ascii=False, indent=indent)

    def content_hash(self) -> str:
        """Bildet den Inhaltshash über die kanonische Fassung.

        Der Hash ist die prüfbare Fassung von Invariante I4. Er umfasst Kopf und
        alle Ereignisse, klammert aber Zeitstempel aus.

        Returns:
            Hexadezimaler SHA-256-Hash.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
