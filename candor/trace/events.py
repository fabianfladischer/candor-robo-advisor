"""Ereignistyp des Decision-Trace und seine kanonische Serialisierung.

Ein Trace ist der Nachweis, dass eine Anlageentscheidung auf benannte Regeln
zurückgeht (Prinzip P2). Der Anspruch ist streng: **Ein Trace muss genügen, um
die Entscheidung ohne den Code nachzuvollziehen.** Daraus folgt, was ein Event
mitführen muss — nicht nur die Regel-Kennung, sondern ihre Beschreibung im
Klartext, sämtliche Eingabewerte, das Ergebnis, den Regelstand und, wo
einschlägig, die Rechtsgrundlage.

Zeitstempel und Determinismus
-----------------------------
Jedes Event trägt einen Zeitstempel. Zugleich verlangt Invariante I4, dass
gleiches Profil, gleicher Seed und gleicher Policy-Stand einen **bitidentischen**
Trace ergeben. Beides zusammen geht nicht: Zwei Läufe finden zu verschiedenen
Zeiten statt.

Aufgelöst wird der Widerspruch durch zwei Darstellungen desselben Trace:

* :meth:`TraceEvent.as_record` — die vollständige Fassung mit Zeitstempel. Sie
  geht in die Ablage und in die Geeignetheitserklärung.
* :meth:`TraceEvent.as_canonical` — die kanonische Fassung **ohne** Zeitstempel.
  Über sie wird der Inhaltshash gebildet, und auf sie bezieht sich Invariante I4.

Die Kennung ``event_id`` ist deshalb aus laufender Nummer und Regel-Kennung
abgeleitet und nicht zufällig: Eine UUID würde denselben Determinismus brechen,
den der ausgeklammerte Zeitstempel gerade rettet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Final

#: Wert, der sich verlustfrei nach JSON abbilden lässt.
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

#: Trennzeichen der kanonischen Serialisierung — ohne Leerraum, damit die
#: Byte-Darstellung eindeutig ist.
CANONICAL_SEPARATORS: Final = (",", ":")


class TraceError(RuntimeError):
    """Ein Trace-Ereignis ist unvollständig oder nicht serialisierbar."""


def encode_value(value: object) -> JsonValue:
    """Bildet einen Wert verlustfrei und deterministisch nach JSON ab.

    Behandelt die Typen, die im Domänenkern vorkommen und die ``json`` von sich
    aus nicht kennt. ``Decimal`` wird bewusst als **Zeichenkette** abgelegt und
    nicht als Gleitkommazahl: ``float(Decimal("0.1"))`` ist nicht ``0.1``, und
    ein Trace, dessen Beträge sich beim Serialisieren verschieben, wäre als
    Nachweis wertlos.

    Args:
        value: Abzubildender Wert.

    Returns:
        Die JSON-fähige Entsprechung.

    Raises:
        TraceError: Wenn der Typ nicht abbildbar ist. Bewusst ein harter Fehler:
            Ein stillschweigendes ``str(value)`` würde Objekte in unlesbare
            Zeichenketten verwandeln und den Nachweiszweck unterlaufen.
    """
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, date | datetime):
        # datetime ist Unterklasse von date; isoformat() ist für beide richtig.
        return value.isoformat()
    if isinstance(value, dict):
        return {str(schluessel): encode_value(wert) for schluessel, wert in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        eintraege = sorted(value, key=repr) if isinstance(value, set | frozenset) else value
        return [encode_value(eintrag) for eintrag in eintraege]
    raise TraceError(
        f"Wert vom Typ {type(value).__name__!r} lässt sich nicht verlustfrei nach JSON "
        "abbilden. Für den Trace ist eine ausdrückliche Abbildung zu ergänzen."
    )


def canonical_json(payload: object) -> str:
    """Serialisiert einen Wert deterministisch nach JSON.

    Schlüssel werden sortiert, Leerraum entfällt, Umlaute bleiben erhalten.
    Zweimal derselbe Inhalt ergibt damit zweimal dieselbe Bytefolge.

    Args:
        payload: Zu serialisierender Wert.

    Returns:
        Die kanonische JSON-Darstellung.
    """
    return json.dumps(
        encode_value(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=CANONICAL_SEPARATORS,
    )


@dataclass(frozen=True, slots=True)
class Rule:
    """Eine benannte Entscheidungsregel.

    Kennung, Beschreibung und Rechtsgrundlage gehören zusammen und werden
    deshalb einmal deklariert statt an jeder Aufrufstelle erneut angegeben.
    Sonst könnte dieselbe Regel-Kennung an zwei Stellen mit unterschiedlicher
    Beschreibung aufgezeichnet werden — der Trace wäre dann in sich
    widersprüchlich, und genau das soll er nie sein.

    Attributes:
        rule_id: Kennung der Regel, etwa ``CAP-LIQ``.
        description: Die Regel im Klartext, lesbar ohne den Code.
        legal_basis: Rechtsgrundlage, sofern einschlägig.
    """

    rule_id: str
    description: str
    legal_basis: str | None = None

    def __post_init__(self) -> None:
        """Prüft, dass Kennung und Beschreibung besetzt sind.

        Raises:
            TraceError: Bei leerer Kennung oder Beschreibung.
        """
        if not self.rule_id.strip():
            raise TraceError("Eine Regel ohne Kennung ist nicht nachvollziehbar.")
        if not self.description.strip():
            raise TraceError(
                f"Regel {self.rule_id!r}: Ohne Beschreibung im Klartext lässt sich der "
                "Trace ohne den Code nicht lesen."
            )


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Ein einzelner Entscheidungsschritt.

    Attributes:
        event_id: Kennung der Form ``0007-CAP-LIQ``, abgeleitet aus laufender
            Nummer und Regel-Kennung. Deterministisch, damit Invariante I4 hält.
        sequence: Laufende Nummer innerhalb des Trace, beginnend bei eins.
        timestamp: Zeitpunkt der Aufzeichnung, zeitzonenbehaftet in UTC. Geht
            **nicht** in den Inhaltshash ein.
        rule_id: Kennung der angewandten Regel, etwa ``CAP-LIQ``.
        rule_description: Die Regel im Klartext. Ohne sie wäre der Trace nur
            ohne den Code nicht lesbar — genau das soll er aber sein.
        inputs: Sämtliche Eingabewerte der Regel.
        output: Ergebnis der Regel.
        policy_version: Fassung des Regelwerks, unter der die Regel angewandt
            wurde (Prinzip P3).
        legal_basis: Rechtsgrundlage, sofern einschlägig.
    """

    event_id: str
    sequence: int
    timestamp: datetime
    rule_id: str
    rule_description: str
    inputs: dict[str, JsonValue]
    output: JsonValue
    policy_version: str
    legal_basis: str | None = None

    def __post_init__(self) -> None:
        """Prüft die Pflichtangaben des Ereignisses.

        Raises:
            TraceError: Bei leerer Regel-Kennung, fehlender Beschreibung,
                unpassender Kennung, nicht positiver laufender Nummer oder
                zeitzonenlosem Zeitstempel.
        """
        if self.sequence < 1:
            raise TraceError(f"Die laufende Nummer muss positiv sein, ist aber {self.sequence}.")
        if not self.rule_id.strip():
            raise TraceError("Ein Trace-Ereignis ohne Regel-Kennung ist nicht nachvollziehbar.")
        if not self.rule_description.strip():
            raise TraceError(
                f"Regel {self.rule_id!r}: Ohne Beschreibung im Klartext lässt sich der Trace "
                "ohne den Code nicht lesen."
            )
        if not self.policy_version.strip():
            raise TraceError(f"Regel {self.rule_id!r}: Der Regelstand fehlt (Prinzip P3).")
        erwartet = build_event_id(self.sequence, self.rule_id)
        if self.event_id != erwartet:
            raise TraceError(
                f"Die Kennung {self.event_id!r} passt nicht zu Nummer und Regel "
                f"(erwartet {erwartet!r})."
            )
        if self.timestamp.tzinfo is None:
            raise TraceError(
                f"Regel {self.rule_id!r}: Der Zeitstempel ist zeitzonenlos. Ein "
                "Beratungsprotokoll ohne eindeutigen Zeitbezug ist nicht verwertbar."
            )

    def as_canonical(self) -> dict[str, JsonValue]:
        """Liefert die kanonische Fassung ohne Zeitstempel.

        Returns:
            Die inhaltliche Darstellung des Ereignisses, Grundlage von
            Invariante I4.
        """
        eintrag: dict[str, JsonValue] = {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "rule_id": self.rule_id,
            "rule_description": self.rule_description,
            "inputs": encode_value(self.inputs),
            "output": encode_value(self.output),
            "policy_version": self.policy_version,
        }
        if self.legal_basis is not None:
            eintrag["legal_basis"] = self.legal_basis
        return eintrag

    def as_record(self) -> dict[str, JsonValue]:
        """Liefert die vollständige Fassung einschließlich Zeitstempel.

        Returns:
            Die Darstellung für Ablage und Geeignetheitserklärung.
        """
        eintrag = self.as_canonical()
        eintrag["timestamp"] = self.timestamp.isoformat()
        return eintrag


def build_event_id(sequence: int, rule_id: str) -> str:
    """Bildet die deterministische Ereigniskennung.

    Args:
        sequence: Laufende Nummer, beginnend bei eins.
        rule_id: Kennung der Regel.

    Returns:
        Kennung der Form ``0007-CAP-LIQ``.
    """
    return f"{sequence:04d}-{rule_id}"


def as_input_mapping(values: dict[str, Any]) -> dict[str, JsonValue]:
    """Wandelt beliebige Eingabewerte in eine JSON-fähige Zuordnung.

    Args:
        values: Eingabewerte der Regel.

    Returns:
        Die abgebildete Zuordnung.

    Raises:
        TraceError: Wenn ein Wert nicht abbildbar ist.
    """
    return {str(schluessel): encode_value(wert) for schluessel, wert in values.items()}
