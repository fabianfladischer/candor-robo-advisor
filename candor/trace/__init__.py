"""Lückenlose Protokollierung jedes Entscheidungsschritts.

Module:
    events.py    ``Rule`` und ``TraceEvent`` samt kanonischer Serialisierung
    recorder.py  ``TraceRecorder``: append-only, deterministisch, hashbar

Prinzip P2 (Erklärbarkeit by design): Es gibt keine Ausgabe des Systems,
die nicht auf eine benannte Regel zurückführbar ist. Kein ML im Entscheidungspfad.

Prinzip P3 (Trennung von Regel und Parameter): Die Fassung des Regelwerks wandert in
jeden Trace, damit ein Ergebnis auch nachträglich dem Regelstand zugeordnet werden
kann, unter dem es entstanden ist.

Der Trace ist zugleich der Nachweis für Invariante I4 (Determinismus): gleiches
Profil + gleicher Seed + gleiche Policy-Version ⇒ bitidentischer Trace.
"""

from __future__ import annotations
