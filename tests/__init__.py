"""Testsuite des Projekts CANDOR.

Aufbau:
    test_smoke.py       Gerüstnachweis: Paket importierbar, Struktur vollständig
    test_invariants.py  Die neun Invarianten als Property-based Tests (hypothesis)
    test_golden.py      Golden-Master-Tests der erzeugten Dokumente

Die Invariantentests sind der eigentliche Qualitätsnachweis:
Sie prüfen nicht Beispiele, sondern Eigenschaften über den gesamten Eingaberaum.
"""

from __future__ import annotations
