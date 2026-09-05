"""Domänenkern: reine Datenmodelle ohne jede Ein-/Ausgabe.

Module:
    errors.py   Fachliche Fehlertypen mit Feldpfad und Rechtsgrundlage
    client.py   Haushalt, Kenntnisse und Erfahrungen, Nachhaltigkeitspräferenzen
    goal.py     Anlageziel mit Priorität MUSS/WUNSCH
    policy.py   Typen des versionierten Regelwerks aus ``policies/*.yaml``
    result.py   Ergebnisobjekte: Risikoprofil, Empfehlung, Konfliktbericht

Prinzip P4 (Domänenkern ohne Ein- und Ausgabe): Module dieses Pakets importieren weder
Netzwerk noch Dateisystem noch Streamlit. Das Einlesen der YAML-Policies erfolgt in
``candor.data``; hierher gelangen ausschließlich bereits geparste Objekte.

Prinzip P5 (Fail loud): Unvollständige Pflichtangaben führen zu einer
Exception, niemals zu stillen Defaults.
"""

from __future__ import annotations
