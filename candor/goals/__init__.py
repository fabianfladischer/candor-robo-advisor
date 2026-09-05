"""Strukturierte Ermittlung und Modellierung des Anlageziels.

Module:
    model.py            Zielkatalog mit Vorbelegungen je Zieltyp sowie Zuteilung
                        der Sparrate auf mehrere Ziele
    required_return.py  Erforderliche Rendite über die nachschüssige
                        Rentenendwertformel; geschlossen gelöst wo möglich,
                        sonst über das Brent-Verfahren. Nominale und reale
                        Größe werden getrennt ausgewiesen.

Die erforderliche Rendite ist die dritte Größe neben Tragfähigkeit und Bereitschaft.
Erst ihr Abgleich mit der zulässigen Risikoklasse macht den Konflikt sichtbar
(Verarbeitung in ``candor.engine.reconcile``).
"""

from __future__ import annotations
