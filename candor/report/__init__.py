"""Erzeugung der regulatorisch geforderten Dokumente.

Module:
    suitability.py  Geeignetheitserklärung als PDF (reportlab).
                    Rechtsgrundlage: Art. 54 Abs. 12 DelVO (EU) 2017/565,
                    § 64 Abs. 4 WpHG
    conflict.py     Konfliktbericht mit den drei quantifizierten Stellschrauben
                    als Ersatz für eine unzulässige Empfehlung.
                    Rechtsgrundlage: Art. 54 Abs. 10 DelVO (EU) 2017/565
    templates/      Textbausteine und Layoutvorlagen der Dokumente

Beide Dokumente werden über Golden-Master-Tests abgesichert: Bei unverändertem
Profil, Seed und Policy-Stand muss das Ergebnis bitidentisch bleiben.

Alle Zahlausgaben verwenden das deutsche Dezimalkomma.

Damit ist die Anlageentscheidung dokumentiert und nachvollziehbar (Invariante I9).
"""

from __future__ import annotations
