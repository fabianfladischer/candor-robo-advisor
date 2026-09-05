"""Strukturierte Ermittlung des Risikoprofils über drei getrennte Achsen.

Module:
    scoring.py    Gemeinsame Bausteine der punktebasierten Bewertung
    capacity.py   Risikotragfähigkeit (objektiv) über sechs Kriterien, Struktur
                  angelehnt an DIN 77230; Liquiditätsreserve zusätzlich als
                  K.-o.-Kriterium
    tolerance.py  Risikobereitschaft (subjektiv): psychometrische Items und
                  Verlustszenarien in Euro; Widersprüche werden ausgewiesen,
                  nicht weggerechnet
    knowledge.py  Kenntnisse und Erfahrungen (Art. 55 DelVO (EU) 2017/565) als
                  Deckel auf die Produktkomplexität — kein Score
    classify.py   Bindendes Minimum aller drei Achsen samt Divergenzbefund

Kernentscheidung: **Nie mitteln.** Die Score-Mittelung ist genau der Fehler, den
dieses Projekt vermeidet — sie lässt den Konflikt zwischen den Achsen verschwinden.

Prinzip P4 (Domänenkern ohne I/O) gilt auch für dieses Paket.

Invarianten I1, I2, I3.
"""

from __future__ import annotations
