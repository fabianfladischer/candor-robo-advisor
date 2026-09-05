"""Entscheidungskern: Simulation, Zielabgleich, Konfliktlösung, Glide-Path.

Module:
    simulate.py     Zirkulärer Block-Bootstrap mit gemeinsamem Zeitindex über
                    alle Assetklassen; dazu eine i.i.d.-Normalziehung als
                    Vergleichsmaß
    convergence.py  Konvergenzanalyse über die Pfadzahl und Vergleich beider
                    Ziehungsverfahren
    glidepath.py    Regelbasierter Abbau der zulässigen Risikoklasse mit
                    abnehmender Restlaufzeit zum Zieltermin
    reconcile.py    Abgleich von erforderlicher Rendite und zulässiger
                    Risikoklasse
    solve.py        Drei per Bisektion gelöste Stellschrauben im Konfliktfall

Prinzip P1 (Determinismus): jede Zufallsquelle erhält einen explizit
übergebenen Seed über ``numpy.random.default_rng(seed)``. Kein globales
``numpy.random.seed()``.

Invarianten I4, I7.
"""

from __future__ import annotations
