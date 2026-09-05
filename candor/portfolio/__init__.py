"""Portfoliokonstruktion: Universum, Schätzung, Optimierung, Benchmark.

Module:
    universe.py   Anlageuniversum aus dem eingefrorenen Datensnapshot;
                  Nachhaltigkeitsfilter mit definierter Mindestgröße
    estimate.py   Erwartungswerte und Kovarianz per Ledoit-Wolf-Shrinkage
                  (``sklearn.covariance.LedoitWolf`` — einzige zulässige
                  scikit-learn-Verwendung im Projekt)
    optimize.py   Restringierte Mean-Variance-Optimierung mit cvxpy.
                  Pflicht-Restriktionen: keine Leerverkäufe, Positionsobergrenze,
                  Assetklassen-Bänder je Risikoklasse
    benchmark.py  Gleichgewichtetes 1/N-Portfolio als Pflicht-Benchmark
                  (DeMiguel et al., 2009)

Kritische Einordnung: Schlägt die MVO das 1/N-Portfolio out-of-sample nicht, wird
das ausgewiesen, nicht versteckt. Michaud (1989) zur Fehlerverstärkung
der MVO ist dabei die theoretische Erklärung.

Invarianten I5, I6, I8.
"""

from __future__ import annotations
