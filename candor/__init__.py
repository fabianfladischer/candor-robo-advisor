"""CANDOR — Robo-Advisor mit ehrlicher Konfliktdiagnose.

Kernidee: Risikotragfähigkeit, Risikobereitschaft und Kenntnisse/Erfahrungen werden
getrennt geführt und über ein **bindendes Minimum** verknüpft, nicht gemittelt.
Reicht die zulässige Risikoklasse nicht für die geforderte Zielerreichung, wird keine
Empfehlung ausgesprochen, sondern ein Konfliktbericht erzeugt
(Art. 54 Abs. 10 DelVO (EU) 2017/565).

Paketübersicht:
    domain      Reine Datenmodelle (Kunde, Ziel, Policy, Ergebnis) — ohne I/O.
    profiling   Ermittlung des Risikoprofils über drei Achsen.
    goals       Modellierung des Anlageziels und der erforderlichen Rendite.
    engine      Simulation, Zielabgleich, Konfliktlösung, Glide-Path.
    portfolio   Anlageuniversum, Kovarianzschätzung, Optimierung, 1/N-Benchmark.
    trace       Lückenlose Protokollierung jedes Entscheidungsschritts.
    report      Geeignetheitserklärung und Konfliktbericht.
    data        Marktdatenbeschaffung, eingefrorene Snapshots, Hash-Manifest.

Entwurfsprinzipien: Prinzip P1 (Determinismus), Prinzip P2 (Erklärbarkeit by design),
Prinzip P3 (Trennung von Regel und Parameter), Prinzip P4 (Domänenkern ohne Ein- und
Ausgabe), Prinzip P5 (Fail loud – keine stillen Vorbelegungen), Prinzip P6
(Reproduzierbarkeit vor Aktualität).
"""

from __future__ import annotations

#: Versionsstand des Pakets. Wandert in jeden Trace und in jede Geeignetheitserklärung.
__version__ = "0.1.0"

__all__ = ["__version__"]
