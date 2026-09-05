"""Marktdatenbeschaffung, eingefrorene Snapshots und Hash-Manifest.

Module:
    universe_meta.py  Anlageuniversum: acht Assetklassen mit ISIN, Symbol und
                      dokumentiertem Verifikationsstand
    fetch.py          Abruf der Monatsrenditen; Quelle hinter dem Protocol
                      ``PriceSource`` gekapselt und damit austauschbar
    snapshot.py       Einfrieren der Zeitreihen als Parquet nach ``data/snapshot/``
    manifest.py       SHA-256-Manifest samt ``verify_snapshot()``
    loader.py         Zugriffspunkt des Systems; liest standardmäßig den Snapshot
    stats.py          Deskriptive Kennzahlen und Korrelationsmatrix

Prinzip P6 (Reproduzierbarkeit vor Aktualität): Wer die Ergebnisse Monate später
nachrechnet, muss dieselben Zahlen erhalten. Der Snapshot ist
deshalb der Primärpfad; ein Live-Abruf erfordert das ausdrückliche Flag
``live=True`` und überschreibt den Snapshot nie stillschweigend.

Prinzip P5 (Fail loud): Ein fehlgeschlagener Abruf oder ein fehlender
Snapshot führt zu einer Exception. Synthetische Ersatzdaten entstehen an keiner
Stelle dieses Pakets.

Dieses Paket ist die einzige Stelle im Projekt, an der Netzwerkzugriffe zulässig
sind. ``candor.domain`` und ``candor.profiling`` bleiben davon frei (Prinzip P4).

Risikobezug: Die Marktdatenquelle kann wegbrechen oder ihr Schema ändern.
"""

from __future__ import annotations
