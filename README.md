# CANDOR

Demonstrator eines regelbasierten Robo-Advisors mit zweiachsiger
Geeignetheitsprüfung. Das System ermittelt die zulässige Risikoklasse aus
Risikotragfähigkeit, Risikobereitschaft sowie Kenntnissen und Erfahrungen,
gleicht sie gegen das Anlageziel ab und verweigert die Empfehlung, wenn Ziel und
zulässiges Risiko unvereinbar sind – statt beides zu einem Mittelwert zu
verrechnen.

## Funktionsumfang

- **Risikoprofilierung** über drei Achsen. Die Ergebnisklasse ist das Minimum aus
  Tragfähigkeit und Bereitschaft, zusätzlich gedeckelt durch Kenntnisse und
  Erfahrungen. Eine unzureichende Liquiditätsreserve wirkt als
  Ausschlusskriterium, unabhängig von der Punktzahl.
- **Zielabgleich** gegen eine zieltypabhängige Mindesterfolgswahrscheinlichkeit,
  mit drei Ausgängen: erreichbar, erreichbar mit geringem Abstand, Konflikt.
- **Simulation** der Zielerreichung über einen zirkulären Block-Bootstrap
  (Blocklänge 12 Monate, 10 000 Pfade). Das Verfahren erhält Autokorrelation,
  Volatilitätscluster und Fat Tails der historischen Reihen.
- **Konfliktdiagnose**: Im Konfliktfall ergeht keine Empfehlung. An ihre Stelle
  treten drei numerisch gelöste Stellschrauben – erforderliche Sparrate,
  erforderliche Horizontverlängerung, erreichbarer Zielbetrag – jeweils mit
  Prüfung gegen die erhobenen wirtschaftlichen Verhältnisse.
- **Portfoliokonstruktion** durch restringierte Erwartungswert-Varianz-Optimierung
  mit Ledoit-Wolf-Shrinkage der Kovarianz, gegen ein gleichgewichtetes
  1/N-Portfolio als Referenz gemessen.
- **Nachhaltigkeitsfilter** nach Taxonomie, Offenlegungsverordnung und
  nachteiligen Auswirkungen, angewandt vor der Optimierung.
- **Dokumentation der Anlageentscheidung**: Geeignetheitserklärung und
  Konfliktbericht als PDF, vollständig aus dem aufgezeichneten
  Entscheidungspfad erzeugt. Jeder Satz ist an eine Regel-Kennung und deren
  Rechtsgrundlage gebunden.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make verify
```

`make verify` führt Linter, strikte Typprüfung und die vollständige Testsuite
aus. Erforderlich ist Python 3.12 oder neuer; entwickelt und geprüft wurde mit
CPython 3.13.

## Start

```bash
make app
```

Öffnet die Demonstrationsoberfläche: Erhebung des Anlageziels, der
wirtschaftlichen Verhältnisse und der Risikobereitschaft, anschließend
Risikoklasse, Zielabgleich und Ergebnisdokument.

## Reproduzierbarkeit

Gleiches Profil, gleicher Seed und gleiche Regelwerksfassung liefern denselben
Entscheidungspfad und dasselbe PDF – bitgenau. Dafür sorgen vier Festlegungen:

| Größe | Wert | Ort |
|---|---|---|
| Marktdaten | eingefrorener Snapshot, kein Live-Abruf | `data/snapshot/` |
| Datenstand | SHA-256 je Datei im Manifest | `data/snapshot/manifest.json` |
| Basis-Seed | `20260726` | `tests/conftest.py`, `Makefile` |
| Regelwerk | Fassungsnummer je Datei | `policies/*.yaml` |

Der Snapshot umfasst 221 Monatsrenditen von 2008-02 bis 2026-06 über acht
Assetklassen, abgeleitet aus ausschüttungsbereinigten Monatsschlusskursen.
Prüfsummen in Kurzform: Preisreihe `29cbdbb244a9`, Renditereihe `1dead02ee323`.

```bash
make verify-snapshot   # Datenstand gegen das Manifest prüfen
make verify-lock       # frische Umgebung allein aus requirements.lock
```

`make verify-lock` baut eine Wegwerf-Umgebung ausschließlich aus
`requirements.lock` und sammelt darin die Testsuite ein. Das ist der Weg, den
eine fremde Maschine geht.

## Projektstruktur

```
candor/
├── domain/       Datenmodelle, ohne Ein- und Ausgabe
├── profiling/    drei Achsen, bindendes Minimum, Sensitivität
├── goals/        Zielkatalog, erforderliche Rendite
├── engine/       Simulation, Zielabgleich, Glide-Path, Stresstests
├── portfolio/    Schätzung, Optimierung, Referenzportfolio, Backtest
├── trace/        Aufzeichnung des Entscheidungspfads
├── report/       Geeignetheitserklärung, Konfliktbericht, Abbildungen
└── data/         Abruf, Snapshot, Manifest, Kennzahlen

app/             Demonstrationsoberfläche
policies/        versioniertes Regelwerk (YAML, kein Code)
tests/           Einzelfälle, Eigenschaften, eingefrorene Referenzausgaben
data/snapshot/   eingefrorene Zeitreihen und Manifest
figures/         erzeugte Abbildungen
```

Die Schwellenwerte, Punktetabellen und Bandbreiten liegen ausschließlich in
`policies/`. Eine fachliche Änderung ist eine Änderung am Regelwerk, nicht am
Code; die verwendete Fassung wird in jedem Entscheidungspfad und in jedem
erzeugten Dokument ausgewiesen.

## Einordnung

Dies ist ein **Demonstrator**, kein Produktivsystem. Er führt keine Orders aus,
bindet keine Depotverwaltung an, berücksichtigt keine Steuern und rechnet Kosten
nur als Größenordnung. Die ausgewiesenen Erfolgswahrscheinlichkeiten sind
Modellrechnungen auf Grundlage historischer Monatsrenditen; ein Marktverlauf,
der schlechter ist als alles historisch Beobachtete, ist darin nicht enthalten.

**Keine Anlageberatung.** Der Quelltext dient der Demonstration eines
regelbasierten Entscheidungs- und Dokumentationsverfahrens.

## Lizenz

MIT – siehe `LICENSE`.

## Autor

Fabian Fladischer
