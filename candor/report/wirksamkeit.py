"""Nachweis der Wirksamkeit als Abschnitte der Auswertungsdatei.

Erzeugt zwei Abschnitte vollständig aus gerechneten Zahlen:

* **§3 Stresstest** — die drei Krisenfenster gegen die zugesagte Verlusterwartung
  jeder Risikoklasse. Die Kernfrage: Hält, was die Klasse verspricht?
* **§5 Sensitivität** — welche einzelne Fragebogenangabe die Risikoklasse am
  stärksten bewegt.

Wie in allen Auswertungsmodulen des Projekts formuliert nichts hier eine
Bewertung. Die Befunde stehen als Zahlen mit ihrer Herleitung; die Einordnung
schreibt der Verfasser.

Aufruf:
    python -m candor.report.wirksamkeit --write-evaluation
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final

from candor import format_de
from candor.data.loader import load_returns, snapshot_info
from candor.data.policy_loader import load_policy_set
from candor.data.universe_meta import ASSET_KEYS
from candor.domain.goal import GoalType
from candor.domain.policy import PolicySet
from candor.engine.stress import StressResult, broken_promises, run_all
from candor.goals.model import build_goal
from candor.profiling.sensitivity import (
    SensitivityReport,
    analyse,
    baseline_clients,
)
from candor.report.figures import SENSITIVITY_DATE

#: Marker des Stresstestabschnitts.
STRESS_START: Final = "<!-- BEGIN:STRESS -->"
STRESS_END: Final = "<!-- END:STRESS -->"

#: Deutsche Anführungszeichen als Konstanten. Das schließende Zeichen beendet
#: innerhalb eines f-Strings sonst die Zeichenkette.
QUOTE_OPEN: Final = "\u201e"
QUOTE_CLOSE: Final = "\u201c"

#: Marker des Sensitivitätsabschnitts.
SENSITIVITY_START: Final = "<!-- BEGIN:SENSITIVITAET -->"
SENSITIVITY_END: Final = "<!-- END:SENSITIVITAET -->"


def _erholung(ergebnis: StressResult) -> str:
    """Formatiert die Erholungsdauer.

    Args:
        ergebnis: Ein Stresstestergebnis.

    Returns:
        Die Dauer in Monaten oder ein Hinweis auf das Ausbleiben.
    """
    if ergebnis.recovery_months is None:
        return "nicht erreicht"
    return f"{ergebnis.recovery_months}"


def render_stress(results: tuple[StressResult, ...], policies: PolicySet) -> str:
    """Baut den Stresstestabschnitt.

    Args:
        results: Ergebnisse aller Krisenfenster und Klassen.
        policies: Regelwerk, für die Fassungsangabe.

    Returns:
        Der Abschnitt als Markdown.
    """
    stand = datetime.now(UTC).date().isoformat()
    fenster = list(dict.fromkeys(ergebnis.window.key for ergebnis in results))
    klassen = sorted({ergebnis.risk_class for ergebnis in results})
    nach_schluessel = {(e.window.key, e.risk_class): e for e in results}
    info = snapshot_info()

    zeilen: list[str] = [
        STRESS_START,
        "",
        f"_Erzeugt am {stand} durch `make wirksamkeit`. Nicht von Hand bearbeiten —",
        "der Abschnitt wird bei jedem Lauf vollständig ersetzt._",
        "",
        f"**Datenstand:** {info.get('period_start')} bis {info.get('period_end')} · "
        f"Snapshot-Kurzhash `{info.get('sha256_short')}`. "
        f"**Regelwerk:** classes {policies.classes.version}.",
        "",
        "Gerechnet wird auf der **regelbasierten Referenzallokation** jeder Klasse, nicht auf "
        "dem optimierten Portfolio: Die Optimierung hängt an geschätzten Momenten aus einem "
        "Zeitfenster; ein Stresstest darauf würde messen, wie gut die Schätzung das "
        "Krisenfenster kannte. Es wird nichts gezogen und nichts simuliert — nur die "
        "tatsächlichen Monatsrenditen der Fenster.",
        "",
        "### 3.1 Die geprüften Fenster",
        "",
        "| Fenster | Zeitraum | verfügbare Monate | Charakter |",
        "|---|---|---:|---|",
    ]
    for schluessel in fenster:
        beispiel = nach_schluessel[schluessel, klassen[0]]
        verfuegbar = (
            f"{beispiel.months} von {beispiel.requested_months}"
            if beispiel.is_truncated
            else str(beispiel.months)
        )
        zeilen.append(
            f"| {beispiel.window.label} | {beispiel.window.start} bis {beispiel.window.end} | "
            f"{verfuegbar} | {beispiel.window.description} |"
        )

    verkuerzt = [
        nach_schluessel[s, klassen[0]]
        for s in fenster
        if nach_schluessel[s, klassen[0]].is_truncated
    ]
    if verkuerzt:
        zeilen += [
            "",
            "**Einschränkung.** "
            + " ".join(
                f"Für das Fenster {QUOTE_OPEN}{e.window.label}{QUOTE_CLOSE} liegen nur "
                f"{e.months} der angeforderten "
                f"{e.requested_months} Monate vor; der Snapshot beginnt später als das Fenster. "
                "Der ausgewiesene Rückgang ist damit eine **Untergrenze**."
                for e in verkuerzt
            ),
            "",
        ]

    zeilen += [
        "### 3.2 Größter Rückgang je Fenster und Klasse",
        "",
        "Die Spalte „Zusage“ nennt den erwarteten Maximalverlust aus "
        "`policies/classes_v1.yaml`. Er ist an die Verlustszenario-Leiter der "
        "Bereitschaftserhebung gebunden: Klasse N entspricht der dort vorgelegten Stufe S_N.",
        "",
        "| Klasse | Zusage | "
        + " | ".join(nach_schluessel[s, klassen[0]].window.label for s in fenster)
        + " |",
        "|---:|---:|" + "---:|" * len(fenster),
    ]
    for klasse in klassen:
        werte = []
        for schluessel in fenster:
            ergebnis = nach_schluessel[schluessel, klasse]
            marke = " ⚠" if not ergebnis.promise_held else ""
            werte.append(f"{format_de.percent(ergebnis.max_drawdown)}{marke}")
        zusage = nach_schluessel[fenster[0], klasse].promised_loss
        zeilen.append(f"| {klasse} | {format_de.percent(zusage)} | " + " | ".join(werte) + " |")

    gebrochen = broken_promises(results)
    zeilen += [
        "",
        "⚠ = die zugesagte Verlusterwartung wurde überschritten.",
        "",
        "### 3.3 Hielt die Zusage?",
        "",
        f"Geprüft wurden {len(results)} Kombinationen aus Fenster und Klasse. Die zugesagte "
        f"Verlusterwartung hielt in {len(results) - len(gebrochen)} Fällen und wurde in "
        f"{len(gebrochen)} Fällen überschritten.",
        "",
    ]
    if gebrochen:
        zeilen += [
            "| Fenster | Klasse | zugesagt | tatsächlich | Abweichung |",
            "|---|---:|---:|---:|---:|",
        ]
        zeilen += [
            f"| {e.window.label} | {e.risk_class} | {format_de.percent(e.promised_loss)} | "
            f"{format_de.percent(e.max_drawdown)} | "
            f"{format_de.decimal(e.deviation, 1)} pp |"
            for e in gebrochen
        ]
        zeilen.append("")

    # Der Befund zur Differenzierung: Wie weit spreizen sich die Klassen je Fenster?
    zeilen += [
        "### 3.4 Wie stark differenziert die Risikoklasse?",
        "",
        "Die Klassifizierung soll gerade bewirken, dass eine niedrige Klasse weniger verliert "
        "als eine hohe. Geprüft wird das als Spanne zwischen Klasse 1 und Klasse 5 je Fenster.",
        "",
        "| Fenster | Klasse 1 | Klasse 5 | Spanne | monoton steigend? |",
        "|---|---:|---:|---:|---|",
    ]
    for schluessel in fenster:
        drawdowns = [abs(nach_schluessel[schluessel, k].max_drawdown) for k in klassen]
        monoton = drawdowns == sorted(drawdowns)
        zeilen.append(
            f"| {nach_schluessel[schluessel, klassen[0]].window.label} | "
            f"{format_de.percent(-drawdowns[0])} | {format_de.percent(-drawdowns[-1])} | "
            f"{format_de.decimal((drawdowns[-1] - drawdowns[0]) * 100, 1)} pp | "
            f"{'ja' if monoton else 'nein'} |"
        )

    zeilen += [
        "",
        "### 3.5 Erholungsdauer in Monaten",
        "",
        "Gezählt wird vom Tiefpunkt bis zum Wiedererreichen des vorherigen Höchststands, über "
        "das Fenster hinaus in die weiterlaufende Reihe. Ein Krisenfenster endet definitionsgemäß "
        "im Tief; innerhalb des Fensters gäbe es keine Erholung zu finden.",
        "",
        "| Klasse | "
        + " | ".join(nach_schluessel[s, klassen[0]].window.label for s in fenster)
        + " |",
        "|---:|" + "---:|" * len(fenster),
    ]
    for klasse in klassen:
        werte = [_erholung(nach_schluessel[s, klasse]) for s in fenster]
        zeilen.append(f"| {klasse} | " + " | ".join(werte) + " |")

    zeilen += ["", STRESS_END, ""]
    return "\n".join(zeilen)


def render_sensitivity(reports: list[SensitivityReport]) -> str:
    """Baut den Sensitivitätsabschnitt.

    Args:
        reports: Berichte je Basisprofil.

    Returns:
        Der Abschnitt als Markdown.
    """
    stand = datetime.now(UTC).date().isoformat()
    gesammelt: dict[str, tuple[str, str, int, int, int]] = {}
    for bericht in reports:
        for eintrag in bericht.items:
            label, achse, spanne, ab, auf = gesammelt.get(
                eintrag.item, (eintrag.label, eintrag.axis, 0, 0, 0)
            )
            gesammelt[eintrag.item] = (
                label,
                achse,
                max(spanne, eintrag.span),
                max(ab, eintrag.downside),
                max(auf, eintrag.upside),
            )
    geordnet = sorted(gesammelt.items(), key=lambda eintrag: (-eintrag[1][2], eintrag[1][0]))

    zeilen: list[str] = [
        SENSITIVITY_START,
        "",
        f"_Erzeugt am {stand} durch `make wirksamkeit`. Nicht von Hand bearbeiten —",
        "der Abschnitt wird bei jedem Lauf vollständig ersetzt._",
        "",
        "Ein-Faktor-Analyse: Je Item wird **eine** Angabe über ihren gesamten zulässigen "
        "Bereich variiert, alles andere bleibt unverändert. Gemessen wird die resultierende "
        "**Ergebnisklasse**, nicht der Punktwert der Achse — ein Item kann viele Punkte bewegen "
        "und trotzdem folgenlos bleiben, wenn seine Achse nicht bindet.",
        "",
        "Gerechnet über drei Basisprofile mit unterschiedlicher bindender Achse; ausgewiesen ist "
        "der jeweils **größte** gemessene Hebel. Wechselwirkungen zwischen Items erfasst das "
        "Verfahren nicht.",
        "",
        "### 5.1 Die drei Basisprofile",
        "",
        "| Profil | Ergebnisklasse | bindende Achse |",
        "|---|---:|---|",
    ]
    zeilen += [
        f"| {bericht.profile_name} | {bericht.baseline_class} | {bericht.binding_axis} |"
        for bericht in reports
    ]

    zeilen += [
        "",
        "### 5.2 Hebel je Item",
        "",
        "| Item | Achse | Hebel abwärts | Hebel aufwärts | Spanne |",
        "|---|---|---:|---:|---:|",
    ]
    zeilen += [
        f"| {werte[0]} | {werte[1]} | {werte[3]} | {werte[4]} | **{werte[2]}** |"
        for _, werte in geordnet
    ]

    staerkste = [werte[0] for _, werte in geordnet if werte[2] == geordnet[0][1][2]]
    ohne_hebel = [werte[0] for _, werte in geordnet if werte[2] == 0]
    zeilen += [
        "",
        "Angaben in Klassenstufen. „Abwärts“ und „aufwärts“ beziehen sich auf das jeweilige "
        "Basisprofil; die Spanne ist der Abstand zwischen der niedrigsten und der höchsten "
        "erreichbaren Ergebnisklasse.",
        "",
        "### 5.3 Befund",
        "",
        f"Den größten Hebel tragen: **{', '.join(staerkste)}** "
        f"({geordnet[0][1][2]} Klassenstufen).",
        "",
        "Beide sind **keine** Bestandteile eines gewichteten Punktescores, sondern wirken als "
        "Deckel beziehungsweise als Ausschlusskriterium: der Kenntnisstand nach Art. 55 DelVO "
        "(EU) 2017/565, die Liquiditätsreserve als K.-o.-Kriterium: Eine fehlende Rücklage "
        "wirkt nicht additiv, sondern entwertet die übrigen Kriterien, statt von ihnen "
        "ausgeglichen zu werden. Das am stärksten "
        "gewichtete Item der Bereitschaftserhebung — der hingenommene Verlustbetrag mit einem "
        "Gewicht von 35 Prozent — bewegt die Ergebnisklasse um höchstens zwei Stufen.",
        "",
        "Das ist die gewollte Folge der Konstruktion und zugleich ihre Kehrseite: Die Sorgfalt "
        "bei der Erhebung dieser beiden Angaben entscheidet mehr als der gesamte übrige "
        "Fragebogen.",
        "",
    ]
    if ohne_hebel:
        zeilen += [
            f"Ohne jeden Hebel über alle drei Basisprofile: {', '.join(ohne_hebel)}. Diese "
            "Angaben gehen in den Punktescore ein, verschieben ihn aber nie über eine "
            "Klassenschwelle — jedenfalls nicht bei den geprüften Profilen.",
            "",
        ]

    zeilen += [SENSITIVITY_END, ""]
    return "\n".join(zeilen)


def _ersetzen(inhalt: str, start: str, ende: str, block: str, marke: str) -> str:
    """Ersetzt einen markierten Abschnitt oder fügt ihn hinter einer Marke ein.

    Args:
        inhalt: Bisheriger Dateiinhalt.
        start: Startmarker.
        ende: Endmarker.
        block: Einzusetzender Abschnitt.
        marke: Überschrift, hinter der beim ersten Mal eingefügt wird.

    Returns:
        Der neue Dateiinhalt.
    """
    if start in inhalt and ende in inhalt:
        vorher = inhalt.split(start, maxsplit=1)[0]
        nachher = inhalt.split(ende, 1)[1]
        return f"{vorher}{block}{nachher}"
    if marke in inhalt:
        vorher, _, rest = inhalt.partition(marke)
        _, _, nachher = rest.partition("---")
        return f"{vorher}{marke}\n\n{block}\n\n---{nachher}"
    return f"{inhalt.rstrip()}\n\n{block}\n"  # pragma: no cover — Marke fehlt


def write_evaluation(
    stress_block: str, sensitivity_block: str, evaluation_path: Path | None = None
) -> Path:
    """Schreibt beide Abschnitte in die Auswertungsdatei.

    Args:
        stress_block: Abschnitt 3.
        sensitivity_block: Abschnitt 5.
        evaluation_path: Abweichender Pfad der Zieldatei.

    Returns:
        Pfad der geschriebenen Datei.

    Raises:
        FileNotFoundError: Wenn die Zieldatei nicht existiert.
    """
    ziel = evaluation_path or (Path(__file__).resolve().parents[2] / "auswertung.md")
    if not ziel.is_file():
        msg = f"Zieldatei fehlt: {ziel}"
        raise FileNotFoundError(msg)

    inhalt = ziel.read_text(encoding="utf-8")
    inhalt = _ersetzen(
        inhalt,
        STRESS_START,
        STRESS_END,
        stress_block,
        "## 3. Stresstest gegen historische Krisenphasen",
    )
    inhalt = _ersetzen(
        inhalt,
        SENSITIVITY_START,
        SENSITIVITY_END,
        sensitivity_block,
        "## 5. Fallbeispiele",
    )
    ziel.write_text(inhalt, encoding="utf-8")
    return ziel


def build_blocks() -> tuple[str, str]:
    """Rechnet beide Abschnitte.

    Returns:
        Stresstest- und Sensitivitätsabschnitt.
    """
    policies = load_policy_set()
    stress = render_stress(run_all(load_returns(), policies, ASSET_KEYS), policies)

    anlageziel = build_goal(
        "Z-SENS-A",
        GoalType.ALTERSVORSORGE,
        Decimal("300000.00"),
        SENSITIVITY_DATE,
        policies.goals,
        initial_capital=Decimal("20000.00"),
        monthly_contribution=Decimal("500.00"),
    )
    berichte = [
        analyse(kunde, anlageziel, policies, name) for name, kunde in baseline_clients().items()
    ]
    return stress, render_sensitivity(berichte)


def main(argv: list[str] | None = None) -> int:
    """Einstiegspunkt der Kommandozeile.

    Args:
        argv: Argumente ohne Programmnamen.

    Returns:
        ``0`` bei Erfolg.
    """
    parser = argparse.ArgumentParser(description="Wirksamkeitsnachweis.")
    parser.add_argument(
        "--write-evaluation",
        action="store_true",
        help="Ergebnis in die Auswertungsdatei schreiben statt auf die Konsole",
    )
    args = parser.parse_args(argv)

    stress, sensitivitaet = build_blocks()
    if args.write_evaluation:
        ziel = write_evaluation(stress, sensitivitaet)
        sys.stdout.write(f"Abschnitte 3 und 5 geschrieben nach {ziel}\n")
    else:
        sys.stdout.write(stress + "\n" + sensitivitaet + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
