"""Konfliktbericht als eigenständiges Dokument.

Warum ein eigenes Dokument und keine Variante der Geeignetheitserklärung
------------------------------------------------------------------------
Ein Konfliktbericht ist das Gegenteil einer Geeignetheitserklärung: Er
dokumentiert, dass **keine** Empfehlung ergeht. Beides in ein Dokument zu legen
und über einen Abschnitt zu unterscheiden, wäre die bequemere Umsetzung und
zugleich die gefährlichere — ein Leser, der ein Dokument mit dem Titel
„Geeignetheitserklärung“ in der Hand hält, liest darin eine Empfehlung, auch
wenn im vierten Abschnitt das Gegenteil steht.

Deshalb: eigener Titel, eigener Aufbau, und der Hinweis auf die Verweigerung an
erster Stelle — vor allen Zahlen, hervorgehoben und mit Rechtsgrundlage. Erst
danach folgt die Diagnose, und erst danach die drei Stellschrauben.

Die Stellschrauben sind keine Empfehlung
----------------------------------------
Sie beantworten die Frage, was sich ändern müsste, und sind ausdrücklich als
Rechengrößen ausgewiesen. Eine nicht umsetzbare Stellschraube wird mit ihrem
Wert genannt **und** als nicht umsetzbar gekennzeichnet; sie wird nicht
verschwiegen und nicht als Vorschlag verkauft.

Aufruf:
    python -m candor.report.conflict --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

from candor import format_de
from candor.engine.advise import Advice
from candor.engine.reconcile import CONFLICT_LEGAL_BASIS, all_levers_infeasible
from candor.report.document import Block, BlockKind, Document, render_pdf
from candor.report.suitability import _nachweisabschnitt, _profilabschnitt, _simulationsabschnitt
from candor.report.textblocks import render, static

#: Titel des Dokuments. Bewusst unmissverständlich.
TITLE: Final = "Konfliktbericht — keine Anlageempfehlung"


class ConflictReportError(ValueError):
    """Für den vorliegenden Fall ist kein Konfliktbericht zu erstellen."""


def _sperrvermerk() -> Block:
    """Baut den hervorgehobenen Hinweis auf die Verweigerung der Empfehlung.

    Steht an erster Stelle des Dokuments, noch vor jeder Zahl. Wer nur den
    Anfang liest, muss die Kernaussage bereits erfasst haben.

    Returns:
        Der Block.
    """
    return Block(
        BlockKind.NOTICE,
        rule_id="REC-CONFLICT",
        text=render("REC-CONFLICT"),
    )


def _diagnoseabschnitt(advice: Advice) -> list[Block]:
    """Baut den Abschnitt mit der Diagnose in Stichpunkten.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Die Blöcke des Abschnitts.

    Raises:
        ConflictReportError: Wenn kein Konfliktbericht vorliegt.
    """
    bericht = advice.reconciliation.conflict_report
    if bericht is None:
        msg = "Ohne Konfliktbericht ist der Diagnoseabschnitt nicht erzeugbar."
        raise ConflictReportError(msg)
    return [
        Block(BlockKind.HEADING, text="3. Befund"),
        Block(BlockKind.BULLETS, rule_id="REC-OUT", items=bericht.diagnosis),
    ]


def _stellschraubenabschnitt(advice: Advice) -> list[Block]:
    """Baut den Abschnitt mit den drei Stellschrauben.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Die Blöcke des Abschnitts.

    Raises:
        ConflictReportError: Wenn kein Konfliktbericht vorliegt.
    """
    bericht = advice.reconciliation.conflict_report
    if bericht is None:
        msg = "Ohne Konfliktbericht ist der Stellschraubenabschnitt nicht erzeugbar."
        raise ConflictReportError(msg)

    if not bericht.levers:
        return [
            Block(BlockKind.HEADING, text="4. Stellschrauben"),
            Block(
                BlockKind.PARAGRAPH,
                text=(
                    "Die Stellschrauben wurden für diesen Lauf nicht berechnet. Ohne sie "
                    "benennt der Bericht den Konflikt, beziffert aber nicht, was sich ändern "
                    "müsste."
                ),
            ),
        ]

    def _einheit(wert: object, unit: str) -> str:
        """Formatiert einen Wert entsprechend seiner Einheit.

        Args:
            wert: Zu formatierender Wert.
            unit: Einheit der Stellschraube.

        Returns:
            Der formatierte Wert.
        """
        if unit == "Monate":
            return f"{int(wert)} Monate"  # type: ignore[call-overload]
        return format_de.euro(wert)  # type: ignore[arg-type]

    zeilen: list[tuple[str, ...]] = [("", "Stellschraube", "Ist", "Erforderlich", "Machbarkeit")]
    absaetze: list[Block] = []
    for hebel in bericht.levers:
        zeilen.append(
            (
                hebel.lever_id.value,
                hebel.name,
                _einheit(hebel.current_value, hebel.unit),
                _einheit(hebel.required_value, hebel.unit),
                hebel.feasibility.value,
            )
        )
        absaetze.append(
            Block(
                BlockKind.PARAGRAPH,
                rule_id="REC-LEVER",
                text=render(
                    "REC-LEVER",
                    kennung=hebel.lever_id.value,
                    bezeichnung=hebel.name,
                    ist=_einheit(hebel.current_value, hebel.unit),
                    soll=_einheit(hebel.required_value, hebel.unit),
                    bewertung=hebel.feasibility_note,
                ),
            )
        )

    einordnung = (
        "Keine der drei Stellschrauben ist unter den erhobenen Verhältnissen umsetzbar. Das "
        "Ziel ist in der vorliegenden Form nicht erreichbar; eine Änderung der Ausgangslage "
        "ist erforderlich."
        if all_levers_infeasible(bericht)
        else (
            "Umsetzbar sind: "
            + ", ".join(hebel.lever_id.value for hebel in bericht.offered_levers)
            + ". Die Stellschrauben sind keine Empfehlung, sondern die quantifizierte Antwort "
            "auf die Frage, was sich ändern müsste. Welche davon in Betracht kommt, ist im "
            "Beratungsgespräch zu klären."
        )
    )

    return [
        Block(BlockKind.HEADING, text="4. Stellschrauben"),
        Block(BlockKind.TABLE, rows=tuple(zeilen)),
        *absaetze,
        Block(BlockKind.SUBHEADING, text="4.1 Einordnung"),
        Block(BlockKind.PARAGRAPH, text=einordnung),
    ]


def _zielabschnitt_kurz(advice: Advice) -> list[Block]:
    """Baut den knappen Zielabschnitt des Konfliktberichts.

    Kürzer als in der Geeignetheitserklärung: Im Konfliktfall interessiert nicht
    die Herleitung der erforderlichen Rendite, sondern ihr Verhältnis zu dem,
    was die zulässige Risikoklasse hergibt.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Die Blöcke des Abschnitts.
    """
    ziel = advice.goal
    rendite = advice.required
    return [
        Block(BlockKind.HEADING, text="2. Anlageziel und zulässige Risikoklasse"),
        Block(
            BlockKind.TABLE,
            rows=(
                ("Angabe", "Wert"),
                ("Zielkennung", ziel.goal_id),
                ("Zieltyp", ziel.goal_type.value),
                ("Priorität", ziel.priority.value),
                ("Zielbetrag", format_de.euro(ziel.target_amount)),
                (
                    "Anlagehorizont",
                    f"{ziel.horizon_months} Monate "
                    f"({format_de.decimal(ziel.horizon_years, 2)} Jahre)",
                ),
                ("Startkapital", format_de.euro(ziel.initial_capital)),
                ("Monatliche Sparrate", format_de.euro(ziel.monthly_contribution)),
                (
                    "Erforderliche Rendite nominal p. a.",
                    format_de.percent(rendite.nominal.annual_rate),
                ),
                (
                    "Zulässige Risikoklasse",
                    f"{advice.profile.risk_class} — bindend: {advice.profile.binding_axis.value}",
                ),
            ),
        ),
        Block(
            BlockKind.PARAGRAPH,
            rule_id="CLS-MIN",
            text=render(
                "CLS-MIN",
                tragfaehigkeit=advice.profile.capacity.risk_class,
                bereitschaft=advice.profile.tolerance.risk_class,
                deckel=advice.profile.knowledge.max_risk_class,
                klasse=advice.profile.class_before_glidepath,
                achse=advice.profile.axis_before_glidepath.value,
                endklasse=advice.profile.risk_class,
            ),
        ),
    ]


def build_document(advice: Advice) -> Document:
    """Baut den Konfliktbericht als Dokumentmodell.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Das Dokument.

    Raises:
        ConflictReportError: Wenn der Fall kein Konflikt ist.
    """
    if not advice.has_conflict:
        msg = (
            f"Für Kunde {advice.client.client_id} und Ziel {advice.goal.goal_id} liegt kein "
            f"Konflikt vor (Ausgang: {advice.reconciliation.outcome.value}). Ein "
            "Konfliktbericht würde eine Verweigerung dokumentieren, die es nicht gibt."
        )
        raise ConflictReportError(msg)

    bloecke: list[Block] = [
        Block(BlockKind.TITLE, text=TITLE),
        _sperrvermerk(),
        Block(
            BlockKind.PARAGRAPH,
            text=(
                f"Kunde {advice.client.client_id} · Ziel {advice.goal.goal_id} · "
                f"Stichtag {advice.goal.reference_date.strftime('%d.%m.%Y')}"
            ),
        ),
        Block(BlockKind.HEADING, text="1. Zweck und Rechtsgrundlage"),
        Block(BlockKind.PARAGRAPH, text=static("zweck_konflikt")),
        Block(
            BlockKind.PARAGRAPH,
            text=f"Rechtsgrundlage der Verweigerung: {CONFLICT_LEGAL_BASIS}.",
        ),
        *_zielabschnitt_kurz(advice),
        *_diagnoseabschnitt(advice),
        Block(BlockKind.PAGEBREAK),
        *_stellschraubenabschnitt(advice),
        Block(BlockKind.PAGEBREAK),
        Block(BlockKind.HEADING, text="5. Erfasstes Kundenprofil"),
        *_profilabschnitt(advice)[1:],
        Block(BlockKind.PAGEBREAK),
        Block(BlockKind.HEADING, text="6. Simulierte Zielerreichung"),
        *_simulationsabschnitt(advice)[1:],
        Block(BlockKind.HEADING, text="7. Risikohinweise"),
        Block(BlockKind.PARAGRAPH, text=static("risiko_allgemein")),
        Block(BlockKind.PARAGRAPH, text=static("risiko_simulation")),
        Block(BlockKind.PARAGRAPH, text=static("risiko_annahmen")),
        Block(BlockKind.PARAGRAPH, rule_id="ADV-STOP", text=render("ADV-STOP")),
        *_nachweisabschnitt(advice, 8),
    ]
    return Document(
        title=TITLE,
        subject=f"Konfliktbericht {advice.client.client_id}/{advice.goal.goal_id}",
        blocks=tuple(bloecke),
    )


def write_pdf(advice: Advice, output: Path) -> Path:
    """Erzeugt den Konfliktbericht als PDF.

    Args:
        advice: Das Ergebnis der Entscheidungskette.
        output: Zielpfad der PDF-Datei.

    Returns:
        Der Pfad der geschriebenen Datei.
    """
    dokument = build_document(advice)
    fusszeile = (
        f"CANDOR · KEINE EMPFEHLUNG · {advice.client.client_id}/{advice.goal.goal_id} · "
        f"Trace {advice.trace_hash[:12]}"
    )
    return render_pdf(dokument, output, fusszeile)


def main(argv: list[str] | None = None) -> int:
    """Einstiegspunkt der Kommandozeile.

    Args:
        argv: Argumente ohne Programmnamen.

    Returns:
        ``0`` bei Erfolg, ``1`` wenn der Fall kein Konflikt ist.
    """
    # Verzoegert: candor.report.examples importiert dieses Modul; ein Import
    # auf Modulebene waere ein Zirkelbezug.
    from candor.report.examples import EXAMPLES, build_example  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Konfliktbericht erzeugen.")
    parser.add_argument("--fall", choices=sorted(EXAMPLES), default="konflikt")
    parser.add_argument("--out", type=Path, default=Path("beispiele"))
    args = parser.parse_args(argv)

    advice = build_example(args.fall)
    try:
        ziel = write_pdf(advice, args.out / f"konflikt_{args.fall}.pdf")
    except ConflictReportError as fehler:
        sys.stdout.write(f"Kein Konfliktbericht: {fehler}\n")
        return 1
    sys.stdout.write(f"Geschrieben: {ziel}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
