"""Dokumentmodell und deterministischer PDF-Satz.

Warum ein Zwischenmodell statt direkter PDF-Erzeugung
-----------------------------------------------------
Die erzeugten Dokumente werden **nicht** unmittelbar nach reportlab
geschrieben, sondern zuerst als Folge typisierter Blöcke aufgebaut
(:class:`Document`) und erst dann gesetzt. Drei Gründe:

1. **Prüfbarkeit.** Der Inhalt lässt sich als Text vergleichen, ohne ein PDF zu
   öffnen. Der Golden-Master-Test prüft damit den *Inhalt* und nicht die
   Satzausgabe — ein geänderter Zeilenumbruch bricht ihn nicht, ein geänderter
   Satz schon.
2. **Determinismus.** Das Zwischenmodell enthält keinen Zeitstempel, keinen
   Pfad und keine Zufallsquelle. Was nicht deterministisch ist, wird erst beim
   Setzen ergänzt und ist dort an einer Stelle greifbar.
3. **Trennung.** Wer den Inhalt ändert, fasst den Satz nicht an, und umgekehrt.

Determinismus des PDF
---------------------
reportlab schreibt standardmäßig Erstellungszeitpunkt und eine zufällige
Dokumentkennung in die Datei. Beides macht zwei Läufe byteverschieden, obwohl
der Inhalt gleich ist. ``rl_config.invariant`` schaltet das ab; zusätzlich wird
das Erzeugungsdatum ausdrücklich gesetzt. Damit gilt Invariante I4 bis in die
Bytes des PDF hinein — nachgewiesen in ``tests/test_golden.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Final

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

#: Farbpalette der erzeugten Dokumente.
NAVY: Final = colors.HexColor("#1F3864")
HELLBLAU: Final = colors.HexColor("#DCE6F4")
AKZENT: Final = colors.HexColor("#C00000")
GRAU: Final = colors.HexColor("#7F7F7F")

#: Seitenränder in Millimetern.
MARGIN_MM: Final = 20

#: Erzeugungsdatum der PDF-Metadaten. Fest gesetzt, damit zwei Läufe
#: bitidentische Dateien liefern (Invariante I4). Der fachlich maßgebliche
#: Stichtag steht im Dokument selbst, nicht in den Dateimetadaten.
PDF_CREATION_DATE: Final = date(2026, 7, 26)


class BlockKind(Enum):
    """Art eines Dokumentblocks.

    Attributes:
        TITLE: Dokumenttitel, einmal je Dokument.
        HEADING: Abschnittsüberschrift.
        SUBHEADING: Unterabschnitt.
        PARAGRAPH: Fließender Absatz aus einem Textbaustein.
        BULLETS: Aufzählung.
        TABLE: Tabelle mit Kopfzeile.
        NOTICE: Hervorgehobener Hinweis, etwa die Verweigerung der Empfehlung.
        PAGEBREAK: Seitenumbruch.
    """

    TITLE = "titel"
    HEADING = "ueberschrift"
    SUBHEADING = "unterueberschrift"
    PARAGRAPH = "absatz"
    BULLETS = "aufzaehlung"
    TABLE = "tabelle"
    NOTICE = "hinweis"
    PAGEBREAK = "seitenumbruch"


@dataclass(frozen=True, slots=True)
class Block:
    """Ein einzelner Dokumentblock.

    Attributes:
        kind: Art des Blocks.
        text: Textinhalt bei Titel, Überschrift, Absatz und Hinweis.
        items: Einträge einer Aufzählung.
        rows: Zeilen einer Tabelle, erste Zeile ist die Kopfzeile.
        rule_id: Regel-Kennung, aus der der Block stammt. Leer bei Blöcken, die
            keiner einzelnen Regel zuzuordnen sind (Überschriften, Umbrüche).
    """

    kind: BlockKind
    text: str = ""
    items: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    rule_id: str = ""

    def as_text(self) -> str:
        """Stellt den Block als Text dar — die Grundlage des Golden-Master.

        Returns:
            Der Block in einer stabilen, zeilenweise vergleichbaren Form.
        """
        marke = f"[{self.rule_id}] " if self.rule_id else ""
        if self.kind is BlockKind.PAGEBREAK:
            return "--- Seitenumbruch ---"
        if self.kind is BlockKind.BULLETS:
            zeilen = "\n".join(f"  - {eintrag}" for eintrag in self.items)
            return f"{self.kind.value.upper()}: {marke}\n{zeilen}"
        if self.kind is BlockKind.TABLE:
            zeilen = "\n".join("  | " + " | ".join(zelle for zelle in zeile) for zeile in self.rows)
            return f"{self.kind.value.upper()}: {marke}\n{zeilen}"
        return f"{self.kind.value.upper()}: {marke}{self.text}"


@dataclass(frozen=True, slots=True)
class Document:
    """Ein vollständiges Dokument als Folge von Blöcken.

    Attributes:
        title: Titel für die PDF-Metadaten.
        subject: Betreff für die PDF-Metadaten.
        blocks: Die Blöcke in Reihenfolge.
    """

    title: str
    subject: str
    blocks: tuple[Block, ...] = field(default_factory=tuple)

    def as_text(self) -> str:
        """Stellt das gesamte Dokument als Text dar.

        Returns:
            Der Inhalt in einer stabilen, zeilenweise vergleichbaren Form.
        """
        kopf = [f"TITEL: {self.title}", f"BETREFF: {self.subject}", ""]
        return "\n".join([*kopf, *(block.as_text() for block in self.blocks)]) + "\n"

    def rule_ids(self) -> tuple[str, ...]:
        """Liefert die Regel-Kennungen aller Blöcke in Reihenfolge.

        Belegt, dass der Inhalt aus benannten Regeln stammt (Prinzip P2).

        Returns:
            Die Kennungen ohne Leereinträge, Reihenfolge erhalten.
        """
        return tuple(block.rule_id for block in self.blocks if block.rule_id)


def _styles() -> dict[str, ParagraphStyle]:
    """Baut die Absatzformate des Dokuments.

    Returns:
        Zuordnung Formatname auf Absatzformat.
    """
    basis = getSampleStyleSheet()
    schrift = "Helvetica"
    fett = "Helvetica-Bold"
    return {
        "titel": ParagraphStyle(
            "CandorTitel",
            parent=basis["Title"],
            fontName=fett,
            fontSize=16,
            leading=20,
            textColor=NAVY,
            spaceAfter=4 * mm,
            alignment=0,
        ),
        "ueberschrift": ParagraphStyle(
            "CandorH1",
            parent=basis["Heading1"],
            fontName=fett,
            fontSize=12,
            leading=15,
            textColor=NAVY,
            spaceBefore=5 * mm,
            spaceAfter=2 * mm,
        ),
        "unterueberschrift": ParagraphStyle(
            "CandorH2",
            parent=basis["Heading2"],
            fontName=fett,
            fontSize=10.5,
            leading=13,
            textColor=NAVY,
            spaceBefore=3 * mm,
            spaceAfter=1.5 * mm,
        ),
        "absatz": ParagraphStyle(
            "CandorBody",
            parent=basis["BodyText"],
            fontName=schrift,
            fontSize=9.5,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=2 * mm,
        ),
        "aufzaehlung": ParagraphStyle(
            "CandorBullet",
            parent=basis["BodyText"],
            fontName=schrift,
            fontSize=9.5,
            leading=13,
            leftIndent=5 * mm,
            bulletIndent=1 * mm,
            spaceAfter=1 * mm,
        ),
        "hinweis": ParagraphStyle(
            "CandorNotice",
            parent=basis["BodyText"],
            fontName=fett,
            fontSize=10,
            leading=14,
            textColor=AKZENT,
            borderColor=AKZENT,
            borderWidth=0.8,
            borderPadding=3 * mm,
            spaceBefore=3 * mm,
            spaceAfter=3 * mm,
        ),
    }


def _table(rows: tuple[tuple[str, ...], ...], styles: dict[str, ParagraphStyle]) -> Table:
    """Setzt eine Tabelle mit Kopfzeile.

    Args:
        rows: Zeilen, erste ist die Kopfzeile.
        styles: Absatzformate.

    Returns:
        Die gesetzte Tabelle.
    """
    zellenformat = ParagraphStyle(
        "CandorCell", parent=styles["absatz"], fontSize=8.5, leading=11, alignment=0, spaceAfter=0
    )
    kopfformat = ParagraphStyle(
        "CandorCellHead", parent=zellenformat, fontName="Helvetica-Bold", textColor=NAVY
    )
    daten = [
        [Paragraph(zelle, kopfformat if i == 0 else zellenformat) for zelle in zeile]
        for i, zeile in enumerate(rows)
    ]
    tabelle = Table(daten, repeatRows=1, hAlign="LEFT")
    tabelle.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HELLBLAU),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY),
                ("LINEBELOW", (0, 1), (-1, -2), 0.25, GRAU),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return tabelle


def _flowables(document: Document, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    """Übersetzt die Blöcke in reportlab-Elemente.

    Args:
        document: Das zu setzende Dokument.
        styles: Absatzformate.

    Returns:
        Die Elemente in Reihenfolge.
    """
    elemente: list[Flowable] = []
    for block in document.blocks:
        if block.kind is BlockKind.PAGEBREAK:
            elemente.append(PageBreak())
        elif block.kind is BlockKind.TITLE:
            elemente.append(Paragraph(block.text, styles["titel"]))
        elif block.kind is BlockKind.HEADING:
            elemente.append(Paragraph(block.text, styles["ueberschrift"]))
        elif block.kind is BlockKind.SUBHEADING:
            elemente.append(Paragraph(block.text, styles["unterueberschrift"]))
        elif block.kind is BlockKind.PARAGRAPH:
            elemente.append(Paragraph(block.text, styles["absatz"]))
        elif block.kind is BlockKind.NOTICE:
            elemente.append(KeepTogether(Paragraph(block.text, styles["hinweis"])))
        elif block.kind is BlockKind.BULLETS:
            elemente.extend(
                Paragraph(eintrag, styles["aufzaehlung"], bulletText="–") for eintrag in block.items
            )
            elemente.append(Spacer(1, 2 * mm))
        elif block.kind is BlockKind.TABLE and block.rows:
            elemente.append(_table(block.rows, styles))
            elemente.append(Spacer(1, 3 * mm))
    return elemente


def render_pdf(document: Document, output: Path, footer: str = "") -> Path:
    """Setzt ein Dokument als PDF.

    Args:
        document: Das zu setzende Dokument.
        output: Zielpfad der PDF-Datei.
        footer: Fußzeile, auf jeder Seite wiederholt.

    Returns:
        Der Pfad der geschriebenen Datei.
    """
    # Muss vor dem Anlegen des Dokuments gesetzt werden: Ohne invariant schreibt
    # reportlab Erstellungszeitpunkt und eine zufällige Dokumentkennung in die
    # Datei, und zwei Läufe wären byteverschieden (Invariante I4). Die Zuweisung
    # steht hier und nicht auf Modulebene, damit der blosse Import von
    # candor.report.document keine globale Einstellung veraendert.
    rl_config.invariant = 1

    styles = _styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    vorlage = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=MARGIN_MM * mm,
        rightMargin=MARGIN_MM * mm,
        topMargin=MARGIN_MM * mm,
        bottomMargin=(MARGIN_MM + 5) * mm,
        title=document.title,
        subject=document.subject,
        author="CANDOR",
        creator="CANDOR",
        lang="de-DE",
    )

    def _fuss(canvas: object, doc: object) -> None:
        """Zeichnet Fußzeile und Seitenzahl.

        Args:
            canvas: Zeichenfläche der Seite.
            doc: Das laufende Dokument.
        """
        canvas.saveState()  # type: ignore[attr-defined]
        canvas.setFont("Helvetica", 7.5)  # type: ignore[attr-defined]
        canvas.setFillColor(GRAU)  # type: ignore[attr-defined]
        seite = f"Seite {doc.page}"  # type: ignore[attr-defined]
        canvas.drawString(MARGIN_MM * mm, 12 * mm, footer)  # type: ignore[attr-defined]
        canvas.drawRightString(  # type: ignore[attr-defined]
            A4[0] - MARGIN_MM * mm, 12 * mm, seite
        )
        canvas.restoreState()  # type: ignore[attr-defined]

    vorlage.build(_flowables(document, styles), onFirstPage=_fuss, onLaterPages=_fuss)
    return output


def extract_text(path: Path) -> str:
    """Liest den Text eines PDF zurück.

    Dient dem Golden-Master-Test: Er belegt, dass der Satz den Inhalt des
    Dokumentmodells tatsächlich in die Datei geschrieben hat und unterwegs
    nichts verloren ging.

    Args:
        path: Pfad der PDF-Datei.

    Returns:
        Der extrahierte Text aller Seiten.

    Raises:
        ImportError: Wenn ``pypdf`` nicht verfügbar ist.
    """
    try:
        # Bewusst verzoegert: pypdf ist eine reine Testabhaengigkeit. Ein Import
        # auf Modulebene wuerde candor.report.document ohne sie unbrauchbar
        # machen, obwohl der PDF-Satz selbst pypdf nicht braucht.
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError as fehler:  # pragma: no cover — nur ohne Testabhängigkeit
        msg = (
            "Für die Textextraktion wird 'pypdf' benötigt. Es ist eine reine "
            'Testabhängigkeit und über `pip install -e ".[dev]"` verfügbar.'
        )
        raise ImportError(msg) from fehler

    leser = PdfReader(str(path))
    return "\n".join(seite.extract_text() or "" for seite in leser.pages)
