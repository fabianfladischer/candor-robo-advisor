"""Golden-Master-Tests der erzeugten Dokumente.

Was ein Golden-Master hier leistet
----------------------------------
Drei feste Profile ergeben drei feste Dokumente. Die Referenzausgaben liegen
unter ``tests/golden/`` im Repository. Jede Änderung an einem Textbaustein, an
einer Formatierung oder an einem Rechenschritt verändert die Ausgabe und bricht
den Test — auch dann, wenn sie unbeabsichtigt war und sonst niemandem aufgefallen
wäre.

Das ist der Zweck: Bei einem Dokument, das eine Anlageempfehlung begründet, ist
eine *stille* Änderung der schlimmere Fall. Ein gebrochener Test zwingt zur
Entscheidung, ob die Änderung gewollt war.

Zwei Ebenen
-----------
1. **Dokumentmodell gegen Referenztext.** Der genaue Vergleich. Er prüft den
   Inhalt und ist unempfindlich gegen Zeilenumbrüche des Satzes.
2. **PDF-Text gegen Dokumentmodell.** Die Gegenprobe. Sie belegt, dass der Satz
   den Inhalt tatsächlich in die Datei geschrieben hat und unterwegs nichts
   verloren ging — ein Fehler, den Ebene 1 allein nicht fände.

Referenzen neu erzeugen
-----------------------
Nach einer *gewollten* Änderung::

    CANDOR_UPDATE_GOLDEN=1 pytest tests/test_golden.py

Danach die Änderung im Diff prüfen und mit committen. Ohne Prüfung des Diffs ist
die Neuerzeugung wertlos — sie würde genau den Schutz aufheben, den der Test
bietet.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from candor.data.universe_meta import get_instrument
from candor.report import conflict, examples, suitability, textblocks
from candor.report.document import Block, BlockKind, Document, extract_text, render_pdf
from candor.report.examples import EXAMPLES, build_example
from candor.report.textblocks import known_rule_ids

#: Verzeichnis der Referenzausgaben.
GOLDEN_DIR: Final = Path(__file__).resolve().parent / "golden"

#: Umgebungsvariable, die die Referenzen neu schreibt.
UPDATE_ENV: Final = "CANDOR_UPDATE_GOLDEN"

#: Erwarteter Ausgang je Beispielfall. Ändert sich einer davon, ist das ein
#: fachlicher Befund und kein Formatierungsdetail — deshalb eigens festgehalten.
EXPECTED_OUTCOME: Final[dict[str, bool]] = {
    "konservativ": True,
    "ausgewogen": True,
    "konflikt": False,
}

#: Zeilen der PDF-Extraktion, die nicht aus dem Dokumentmodell stammen:
#: Fußzeile und Seitenzahl setzt der Satz, nicht der Inhalt.
FOOTER_PATTERN: Final = re.compile(r"^(CANDOR · .*|Seite \d+)$")


def _dokument(name: str) -> Document:
    """Baut das zum Fall passende Dokument.

    Args:
        name: Kennung des Beispielfalls.

    Returns:
        Das Dokumentmodell.
    """
    advice = build_example(name)
    modul = suitability if advice.is_recommendation else conflict
    return modul.build_document(advice)


def _referenz(name: str, inhalt: str) -> str:
    """Liest die Referenz oder schreibt sie neu.

    Args:
        name: Kennung des Beispielfalls.
        inhalt: Aktuell erzeugter Inhalt.

    Returns:
        Der Referenzinhalt.
    """
    pfad = GOLDEN_DIR / f"{name}.txt"
    if os.environ.get(UPDATE_ENV) == "1" or not pfad.is_file():
        pfad.parent.mkdir(parents=True, exist_ok=True)
        pfad.write_text(inhalt, encoding="utf-8")
    return pfad.read_text(encoding="utf-8")


# --- Ebene 1: Dokumentmodell gegen Referenztext ----------------------------


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_dokument_entspricht_der_referenz(name: str) -> None:
    """Das erzeugte Dokument stimmt zeichengleich mit der Referenz überein.

    Args:
        name: Kennung des Beispielfalls.
    """
    erzeugt = _dokument(name).as_text()
    referenz = _referenz(name, erzeugt)

    if erzeugt != referenz:
        erzeugte_zeilen = erzeugt.splitlines()
        referenzzeilen = referenz.splitlines()
        for nummer, (links, rechts) in enumerate(
            zip(referenzzeilen, erzeugte_zeilen, strict=False), start=1
        ):
            if links != rechts:
                pytest.fail(
                    f"Abweichung von der Referenz in Zeile {nummer}:\n"
                    f"  Referenz: {links!r}\n"
                    f"  erzeugt : {rechts!r}\n"
                    f"War die Änderung gewollt, Referenzen mit {UPDATE_ENV}=1 neu erzeugen."
                )
        pytest.fail(
            f"Die Referenz hat {len(referenzzeilen)} Zeilen, das erzeugte Dokument "
            f"{len(erzeugte_zeilen)}."
        )


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_ausgang_des_beispielfalls_bleibt_stabil(name: str) -> None:
    """Der Ausgang jedes Beispielfalls ändert sich nicht unbemerkt.

    Der Test steht neben dem Golden-Master, weil er eine andere Frage
    beantwortet: Nicht „hat sich der Text geändert", sondern „hat sich die
    Entscheidung geändert". Das Zweite wäre der schwerere Befund.

    Args:
        name: Kennung des Beispielfalls.
    """
    advice = build_example(name)

    assert advice.is_recommendation == EXPECTED_OUTCOME[name], (
        f"Der Fall {name!r} endet jetzt "
        f"mit {advice.reconciliation.outcome.value!r} statt wie erwartet. "
        "Das ist ein fachlicher Befund, keine Formatierungsfrage."
    )


# --- Ebene 2: PDF-Text gegen Dokumentmodell --------------------------------


def _normalisiert(text: str) -> str:
    """Vereinheitlicht Leerraum für den Vergleich mit dem PDF-Text.

    Nötig aus zwei Gründen: Der Satz bricht lange Tabellenzellen über mehrere
    Zeilen um, und die Textextraktion gibt das geschützte Leerzeichen des
    Tausendertrenners als gewöhnliches Leerzeichen zurück. Beides ist kein
    Inhaltsunterschied und darf den Vergleich nicht brechen.

    Args:
        text: Zu normalisierender Text.

    Returns:
        Der Text mit vereinheitlichtem Leerraum.
    """
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _pdf_text(pfad: Path) -> str:
    """Liest den PDF-Text und entfernt die Zeilen des Satzes.

    Args:
        pfad: Pfad der PDF-Datei.

    Returns:
        Der normalisierte Inhalt ohne Fußzeile und Seitenzahl.
    """
    zeilen = [
        zeile
        for zeile in extract_text(pfad).splitlines()
        if zeile.strip() and not FOOTER_PATTERN.match(zeile.strip())
    ]
    return _normalisiert(" ".join(zeilen))


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_pdf_enthaelt_den_inhalt_des_dokuments(name: str, tmp_path: Path) -> None:
    """Der Satz übernimmt jede Überschrift und jeden Tabellenwert ins PDF.

    Geprüft werden Überschriften und Tabellenzellen, nicht die Fließabsätze:
    Absätze bricht der Satz um, ihre Zeilen stehen im PDF deshalb anders als im
    Modell. Überschriften und Zellen bleiben unverändert und sind damit der
    belastbare Nachweis, dass nichts verloren ging.

    Args:
        name: Kennung des Beispielfalls.
        tmp_path: Temporäres Verzeichnis.
    """
    advice = build_example(name)
    modul = suitability if advice.is_recommendation else conflict
    pfad = modul.write_pdf(advice, tmp_path / f"{name}.pdf")
    im_pdf = _pdf_text(pfad)

    fehlend: list[str] = []
    for block in _dokument(name).blocks:
        if block.kind.value in {"ueberschrift", "unterueberschrift", "titel"}:
            erwartet = _normalisiert(block.text)
            if erwartet and erwartet not in im_pdf:
                fehlend.append(block.text)
        for zeile in block.rows:
            fehlend.extend(
                zelle
                for zelle in zeile
                if len(_normalisiert(zelle)) > 2 and _normalisiert(zelle) not in im_pdf
            )

    assert not fehlend, f"Im PDF fehlen {len(fehlend)} Inhalte, zuerst: {fehlend[:5]}"


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_pdf_ist_bitidentisch_reproduzierbar(name: str, tmp_path: Path) -> None:
    """I4 hält über die gesamte Kette bis in die Bytes des PDF.

    Der schärfste Determinismustest des Projekts. Er schließt die Kette von der
    Profilierung über Simulation und Optimierung bis zum Satz. reportlab
    schreibt ohne ``rl_config.invariant`` einen Erstellungszeitpunkt und eine
    zufällige Dokumentkennung in die Datei; beides würde hier auffallen.

    Args:
        name: Kennung des Beispielfalls.
        tmp_path: Temporäres Verzeichnis.
    """
    pruefsummen: list[str] = []
    trace_hashes: list[str] = []
    for lauf in range(2):
        advice = build_example(name)
        modul = suitability if advice.is_recommendation else conflict
        pfad = modul.write_pdf(advice, tmp_path / f"{name}_{lauf}.pdf")
        pruefsummen.append(hashlib.sha256(pfad.read_bytes()).hexdigest())
        trace_hashes.append(advice.trace_hash)

    assert trace_hashes[0] == trace_hashes[1], "Der Trace-Hash unterscheidet sich zwischen Läufen."
    assert pruefsummen[0] == pruefsummen[1], (
        f"Zwei Läufe liefern verschiedene PDF-Dateien: {pruefsummen[0][:16]} gegen "
        f"{pruefsummen[1][:16]}."
    )


# --- Bindung von Text an Regeln (Prinzip P2) -------------------------------


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_jeder_inhaltsblock_haengt_an_einer_bekannten_regel(name: str) -> None:
    """Jede Regel-Kennung im Dokument hat einen hinterlegten Textbaustein.

    Args:
        name: Kennung des Beispielfalls.
    """
    verwendet = set(_dokument(name).rule_ids())

    unbekannt = verwendet - known_rule_ids()
    assert not unbekannt, f"Ohne Textbaustein: {sorted(unbekannt)}"


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_jede_verwendete_regel_steht_auch_im_trace(name: str) -> None:
    """Was das Dokument einer Regel zuschreibt, hat diese Regel auch aufgezeichnet.

    Die zentrale Zusicherung dieser Tests: Das Dokument kann nichts behaupten, was
    im Entscheidungspfad nicht steht. Ausgenommen ist allein ``ADV-STOP`` im
    Konfliktfall — die Regel wird dort aufgezeichnet, ihr Textbaustein beschreibt
    aber das Ausbleiben eines Schritts und nicht dessen Ergebnis.

    Args:
        name: Kennung des Beispielfalls.
    """
    advice = build_example(name)
    modul = suitability if advice.is_recommendation else conflict
    dokument = modul.build_document(advice)
    im_trace = {ereignis.rule_id for ereignis in advice.trace}

    for kennung in set(dokument.rule_ids()):
        assert kennung in im_trace, (
            f"Das Dokument beschreibt Regel {kennung!r}, der Entscheidungspfad enthält sie "
            "aber nicht."
        )


def test_alle_textbausteine_werden_verwendet() -> None:
    """Kein Textbaustein bleibt über alle drei Fälle hinweg ungenutzt.

    Ein nie verwendeter Baustein ist entweder toter Text oder ein vergessener
    Aufruf. Beides ist ein Befund. Die K.-o.-Bausteine sind ausgenommen: Sie
    greifen nur bei Profilen, die die Beispielfälle bewusst nicht abbilden.
    """
    nur_bei_sonderfaellen = {"CAP-KO-LIQ", "TOL-CONFLICT", "GLD-CAP", "REC-LEVER"}
    verwendet: set[str] = set()
    for name in EXAMPLES:
        verwendet.update(_dokument(name).rule_ids())

    ungenutzt = known_rule_ids() - verwendet - nur_bei_sonderfaellen
    assert not ungenutzt, f"Ungenutzte Textbausteine: {sorted(ungenutzt)}"


# --- Abgrenzung der beiden Dokumenttypen -----------------------------------


def test_geeignetheitserklaerung_verweigert_den_konfliktfall() -> None:
    """Für einen Konfliktfall wird keine Geeignetheitserklärung erzeugt.

    Sie würde eine Empfehlung dokumentieren, die es nicht gibt.
    """
    advice = build_example("konflikt")

    with pytest.raises(suitability.SuitabilityError, match="keine"):
        suitability.build_document(advice)


def test_konfliktbericht_verweigert_den_empfehlungsfall() -> None:
    """Für einen empfehlbaren Fall wird kein Konfliktbericht erzeugt."""
    advice = build_example("konservativ")

    with pytest.raises(conflict.ConflictReportError, match="kein Konflikt"):
        conflict.build_document(advice)


def test_konfliktbericht_nennt_die_verweigerung_vor_allen_zahlen() -> None:
    """Der Sperrvermerk steht unmittelbar hinter dem Titel.

    Wer nur den Anfang liest, muss die Kernaussage bereits erfasst haben. Ein
    Hinweis im vierten Abschnitt käme zu spät.
    """
    dokument = conflict.build_document(build_example("konflikt"))
    erste_bloecke = dokument.blocks[:2]

    assert erste_bloecke[0].kind.value == "titel"
    assert "keine Anlageempfehlung" in erste_bloecke[0].text
    assert erste_bloecke[1].kind.value == "hinweis"
    assert "keine Anlageempfehlung" in erste_bloecke[1].text
    assert "Art. 54 Abs. 10" in erste_bloecke[1].text


def test_konfliktbericht_traegt_die_verweigerung_im_titel() -> None:
    """Der Dokumenttitel selbst sagt bereits, dass keine Empfehlung ergeht."""
    dokument = conflict.build_document(build_example("konflikt"))

    assert "keine Anlageempfehlung" in dokument.title
    assert "Geeignetheitserklärung" not in dokument.title


def test_geeignetheitserklaerung_weist_das_portfolio_vollstaendig_aus() -> None:
    """Jede Position des Portfolios erscheint mit ISIN und Bandbreite."""
    advice = build_example("ausgewogen")
    assert advice.optimization is not None
    dokument = suitability.build_document(advice)
    text = dokument.as_text()

    for i, schluessel in enumerate(advice.optimization.asset_keys):
        if float(advice.optimization.weights[i]) < suitability.MIN_WEIGHT:
            continue
        assert get_instrument(schluessel).isin in text


def test_gefilterte_position_traegt_im_dokument_kein_gewicht() -> None:
    """Eine aus Nachhaltigkeitsgründen ausgeschlossene Position taucht nicht auf.

    Der Fall ``ausgewogen`` verlangt die Berücksichtigung nachteiliger
    Auswirkungen; GOLD erfüllt das laut Regelwerk nicht und wird ausgeschlossen.
    """
    advice = build_example("ausgewogen")
    assert advice.universe is not None
    assert advice.optimization is not None
    assert "GOLD" in advice.universe.excluded

    assert float(advice.optimization.weights[advice.optimization.asset_keys.index("GOLD")]) == 0.0
    assert "WisdomTree Physical Gold" not in suitability.build_document(advice).as_text()


# --- Textbausteine: Fehlerpfade (Prinzip P5) -------------------------------


def test_unbekannte_regel_hat_keinen_baustein() -> None:
    """Für eine Regel ohne Baustein wird kein Text erfunden."""
    with pytest.raises(KeyError, match="kein Textbaustein"):
        textblocks.render("GIBT-ES-NICHT")


def test_fehlender_wert_bricht_den_baustein_ab() -> None:
    """Ein Baustein mit fehlendem Wert erzeugt keinen Platzhalter, sondern einen Fehler.

    Eine Geeignetheitserklärung mit einer Lücke ist schlechter als keine: Die
    Lücke bliebe unbemerkt, der fehlende Wert nicht.
    """
    with pytest.raises(KeyError, match="fehlt der Wert"):
        textblocks.render("CLS-MIN", tragfaehigkeit=3)


def test_unbekannter_statischer_baustein_bricht_ab() -> None:
    """Auch ein statischer Baustein wird nicht stillschweigend leer geliefert."""
    with pytest.raises(KeyError, match="Kein statischer"):
        textblocks.static("gibt-es-nicht")


def test_jeder_statische_baustein_ist_gefuellt() -> None:
    """Kein statischer Baustein ist leer oder ein Platzhalter."""
    for schluessel in textblocks.STATIC:
        text = textblocks.static(schluessel)
        assert len(text) > 40
        assert "{" not in text


def test_jeder_regelbaustein_nennt_seine_felder_vollstaendig() -> None:
    """Kein Baustein enthält nach dem Füllen noch eine offene Klammer.

    Fängt Tippfehler in den Vorlagen ab, die sonst erst im fertigen Dokument
    auffielen — und dort als seltsamer Text und nicht als Fehler.
    """
    for kennung, vorlage in textblocks.TEMPLATES.items():
        felder = re.findall(r"\{(\w+)\}", vorlage)
        gefuellt = textblocks.render(kennung, **dict.fromkeys(felder, "X"))
        assert "{" not in gefuellt, kennung
        assert "}" not in gefuellt, kennung


# --- Dokumentmodell --------------------------------------------------------


def test_jeder_blocktyp_hat_eine_textdarstellung() -> None:
    """Alle Blockarten lassen sich als Text darstellen — Grundlage des Vergleichs."""
    bloecke = [
        Block(BlockKind.TITLE, text="Titel"),
        Block(BlockKind.HEADING, text="Überschrift"),
        Block(BlockKind.PARAGRAPH, text="Absatz", rule_id="CLS-MIN"),
        Block(BlockKind.BULLETS, items=("eins", "zwei")),
        Block(BlockKind.TABLE, rows=(("Kopf", "Wert"), ("a", "b"))),
        Block(BlockKind.NOTICE, text="Hinweis"),
        Block(BlockKind.PAGEBREAK),
    ]

    for block in bloecke:
        assert block.as_text().strip()

    assert "[CLS-MIN]" in bloecke[2].as_text()
    assert "Seitenumbruch" in bloecke[6].as_text()


def test_leeres_dokument_laesst_sich_setzen(tmp_path: Path) -> None:
    """Auch ein Dokument ohne Blöcke erzeugt eine gültige Datei.

    Args:
        tmp_path: Temporäres Verzeichnis.
    """
    pfad = render_pdf(Document(title="Leer", subject="Leer"), tmp_path / "leer.pdf")

    assert pfad.is_file()
    assert pfad.read_bytes().startswith(b"%PDF")


def test_dokumenttext_nennt_titel_und_betreff() -> None:
    """Der Kopf der Textdarstellung trägt die Metadaten des Dokuments."""
    text = Document(title="T", subject="B").as_text()

    assert text.startswith("TITEL: T\nBETREFF: B")


# --- Beispielfälle und Kommandozeile ---------------------------------------


def test_unbekannter_beispielfall_wird_abgewiesen() -> None:
    """Ein nicht hinterlegter Beispielfall ist ein Fehler."""
    with pytest.raises(KeyError, match="Unbekannter Beispielfall"):
        build_example("gibt-es-nicht")


def test_kommandozeile_schreibt_die_geeignetheitserklaerung(tmp_path: Path) -> None:
    """``python -m candor.report.suitability`` erzeugt die Datei.

    Args:
        tmp_path: Temporäres Verzeichnis.
    """
    rueckgabe = suitability.main(["--fall", "konservativ", "--out", str(tmp_path)])

    assert rueckgabe == 0
    assert (tmp_path / "geeignetheit_konservativ.pdf").is_file()


def test_kommandozeile_verweigert_die_erklaerung_im_konfliktfall(tmp_path: Path) -> None:
    """Für den Konfliktfall liefert die Kommandozeile einen Fehlercode.

    Args:
        tmp_path: Temporäres Verzeichnis.
    """
    assert suitability.main(["--fall", "konflikt", "--out", str(tmp_path)]) == 1


def test_kommandozeile_schreibt_den_konfliktbericht(tmp_path: Path) -> None:
    """``python -m candor.report.conflict`` erzeugt die Datei.

    Args:
        tmp_path: Temporäres Verzeichnis.
    """
    rueckgabe = conflict.main(["--fall", "konflikt", "--out", str(tmp_path)])

    assert rueckgabe == 0
    assert (tmp_path / "konflikt_konflikt.pdf").is_file()


def test_kommandozeile_verweigert_den_bericht_ohne_konflikt(tmp_path: Path) -> None:
    """Ohne Konflikt liefert die Kommandozeile einen Fehlercode.

    Args:
        tmp_path: Temporäres Verzeichnis.
    """
    assert conflict.main(["--fall", "konservativ", "--out", str(tmp_path)]) == 1


def test_kommandozeile_erzeugt_alle_beispieldokumente(tmp_path: Path) -> None:
    """``python -m candor.report.examples`` schreibt genau drei Dokumente.

    Args:
        tmp_path: Temporäres Verzeichnis.
    """
    rueckgabe = examples.main(["--out", str(tmp_path)])
    dateien = sorted(p.name for p in tmp_path.glob("*.pdf"))

    assert rueckgabe == 0
    assert len(dateien) == len(EXAMPLES)
    assert any("konfliktbericht" in name for name in dateien)
    assert sum("geeignetheitserklaerung" in name for name in dateien) == 2


# --- Der Konfliktbericht ohne gerechnete Stellschrauben --------------------


def test_konfliktbericht_ohne_stellschrauben_benennt_die_luecke(
    tmp_path: Path,
) -> None:
    """Wurden die Stellschrauben nicht gerechnet, sagt der Bericht das ausdrücklich.

    Die Suche ist abschaltbar, weil sie mehrere Simulationen kostet. Ein Bericht
    ohne sie darf aber nicht so aussehen, als gäbe es keine Auswege — er muss
    benennen, dass sie nicht bestimmt wurden.

    Args:
        tmp_path: Temporäres Verzeichnis.
    """
    advice = build_example("konflikt")
    assert advice.reconciliation.conflict_report is not None
    ohne = replace(
        advice,
        reconciliation=replace(
            advice.reconciliation,
            conflict_report=replace(advice.reconciliation.conflict_report, levers=()),
        ),
    )

    text = conflict.build_document(ohne).as_text()
    assert "nicht berechnet" in text

    pfad = conflict.write_pdf(ohne, tmp_path / "ohne_hebel.pdf")
    assert pfad.is_file()
