"""Wächter gegen Aussagen des Dokuments, die die Oberfläche nicht einlöst.

**Anlass.** Die Geeignetheitserklärung behauptete, zur Erhebung der
Verlusttoleranz seien konkrete Eurobeträge vorgelegt worden. Die einzige
Oberfläche legte Prozentwerte vor. Der Widerspruch blieb lange unbemerkt und
wurde von keiner Schranke erfasst:

* Kein Test importierte ``app/``.
* mypy prüft nur ``candor/`` (``pyproject.toml``, ``files = ["candor"]``).
* Der Golden-Master fixiert den Dokumenttext, nicht den Fragebogen.
* Der Textbaustein ``TOL-SCEN`` trägt kein einziges Feld und ist deshalb
  strukturell nicht falsifizierbar — er bliebe auch grün, wenn er das Gegenteil
  behauptete.

**Was diese Tests fixieren.** Nicht, dass die Funktion tut, was sie tut — das
wäre zirkulär. Sondern die drei Eigenschaften, deren Verlust den Befund erzeugt
hat:

1. Die Beschriftungen hängen am Anlageziel. Wer sie wieder als Konstanten
   pflegt, bricht ``test_beschriftung_folgt_dem_anlageziel``.
2. Die Verlustfrage zeigt keine Prozentangabe. Das ist die Aussage, die das
   Kundendokument trifft.
3. Der Divergenzsatz existiert genau einmal im Quelltext.

**Reichweite.** Die Tests prüfen die Datenseite der Oberfläche, nicht ihr
Rendering. Ob ``st.radio`` die Beschriftungen tatsächlich anzeigt, ist damit
weiterhin ungeprüft; dafür wäre ein Streamlit-Testlauf nötig, den das Projekt
nicht unterhält.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Final

import pytest

from app import streamlit_app
from candor import format_de
from candor.data.policy_loader import load_policy_set
from candor.domain.client import SustainabilityComponent, SustainabilityPreferences
from candor.domain.errors import DomainError
from candor.domain.goal import Goal, GoalType
from candor.domain.policy import PolicySet
from candor.goals.model import build_goal
from candor.profiling.tolerance import scenario_amounts
from candor.report import suitability
from candor.report.examples import build_example
from candor.report.textblocks import TEMPLATES

PROJEKTWURZEL: Final = Path(__file__).resolve().parent.parent

#: Stichtag der Testziele, gleichgezogen mit ``tests/builders.py``.
STICHTAG: Final = date(2026, 7, 26)

#: Monate je Jahr — reine Kalenderkonstante.
MONATE_JE_JAHR: Final = 12


@pytest.fixture(scope="module")
def regelwerk() -> PolicySet:
    """Lädt das Regelwerk einmal je Modul.

    :return: Das geprüfte Regelwerk.
    """
    return load_policy_set()


def _ziel(regelwerk: PolicySet, *, startkapital: str, sparrate: str, jahre: int) -> Goal:
    """Baut ein Anlageziel für die Beschriftungsprobe.

    :param regelwerk: Das geprüfte Regelwerk.
    :param startkapital: Bereits vorhandenes Kapital in Euro.
    :param sparrate: Monatliche Sparrate in Euro.
    :param jahre: Anlagehorizont in Jahren.
    :return: Das geprüfte Ziel.
    """
    return build_goal(
        "Z-UI-0001",
        GoalType.ALTERSVORSORGE,
        Decimal("300000"),
        STICHTAG,
        regelwerk.goals,
        target_date=date(STICHTAG.year + jahre, STICHTAG.month, 1),
        initial_capital=Decimal(startkapital),
        monthly_contribution=Decimal(sparrate),
    )


def test_beschriftung_folgt_dem_anlageziel(regelwerk: PolicySet) -> None:
    """Zwei verschiedene Ziele müssen zu verschiedenen Beschriftungen führen.

    Das ist der eigentliche Wächter: Fest gepflegte Beschriftungen — gleich ob
    Prozentwerte oder Eurobeträge — sind für beide Ziele identisch und brechen
    diesen Test. Nur eine Beschriftung, die aus der Bezugsgröße gerechnet wird,
    besteht ihn.
    """
    klein = streamlit_app.verlustszenario_optionen(
        _ziel(regelwerk, startkapital="0", sparrate="100", jahre=10), regelwerk.tolerance
    )
    gross = streamlit_app.verlustszenario_optionen(
        _ziel(regelwerk, startkapital="500000", sparrate="2000", jahre=30), regelwerk.tolerance
    )

    assert klein.keys() == gross.keys()
    abweichend = [stufe for stufe in klein if klein[stufe] != gross[stufe]]
    assert len(abweichend) == len(klein) - 1, (
        "Die Beschriftungen der Verlustfrage hängen nicht am Anlageziel. Erwartet "
        "war, dass sich alle Stufen außer der Nullstufe zwischen zwei sehr "
        f"verschiedenen Zielen unterscheiden. Klein: {klein}. Groß: {gross}."
    )


def test_beschriftung_stammt_aus_dem_entscheidungspfad(regelwerk: PolicySet) -> None:
    """Die Beträge müssen aus ``scenario_amounts`` kommen, nicht aus einer Eigenrechnung.

    Eine naheliegende Eigenrechnung in der Oberfläche — Startkapital zuzüglich
    Sparrate mal zwölf mal Jahre — weicht ab, weil der Anlagehorizont
    angebrochene Monate abschneidet. Bei den Vorbelegungen der Demo sind das 299
    statt 300 Monate und damit 100,00 € Unterschied auf der Stufe S3.
    """
    ziel = _ziel(regelwerk, startkapital="20000", sparrate="500", jahre=25)
    beschriftungen = streamlit_app.verlustszenario_optionen(ziel, regelwerk.tolerance)
    betraege = scenario_amounts(ziel, regelwerk.tolerance)

    assert beschriftungen == {stufe: format_de.euro(b) for stufe, b in betraege.items()}

    # Gegenprobe: Die naheliegende Eigenrechnung ergibt andere Betraege. Ohne
    # diese Abweichung belegte die Zusicherung oben nur eine Tautologie.
    naiv = Decimal("20000") + Decimal("500") * (25 * MONATE_JE_JAHR)
    aus_dem_kern = ziel.initial_capital + ziel.monthly_contribution * ziel.horizon_months
    assert aus_dem_kern != naiv, "Der Testfall trifft die Abweichung nicht mehr."
    assert format_de.euro(naiv * Decimal("0.20")) not in beschriftungen.values(), (
        "Die Beschriftung stimmt mit der Eigenrechnung ueberein — dann ist offen, "
        "ob sie aus scenario_amounts stammt."
    )


def test_verlustfrage_zeigt_keine_prozentangabe(regelwerk: PolicySet) -> None:
    """Weder Fragetext noch Beschriftungen der Verlustfrage dürfen ein Prozentzeichen führen.

    Das ist die Aussage, die der Textbaustein ``TOL-SCEN`` im Kundendokument
    trifft. Sie muss an der Oberfläche prüfbar sein, sonst ist sie eine
    Behauptung über ein Verhalten, das niemand kontrolliert.
    """
    ziel = _ziel(regelwerk, startkapital="20000", sparrate="500", jahre=25)
    beschriftungen = streamlit_app.verlustszenario_optionen(ziel, regelwerk.tolerance)

    assert "%" not in streamlit_app.SCENARIO_QUESTION
    assert "Prozent" not in streamlit_app.SCENARIO_QUESTION
    treffer = sorted(text for text in beschriftungen.values() if "%" in text or "Prozent" in text)
    assert not treffer, (
        f"Die Verlustfrage zeigt Prozentangaben: {treffer}. Die "
        "Geeignetheitserklärung behauptet das Gegenteil (Baustein TOL-SCEN)."
    )


def test_verlustfrage_steht_nicht_mehr_in_der_konstantenliste() -> None:
    """Die Verlustfrage darf nicht nebenher als Konstante gepflegt werden.

    ``TOLERANCE_CHOICES`` ist die Liste der Items mit festen Beschriftungen. Ein
    Eintrag für die Verlustszenario-Dimension dort wäre der Rückweg in den
    Befund: zwei Quellen für dieselbe Angabe.
    """
    regelwerk = load_policy_set()
    dimension = regelwerk.tolerance.verlustszenarien.dimension
    assert dimension not in streamlit_app.TOLERANCE_CHOICES, (
        f"Das Item {dimension} steht wieder in TOLERANCE_CHOICES. Seine "
        "Beschriftung wird aus dem Anlageziel berechnet und darf nicht "
        "daneben als Konstante gepflegt werden."
    )


def test_der_textbaustein_belegt_seine_eigene_behauptung() -> None:
    """``TOL-SCEN`` muss den vorgelegten Betrag nennen, nicht nur behaupten, es gebe ihn.

    Ein feldloser Baustein ist nicht falsifizierbar. Erst ein Feld, das aus dem
    Ergebnisobjekt gefüllt wird, bindet den Satz an den Entscheidungspfad.
    """
    baustein = TEMPLATES["TOL-SCEN"]
    felder = set(re.findall(r"\{(\w+)\}", baustein))
    assert felder == {"bezugsbetrag", "betrag"}, (
        f"TOL-SCEN führt die Felder {sorted(felder)}. Erwartet waren "
        "bezugsbetrag und betrag — ohne sie behauptet der Baustein eine "
        "Erhebungsform, die das Dokument nicht belegt."
    )


def test_divergenzsatz_existiert_nur_einmal_im_quelltext() -> None:
    """Der Bildschirmsatz zur Divergenz darf nur an einer Stelle stehen.

    Der Befund, der zu dieser Prüfung geführt hat, entstand nicht aus einem
    Rechenfehler, sondern daraus, dass derselbe Sachverhalt an mehreren Stellen
    unabhängig formuliert wurde. Ein zweites Vorkommen ist der Anfang derselben Drift.
    """
    kennsatz = "Gesprächsstoff, nicht Rechengröße"
    fundstellen = [
        pfad.relative_to(PROJEKTWURZEL)
        for ordner in ("candor", "app")
        for pfad in (PROJEKTWURZEL / ordner).rglob("*.py")
        if kennsatz in pfad.read_text(encoding="utf-8")
    ]
    assert fundstellen == [Path("candor/domain/result.py")], (
        f"Der Divergenzsatz steht an {len(fundstellen)} Stellen: {fundstellen}. "
        "Einzige Quelle ist Divergence.as_note()."
    )


# --- Nachhaltigkeitspräferenzen --------------------------------------------


def _praeferenz(gewuenscht: bool, *, pai: bool, anteil: str = "0.50") -> SustainabilityPreferences:
    """Baut die Präferenzen so, wie die Oberfläche sie zusammensetzt.

    :param gewuenscht: Zustand des Häkchens „Ich habe Nachhaltigkeitspräferenzen".
    :param pai: Ob die PAI-Ausprägung gewählt wurde.
    :param anteil: Erhobener Mindestanteil der gewählten Ausprägung.
    :return: Die Präferenzen.
    """
    return SustainabilityPreferences(
        has_preferences=gewuenscht,
        pai_considered=SustainabilityComponent(
            requested=pai, minimum_share=Decimal(anteil) if pai else None
        ),
    )


def test_haekchen_ohne_auswahl_ist_keine_verneinung() -> None:
    """Die drei Zustände der Präferenzerhebung müssen unterscheidbar bleiben.

    Zeitweise bog die Oberfläche „Häkchen gesetzt, nichts gewählt" auf
    ``has_preferences=False`` um. Die Geeignetheitserklärung schrieb dann „Die
    Angabe wurde erhoben und ausdrücklich verneint" — eine Erklärung, die der
    Kunde nie abgegeben hatte. Der Zustand ist eine unvollständige
    Angabe und bricht ab (Prinzip P5).
    """
    assert not _praeferenz(False, pai=False).has_preferences
    assert _praeferenz(True, pai=True).has_preferences

    with pytest.raises(DomainError, match="keine der drei Ausprägungen"):
        _praeferenz(True, pai=False)

    # Die Oberfläche muss das Häkchen unverändert durchreichen. Jede Umrechnung
    # an dieser Stelle stellt den dritten Zustand als einen der beiden anderen dar.
    quelltext = (PROJEKTWURZEL / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "has_preferences=gewuenscht" in quelltext, (
        "Die Oberfläche leitet has_preferences nicht mehr unmittelbar aus dem "
        "Häkchen ab. Eine Umrechnung verwischt den Unterschied zwischen "
        "Verneinung und unvollständiger Angabe."
    )


def test_kein_erfundener_mindestanteil_in_der_oberflaeche() -> None:
    """Die Oberfläche darf keinen Mindestanteil setzen, den sie nicht erhoben hat.

    Zeitweise stand in ``app/streamlit_app.py`` unsichtbar
    ``minimum_share=Decimal("0.50")``. Der Wert war nie abgefragt, nie
    veränderbar und wurde dennoch als erhobene Kundenangabe ausgewiesen —
    ein stiller Vorgabewert in einer Pflichtangabe.
    """
    quelltext = (PROJEKTWURZEL / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    treffer = re.findall(r"minimum_share=Decimal\(", quelltext)
    assert not treffer, (
        "Die Oberfläche setzt den Mindestanteil wieder als Literal. Er muss aus "
        "der Erhebung stammen (_mindestanteil_eingabe)."
    )
    assert "_mindestanteil_eingabe" in quelltext


def test_dokument_weist_verlangten_und_erreichten_anteil_aus() -> None:
    """Die Nachhaltigkeitstabelle muss beide Größen führen.

    Der verlangte Mindestanteil allein ist eine Behauptung. Erst neben dem
    erreichten Anteil belegt das Dokument, dass die Vorgabe eingehalten wurde —
    und macht sichtbar, dass der harte Ausschluss mehr durchsetzt, als der
    Kunde verlangt hat.
    """
    advice = build_example("ausgewogen")
    dokument = suitability.build_document(advice)
    tabellen = [block.rows for block in dokument.blocks if block.rows]
    kopfzeilen = [rows[0] for rows in tabellen]

    assert ("Ausprägung", "Verlangt", "Mindestanteil", "Erreicht") in kopfzeilen

    esg = next(rows for rows in tabellen if rows[0][0] == "Ausprägung")
    pai = next(zeile for zeile in esg[1:] if zeile[0].startswith("Nachteilige"))
    assert pai[1] == "ja"
    assert pai[2] == "60,00 %"
    assert pai[3] == "100,00 %"


def test_erreichter_anteil_stammt_aus_dem_trace() -> None:
    """Der ausgewiesene Anteil muss aus dem Trace kommen, nicht aus einer Zweitrechnung.

    Das Dokument liest ihn über ``Advice.events_for``. Damit gilt auch für diese
    Zahl die Zusicherung des Moduls: Was im Dokument steht, steht im Trace.
    """
    advice = build_example("ausgewogen")
    ereignisse = advice.events_for("ADV-ESG-QUOTE")

    assert len(ereignisse) == 1
    ausgabe = ereignisse[0].output
    assert isinstance(ausgabe, dict)
    assert ausgabe["erreichte_anteile"] == {"pai": "1.0000"}
    assert ausgabe["alle_vorgaben_eingehalten"] is True
