"""Tests des Zielabgleichs und der Stellschraubensuche.

Schwerpunkte:

* **I7 — Monotonie der Erfolgswahrscheinlichkeit.** Die Invariante zerfällt in
  zwei ungleich starke Aussagen. In Richtung Zielbetrag gilt sie *exakt*: Für
  jede feste Pfadschar ist der Anteil der Pfade oberhalb einer Schwelle eine
  fallende Treppenfunktion dieser Schwelle. In Richtung Horizont gilt sie *nur
  im Rahmen des Monte-Carlo-Rauschens*; siehe die Begründung an
  :func:`test_i7_horizont_monoton_im_rahmen_des_rauschens`.
* **I9 — keine Empfehlung ohne bestandene Prüfung.** Geprüft auf der Abbildung
  (``classify_outcome``) und auf der gesamten Kette (``reconcile``): Im
  Konfliktfall ist die Empfehlung gesperrt, ein Konfliktbericht vorhanden und
  Art. 54 Abs. 10 DelVO im Trace belegt.
* **Bisektion auf verrauschter Zielfunktion.** Der fixierte Seed macht die
  Zielfunktion deterministisch; Nichtkonvergenz führt zu ``SolverError`` und
  nicht zu einem stillen Zwischenwert.
* **Grenzfälle.** Ziel bereits erreicht · Ziel unmöglich · Erfolgswahrschein-
  lichkeit exakt auf der Schwelle.

Die Tests rechnen mit reduzierter Pfadzahl. Das ist zulässig, weil sie
Struktur­eigenschaften prüfen und nicht die Genauigkeit der Schätzung; die
Genauigkeitsfrage beantwortet die Konvergenzanalyse.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from candor import format_de
from candor.data.policy_loader import load_policy_set
from candor.data.universe_meta import ASSET_KEYS
from candor.domain.goal import GoalType
from candor.domain.policy import PolicySet
from candor.domain.result import (
    ConflictReport,
    Feasibility,
    LeverId,
    ReconcileOutcome,
    Reconciliation,
    RiskProfile,
)
from candor.engine.convergence import load_history
from candor.engine.reconcile import (
    CONFLICT_LEGAL_BASIS,
    all_levers_infeasible,
    build_diagnosis,
    classify_outcome,
    format_probability,
    reconcile,
)
from candor.engine.simulate import BlockBootstrapSampler, FloatArray, simulate
from candor.engine.solve import (
    MAX_CONTRIBUTION,
    MAX_HORIZON_MONTHS,
    BisectionResult,
    SolverContext,
    SolverError,
    bisect_decreasing,
    bisect_increasing,
    build_levers,
    solve_contribution,
    solve_horizon,
    solve_target,
    solve_target_by_bisection,
)
from candor.portfolio.universe import reference_weights
from candor.profiling.classify import classify
from candor.report.factsheet import main as factsheet_main
from candor.report.factsheet import reference_conflict_case, render_factsheet
from candor.trace.recorder import TraceMetadata, TraceRecorder
from tests.builders import STICHTAG, haushalt, kunde, ziel

#: Pfadzahl der schnellen Tests. Für Struktureigenschaften ausreichend.
TEST_PATHS = 400

#: Pfadzahl der Monotonietests. Höher, weil dort das Rauschen selbst gemessen wird.
MONOTONY_PATHS = 2000

#: Seed aller Tests dieses Moduls.
TEST_SEED = 20260726

#: Das Regelwerk wird einmal geladen; es ist unveränderlich.
POLICIES: PolicySet = load_policy_set()

#: Versionsangaben der Form ``1.2.0`` — dort ist der Punkt kein Dezimaltrenner.
VERSION_MUSTER = re.compile(r"\d+\.\d+\.\d+")

#: Trennzeichen der Ausgabe, aus dem Modul bezogen statt hier wiederholt. Eine
#: eigene Kopie hätte genau den Fehler ermöglicht, den diese Tests suchen.
TRENNER = format_de.THOUSANDS_SEPARATOR

#: Jede zusammenhängende Folge aus Ziffern und Trennzeichen.
ZAHL_MUSTER = re.compile(rf"\d[\d.,{TRENNER}]*")

#: Deutsche Zahlgrammatik: Tausendertrennzeichen in Dreiergruppen, Komma als
#: Dezimaltrenner. Ein Punkt zwischen Ziffern — also ein als Dezimaltrenner
#: missbrauchter Punkt — bricht das Muster.
DEUTSCHE_ZAHL = re.compile(rf"^\d{{1,3}}({TRENNER}\d{{3}})*(,\d+)?$")

#: Datumsangaben nach DIN 5008; dort trennt der Punkt keine Dezimalstellen.
DATUM_MUSTER = re.compile(r"\d{2}\.\d{2}\.\d{4}")

#: Gliederungsverweise wie „siehe Abschnitt 3.2“ — ebenfalls keine Rechengröße.
VERWEIS_MUSTER = re.compile(r"Abschnitt\s+\d+(\.\d+)*")


def _ist_ausgenommen(zeile: str) -> bool:
    """Gibt an, ob eine Zeile von der Zahlprüfung ausgenommen ist.

    Ausgenommen sind Überschriften (deren Gliederungsnummern keine gerechneten
    Zahlen sind), Instrumentennamen wie ``1.5-10.5yr``, die Aufzählung der
    Policy-Fassungen und der Trace-Hash.

    Args:
        zeile: Zu prüfende Zeile des Faktenblatts.

    Returns:
        ``True``, wenn die Zeile nicht geprüft wird.
    """
    return (
        zeile.startswith("#")
        or "UCITS" in zeile
        or "Policy-Fassungen" in zeile
        or zeile.startswith("| Trace-Hash")
    )


# --- Gemeinsame Fixtures ---------------------------------------------------


@pytest.fixture(scope="module")
def sampler() -> BlockBootstrapSampler:
    """Ziehungsverfahren auf der vollständigen Historie.

    Returns:
        Der Block-Bootstrap-Sampler des Projekts.
    """
    return BlockBootstrapSampler(load_history(POLICIES), POLICIES.simulation.blocklaenge_monate)


def _recorder(client_id: str = "K-REC", goal_id: str = "Z-REC") -> TraceRecorder:
    """Baut einen leeren Recorder.

    Args:
        client_id: Kennung des Kunden.
        goal_id: Kennung des Ziels.

    Returns:
        Der Recorder.
    """
    return TraceRecorder(
        TraceMetadata(
            client_id=client_id,
            goal_id=goal_id,
            policy_versions=dict(POLICIES.versions),
            package_version="0.1.0",
        )
    )


def _context(sampler: BlockBootstrapSampler, risk_class: int = 4) -> SolverContext:
    """Baut den Rahmen der Stellschraubensuche.

    Args:
        sampler: Ziehungsverfahren.
        risk_class: Zulässige Risikoklasse.

    Returns:
        Der Suchrahmen mit fixiertem Seed.
    """
    return SolverContext(
        sampler=sampler,
        weights=reference_weights(risk_class, POLICIES.classes, ASSET_KEYS),
        policies=POLICIES,
        paths=TEST_PATHS,
        seed=TEST_SEED,
        asset_keys=ASSET_KEYS,
    )


def _abgleich(sampler: BlockBootstrapSampler, **abweichungen: Any) -> Reconciliation:
    """Rechnet einen vollständigen Abgleich.

    Args:
        sampler: Ziehungsverfahren.
        **abweichungen: Abweichungen am Ziel gegenüber dem Grundfall.

    Returns:
        Das Ergebnis des Abgleichs.
    """
    kunde_ = kunde()
    ziel_ = ziel(**abweichungen)
    recorder = _recorder(kunde_.client_id, ziel_.goal_id)
    profil = classify(kunde_, ziel_, POLICIES, recorder)
    return reconcile(
        kunde_,
        ziel_,
        profil,
        POLICIES,
        sampler,
        recorder,
        asset_keys=ASSET_KEYS,
        paths=TEST_PATHS,
        seed=TEST_SEED,
    )


@pytest.fixture(scope="module")
def wertverlauf(sampler: BlockBootstrapSampler) -> FloatArray:
    """Eine einzige Pfadschar samt Monatsverlauf für die Monotonietests.

    Beide Richtungen von I7 werden an **derselben** Pfadschar geprüft. Nur so
    misst der Test die Eigenschaft der Abbildung und nicht den Unterschied
    zwischen zwei Ziehungen.

    Args:
        sampler: Ziehungsverfahren.

    Returns:
        Wertmatrix der Form ``(pfade, monate + 1)``.
    """
    ergebnis = simulate(
        sampler,
        reference_weights(4, POLICIES.classes, ASSET_KEYS),
        initial_capital=20_000.0,
        monthly_contribution=500.0,
        horizon_months=360,
        target_amount=1.0,
        policy=POLICIES.simulation,
        paths=MONOTONY_PATHS,
        seed=TEST_SEED,
        asset_keys=ASSET_KEYS,
        track_bands=False,
        track_values=True,
    )
    return ergebnis.value_paths


# --- I7: Monotonie im Zielbetrag (exakt) -----------------------------------


@given(
    niedrig=st.floats(min_value=1_000.0, max_value=2_000_000.0),
    aufschlag=st.floats(min_value=0.01, max_value=1_000_000.0),
)
def test_i7_zielbetrag_monoton_fallend(
    wertverlauf: FloatArray, niedrig: float, aufschlag: float
) -> None:
    """I7, erste Richtung: Ein höherer Zielbetrag senkt die Erfolgswahrscheinlichkeit nie.

    Diese Richtung gilt **exakt und ohne Toleranz**: Für eine feste Pfadschar
    ist ``P(V ≥ t)`` eine fallende Treppenfunktion in ``t``. Ein Gegenbeispiel
    wäre kein statistischer Ausreißer, sondern ein Rechenfehler.

    Args:
        wertverlauf: Wertmatrix der gemeinsamen Pfadschar.
        niedrig: Niedrigerer Zielbetrag.
        aufschlag: Aufschlag auf den höheren Zielbetrag.
    """
    endwerte = wertverlauf[:, -1]
    p_niedrig = float(np.mean(endwerte >= niedrig))
    p_hoch = float(np.mean(endwerte >= niedrig + aufschlag))

    assert p_hoch <= p_niedrig


@given(zielbetrag=st.floats(min_value=1_000.0, max_value=3_000_000.0))
def test_i7_zielbetrag_grenzwerte(wertverlauf: FloatArray, zielbetrag: float) -> None:
    """Die Erfolgswahrscheinlichkeit bleibt für jeden Zielbetrag im Einheitsintervall.

    Args:
        wertverlauf: Wertmatrix der gemeinsamen Pfadschar.
        zielbetrag: Geprüfter Zielbetrag.
    """
    p = float(np.mean(wertverlauf[:, -1] >= zielbetrag))
    assert 0.0 <= p <= 1.0


# --- I7: Monotonie im Horizont (im Rahmen des Rauschens) -------------------


@given(zielbetrag=st.sampled_from([50_000.0, 100_000.0, 200_000.0, 300_000.0, 500_000.0]))
def test_i7_horizont_monoton_im_rahmen_des_rauschens(
    wertverlauf: FloatArray, zielbetrag: float
) -> None:
    """I7, zweite Richtung: steigend im Horizont — aber nur bis auf Schätzfehler.

    Anders als die Zielbetragsrichtung gilt diese Richtung *nicht* exakt. Ein
    Wertpfad kann in einem einzelnen Monat unter den Zielbetrag zurückfallen;
    über 360 Monate traten bei 2 000 Pfaden 15 Rückgänge von Monat zu Monat
    auf. Sie sind kein Fehler, sondern der Marktverlauf selbst: Die Sparrate
    hebt den Wert, ein Kursrückgang senkt ihn, und in einzelnen Monaten
    überwiegt der zweite Effekt.

    Der Test formuliert die Invariante deshalb so, wie sie tatsächlich gilt:
    Jeder Rückgang bleibt unterhalb **eines** Standardfehlers der Schätzung.
    Gemessen wurden höchstens 0,25 Standardfehler; die Schranke lässt Luft für
    andere Ziehungen, ohne beliebig zu werden. Ein systematischer Bruch der
    Invariante — etwa durch einen Vorzeichenfehler beim Zeitindex — läge um
    Größenordnungen darüber und würde erkannt.

    Args:
        wertverlauf: Wertmatrix der gemeinsamen Pfadschar.
        zielbetrag: Geprüfter Zielbetrag.
    """
    p = np.array(
        [float(np.mean(wertverlauf[:, m] >= zielbetrag)) for m in range(wertverlauf.shape[1])]
    )
    standardfehler = float(np.sqrt(np.max(p * (1.0 - p)) / wertverlauf.shape[0]))
    aenderungen = np.diff(p)

    assert float(aenderungen.min()) > -standardfehler


@given(zielbetrag=st.sampled_from([50_000.0, 100_000.0, 200_000.0, 300_000.0, 500_000.0]))
def test_i7_horizont_steigt_ueber_den_gesamten_zeitraum(
    wertverlauf: FloatArray, zielbetrag: float
) -> None:
    """Der Trend über den gesamten Horizont ist eindeutig steigend.

    Die Aussage, auf die es fachlich ankommt: Mehr Zeit hilft. Sie gilt über
    den vollen Zeitraum ohne jede Toleranz, auch wenn einzelne Monate sie
    lokal verletzen.

    Args:
        wertverlauf: Wertmatrix der gemeinsamen Pfadschar.
        zielbetrag: Geprüfter Zielbetrag.
    """
    erster = float(np.mean(wertverlauf[:, 0] >= zielbetrag))
    letzter = float(np.mean(wertverlauf[:, -1] >= zielbetrag))

    assert letzter >= erster


# --- I9: keine Empfehlung ohne bestandene Prüfung --------------------------


@given(
    erreicht=st.floats(min_value=0.0, max_value=1.0),
    gefordert=st.floats(min_value=0.5, max_value=0.99),
    puffer=st.floats(min_value=0.0, max_value=0.2),
)
def test_i9_empfehlung_nur_ueber_der_schwelle(
    erreicht: float, gefordert: float, puffer: float
) -> None:
    """I9 auf der Abbildung: Eine Empfehlung setzt das Erreichen der Schwelle voraus.

    Die Äquivalenz gilt in beide Richtungen — unter der Schwelle nie eine
    Empfehlung, über der Schwelle nie eine Sperre. Das ist die formale Fassung
    von Art. 54 Abs. 10 DelVO (EU) 2017/565.

    Args:
        erreicht: Erreichte Erfolgswahrscheinlichkeit.
        gefordert: Geforderte Mindest-Erfolgswahrscheinlichkeit.
        puffer: Breite des Pufferbands.
    """
    ausgang = classify_outcome(erreicht, gefordert, puffer)

    assert ausgang.allows_recommendation == (erreicht >= gefordert)
    assert (ausgang is ReconcileOutcome.KONFLIKT) == (erreicht < gefordert)


@given(
    erreicht=st.floats(min_value=0.0, max_value=1.0),
    gefordert=st.floats(min_value=0.5, max_value=0.99),
    puffer=st.floats(min_value=0.001, max_value=0.2),
)
def test_i9_pufferband_liegt_zwischen_den_beiden_anderen(
    erreicht: float, gefordert: float, puffer: float
) -> None:
    """Der Pufferfall tritt genau im Band oberhalb der Schwelle auf.

    Args:
        erreicht: Erreichte Erfolgswahrscheinlichkeit.
        gefordert: Geforderte Mindest-Erfolgswahrscheinlichkeit.
        puffer: Breite des Pufferbands.
    """
    ausgang = classify_outcome(erreicht, gefordert, puffer)
    im_band = gefordert <= erreicht < gefordert + puffer

    assert (ausgang is ReconcileOutcome.ERREICHBAR_MIT_PUFFER) == im_band


def test_i9_konflikt_sperrt_die_empfehlung_und_belegt_die_rechtsgrundlage(
    sampler: BlockBootstrapSampler,
) -> None:
    """I9 auf der gesamten Kette: Der Konflikt sperrt und begründet die Sperre.

    Geprüft wird nicht nur, dass keine Empfehlung ergeht, sondern dass der
    Trace die Rechtsgrundlage trägt. Ein Trace muss ohne den Code genügen.

    Args:
        sampler: Ziehungsverfahren.
    """
    ergebnis = _abgleich(sampler, target_amount=Decimal("900000.00"))

    assert ergebnis.outcome is ReconcileOutcome.KONFLIKT
    assert not ergebnis.allows_recommendation
    assert ergebnis.conflict_report is not None
    assert ergebnis.conflict_report.legal_basis == CONFLICT_LEGAL_BASIS

    grundlagen = [
        ereignis.legal_basis for ereignis in ergebnis.trace if ereignis.legal_basis is not None
    ]
    assert CONFLICT_LEGAL_BASIS in grundlagen


def test_erreichbarer_fall_erzeugt_keinen_konfliktbericht(
    sampler: BlockBootstrapSampler,
) -> None:
    """Ohne Konflikt gibt es keinen Konfliktbericht — und keine Stellschrauben.

    Args:
        sampler: Ziehungsverfahren.
    """
    ergebnis = _abgleich(sampler, target_amount=Decimal("60000.00"))

    assert ergebnis.outcome is ReconcileOutcome.ERREICHBAR
    assert ergebnis.allows_recommendation
    assert ergebnis.conflict_report is None


# --- Grenzfall: Erfolgswahrscheinlichkeit exakt auf der Schwelle -----------


def test_grenzfall_exakt_auf_der_schwelle_ist_kein_konflikt() -> None:
    """Gleichheit erfüllt die Anforderung — die Schwelle ist einschließend.

    Der Fall entscheidet über die Richtung des Vergleichsoperators. Bei
    ``>`` statt ``>=`` würde ein exakt erfüllendes Ziel abgelehnt.
    """
    ausgang = classify_outcome(0.85, 0.85, 0.05)

    assert ausgang is not ReconcileOutcome.KONFLIKT
    assert ausgang is ReconcileOutcome.ERREICHBAR_MIT_PUFFER
    assert ausgang.allows_recommendation


def test_grenzfall_exakt_am_oberen_rand_des_puffers() -> None:
    """Am oberen Rand des Bands endet der Hinweis auf den geringen Abstand."""
    assert classify_outcome(0.90, 0.85, 0.05) is ReconcileOutcome.ERREICHBAR
    assert classify_outcome(0.8999, 0.85, 0.05) is ReconcileOutcome.ERREICHBAR_MIT_PUFFER


def test_grenzfall_knapp_unter_der_schwelle_ist_konflikt() -> None:
    """Ein beliebig kleiner Abstand nach unten ist bereits ein Konflikt.

    Die Schwelle wird nicht gerundet und nicht geglättet. Das ist die
    Entscheidung gegen ein stilles Nachgeben (Prinzip P5).
    """
    ausgang = classify_outcome(0.8499, 0.85, 0.05)

    assert ausgang is ReconcileOutcome.KONFLIKT
    assert not ausgang.allows_recommendation


# --- Grenzfall: Ziel bereits erreicht --------------------------------------


def test_grenzfall_ziel_bereits_erreicht(sampler: BlockBootstrapSampler) -> None:
    """Ist das Ziel schon mit dem Startkapital sicher, bleibt die Sparrate unverändert.

    Die Suche darf hier nicht iterieren: Bereits die untere Intervallgrenze
    erfüllt die Anforderung. Der geforderte Wert ist dann die Ist-Sparrate,
    die Differenz null.

    Args:
        sampler: Ziehungsverfahren.
    """
    kontext = _context(sampler)
    ziel_ = ziel(target_amount=Decimal("1000.00"), initial_capital=Decimal("20000.00"))

    ergebnis = solve_contribution(kontext, ziel_, 0.90)

    assert ergebnis.reachable
    assert ergebnis.probability >= 0.90
    assert ergebnis.value == pytest.approx(float(ziel_.monthly_contribution))


def test_grenzfall_ziel_bereits_erreicht_ganze_kette(sampler: BlockBootstrapSampler) -> None:
    """Ein bereits erreichtes Ziel führt zur Empfehlung ohne Vorbehalt.

    Args:
        sampler: Ziehungsverfahren.
    """
    ergebnis = _abgleich(sampler, target_amount=Decimal("30000.00"))

    assert ergebnis.outcome is ReconcileOutcome.ERREICHBAR
    assert ergebnis.achieved_probability > 0.99
    assert ergebnis.margin > 0.0


# --- Grenzfall: Ziel unmöglich ---------------------------------------------


def _unmoegliches_ziel(sampler: BlockBootstrapSampler) -> ConflictReport:
    """Rechnet einen Fall, in dem keine Stellschraube den Konflikt löst.

    Gewählt ist ein Ausbildungsziel: Sein Termin ist nicht verschiebbar und
    sein Betrag nicht reduzierbar, weil Studienbeginn und Studienkosten nicht
    zur Disposition stehen. Damit scheiden S2 und S3 fachlich aus, und der
    Betrag ist so hoch, dass auch S1 an der Suchgrenze scheitert — nicht bloß
    an der freien Sparquote des Haushalts.

    Args:
        sampler: Ziehungsverfahren.

    Returns:
        Der Konfliktbericht des unmöglichen Falls.
    """
    kunde_ = kunde()
    ziel_ = ziel(
        goal_id="Z-UNMOEGLICH",
        goal_type=GoalType.AUSBILDUNG,
        name="Studium",
        target_date=date(2031, 7, 1),
        target_amount=Decimal("50000000.00"),
        initial_capital=Decimal("1000.00"),
        monthly_contribution=Decimal("300.00"),
    )
    recorder = _recorder(kunde_.client_id, ziel_.goal_id)
    profil = classify(kunde_, ziel_, POLICIES, recorder)
    ergebnis = reconcile(
        kunde_,
        ziel_,
        profil,
        POLICIES,
        sampler,
        recorder,
        asset_keys=ASSET_KEYS,
        paths=TEST_PATHS,
        seed=TEST_SEED,
    )
    assert ergebnis.conflict_report is not None
    return ergebnis.conflict_report


def test_grenzfall_ziel_unmoeglich_keine_stellschraube_wird_angeboten(
    sampler: BlockBootstrapSampler,
) -> None:
    """Löst keine Stellschraube den Konflikt, wird auch keine als Vorschlag ausgegeben.

    Das ist der Kern der ehrlichen Diagnose: Das System bietet dann nichts an,
    statt den unrealistischen Wert als Vorschlag auszuweisen.

    Args:
        sampler: Ziehungsverfahren.
    """
    bericht = _unmoegliches_ziel(sampler)

    assert all_levers_infeasible(bericht)
    assert not bericht.has_way_out
    assert bericht.offered_levers == ()
    assert len(bericht.levers) == 3


def test_grenzfall_ziel_unmoeglich_jede_stellschraube_hat_eine_begruendung(
    sampler: BlockBootstrapSampler,
) -> None:
    """Auch die abgelehnte Stellschraube trägt Zahl und Begründung.

    Nicht umsetzbar heißt nicht: nicht ausgewiesen. Der Bericht nennt den
    erforderlichen Wert und den Grund, warum er ausscheidet.

    Args:
        sampler: Ziehungsverfahren.
    """
    bericht = _unmoegliches_ziel(sampler)

    for hebel in bericht.levers:
        assert hebel.feasibility is not Feasibility.UMSETZBAR
        assert hebel.feasibility_note != ""
        assert not hebel.is_offered


def test_grenzfall_ziel_unmoeglich_sparrate_erreicht_die_suchgrenze(
    sampler: BlockBootstrapSampler,
) -> None:
    """S1 scheitert an der Suchgrenze — das ist ein Befund, kein Fehler.

    Die Suche gibt den Randwert mit ``reachable=False`` zurück, statt eine
    Ausnahme zu werfen. Nur die *Nichtkonvergenz* ist ein Fehler.

    Der Fall muss rechnerisch wirklich unerreichbar sein. Bei 60 Monaten und
    der Suchobergrenze von 100 000 € je Monat sind rund 6 Mio. € einzahlbar;
    ein Ziel von 5 Mio. € wäre also erreichbar. Der Zielbetrag liegt deshalb
    eine Größenordnung darüber.

    Args:
        sampler: Ziehungsverfahren.
    """
    kontext = _context(sampler)
    ziel_ = ziel(
        target_date=date(2031, 7, 1),
        target_amount=Decimal("50000000.00"),
        initial_capital=Decimal("1000.00"),
        monthly_contribution=Decimal("300.00"),
    )

    ergebnis = solve_contribution(kontext, ziel_, 0.85)

    assert not ergebnis.reachable
    assert ergebnis.value == pytest.approx(MAX_CONTRIBUTION)


# --- Machbarkeitsprüfung gegen die freie Sparquote -------------------------


def test_sparrate_ueber_der_freien_sparquote_gilt_als_nicht_umsetzbar(
    sampler: BlockBootstrapSampler,
) -> None:
    """Übersteigt S1 die freie Sparquote, wird sie nicht als Vorschlag verkauft.

    Der Haushalt hat 600 € frei. Die Suche liefert einen höheren Wert; die
    Bewertung muss ihn zurückweisen, obwohl er rechnerisch das Ziel löst.

    Args:
        sampler: Ziehungsverfahren.
    """
    kontext = _context(sampler)
    kunde_ = kunde(household=haushalt(free_savings_amount=Decimal("520.00")))
    ziel_ = ziel(target_amount=Decimal("900000.00"))

    hebel = build_levers(kontext, kunde_, ziel_, 0.90)
    sparrate = next(h for h in hebel if h.lever_id is LeverId.SPARRATE)

    assert sparrate.required_value > kunde_.household.free_savings_amount
    assert sparrate.feasibility is not Feasibility.UMSETZBAR
    assert not sparrate.is_offered


def test_sparrate_innerhalb_der_freien_sparquote_wird_angeboten(
    sampler: BlockBootstrapSampler,
) -> None:
    """Bleibt S1 im Rahmen der freien Sparquote, ist sie ein echter Vorschlag.

    Args:
        sampler: Ziehungsverfahren.
    """
    kontext = _context(sampler)
    kunde_ = kunde(household=haushalt(free_savings_amount=Decimal("1500.00")))
    ziel_ = ziel(target_amount=Decimal("300000.00"))

    hebel = build_levers(kontext, kunde_, ziel_, 0.90)
    sparrate = next(h for h in hebel if h.lever_id is LeverId.SPARRATE)

    assert sparrate.required_value <= kunde_.household.free_savings_amount
    assert sparrate.feasibility is Feasibility.UMSETZBAR
    assert sparrate.is_offered


def test_nicht_verschiebbares_ziel_bietet_den_horizont_nicht_an(
    sampler: BlockBootstrapSampler,
) -> None:
    """Ein fester Termin schließt S2 aus, auch wenn die Rechnung aufginge.

    Die Machbarkeit ist nicht nur eine Frage des Geldes, sondern auch des
    Zieltyps. Der Liquiditätspuffer ist weder verschiebbar noch reduzierbar.

    Args:
        sampler: Ziehungsverfahren.
    """
    kontext = _context(sampler)
    ziel_ = ziel(
        goal_type=GoalType.LIQUIDITAETSPUFFER,
        name="Notreserve",
        target_date=date(2028, 7, 1),
        target_amount=Decimal("80000.00"),
        initial_capital=Decimal("5000.00"),
        monthly_contribution=Decimal("500.00"),
    )

    hebel = build_levers(kontext, kunde(), ziel_, 0.95)
    horizont = next(h for h in hebel if h.lever_id is LeverId.HORIZONT)
    betrag = next(h for h in hebel if h.lever_id is LeverId.ZIELBETRAG)

    assert horizont.feasibility is Feasibility.NICHT_UMSETZBAR
    assert betrag.feasibility is Feasibility.NICHT_UMSETZBAR


# --- Bisektion: Konvergenz, Abbruch, Fehler --------------------------------


def test_bisektion_findet_die_nullstelle_einer_bekannten_funktion() -> None:
    """Gegenprobe an einer analytisch bekannten Funktion.

    Gesucht ist die Stelle, an der ``x/100`` den Wert ``0,85`` erreicht, also
    ``x = 85``. Die Bisektion muss sie bis auf die Toleranz treffen.
    """
    ergebnis = bisect_increasing(lambda x: x / 100.0, 0.0, 1000.0, 0.85, tolerance=0.01)

    assert ergebnis.reachable
    assert ergebnis.value == pytest.approx(85.0, abs=0.01)


def test_bisektion_fallend_spiegelt_korrekt() -> None:
    """Die fallende Variante findet den größten haltenden Wert.

    ``1 − x/100`` hält ``0,25`` bis einschließlich ``x = 75``.
    """
    ergebnis = bisect_decreasing(lambda x: 1.0 - x / 100.0, 0.0, 100.0, 0.25, tolerance=0.01)

    assert ergebnis.reachable
    assert ergebnis.value == pytest.approx(75.0, abs=0.01)


def test_bisektion_meldet_unerreichbarkeit_statt_zu_scheitern() -> None:
    """Wird die Schwelle am oberen Rand nicht erreicht, ist das ein Befund.

    Der Rückgabewert ist der Randwert mit ``reachable=False`` — keine
    Ausnahme, denn die Rechnung ist korrekt; nur das Ziel liegt außerhalb.
    """
    ergebnis = bisect_increasing(lambda x: x / 1000.0, 0.0, 100.0, 0.85, tolerance=0.01)

    assert not ergebnis.reachable
    assert ergebnis.value == pytest.approx(100.0)


def test_bisektion_erzwingt_konvergenz_statt_stiller_rueckgabe() -> None:
    """Nichtkonvergenz führt zu ``SolverError``, nicht zu einem Zwischenwert.

    Mit zu wenigen Iterationen für die geforderte Toleranz muss die Suche
    abbrechen. Ein stiller Zwischenwert wäre die gefährlichere Variante: Er
    sähe aus wie ein Ergebnis.
    """
    with pytest.raises(SolverError, match="Toleranz"):
        bisect_increasing(lambda x: x / 1000.0, 0.0, 1000.0, 0.85, tolerance=1e-9, max_iterations=3)


def test_bisektion_weist_unzulaessiges_intervall_zurueck() -> None:
    """Ein leeres oder verkehrtes Intervall ist ein Fehler, kein Sonderfall."""
    with pytest.raises(SolverError, match="Suchintervall"):
        bisect_increasing(lambda x: x, 10.0, 10.0, 0.5, tolerance=0.1)


def test_bisektion_ist_bei_fixiertem_seed_reproduzierbar(
    sampler: BlockBootstrapSampler,
) -> None:
    """Zwei Läufe derselben Suche liefern denselben Wert (Prinzip P1).

    Das ist die Probe auf den fixierten Seed: Ohne ihn liefe die Bisektion auf
    einer bei jeder Auswertung neu gezogenen Zielfunktion und käme zu
    wechselnden Ergebnissen.

    Args:
        sampler: Ziehungsverfahren.
    """
    kontext = _context(sampler)
    ziel_ = ziel(target_amount=Decimal("400000.00"))

    erst = solve_contribution(kontext, ziel_, 0.85)
    zweit = solve_contribution(kontext, ziel_, 0.85)

    assert erst.value == zweit.value
    assert erst.probability == zweit.probability
    assert erst.iterations == zweit.iterations


# --- Die drei Stellschrauben -----------------------------------------------


def test_s1_erreicht_die_geforderte_wahrscheinlichkeit(
    sampler: BlockBootstrapSampler,
) -> None:
    """Die gefundene Sparrate hält die Anforderung tatsächlich ein.

    Args:
        sampler: Ziehungsverfahren.
    """
    kontext = _context(sampler)
    ziel_ = ziel(target_amount=Decimal("400000.00"))

    ergebnis = solve_contribution(kontext, ziel_, 0.85)

    assert ergebnis.reachable
    assert ergebnis.probability >= 0.85
    assert ergebnis.value > float(ziel_.monthly_contribution)


def test_s2_verlaengert_den_horizont_und_haelt_die_schwelle(
    sampler: BlockBootstrapSampler,
) -> None:
    """Der gefundene Horizont ist ganzzahlig, länger als der Ist-Horizont und ausreichend.

    Args:
        sampler: Ziehungsverfahren.
    """
    kontext = _context(sampler)
    ziel_ = ziel(target_amount=Decimal("400000.00"))

    ergebnis = solve_horizon(kontext, ziel_, 0.85)

    assert ergebnis.reachable
    assert ergebnis.value == float(int(ergebnis.value))
    assert ergebnis.value > ziel_.horizon_months
    assert ergebnis.value <= MAX_HORIZON_MONTHS
    assert ergebnis.probability >= 0.85


def test_s3_geschlossene_loesung_stimmt_mit_der_bisektion_ueberein(
    sampler: BlockBootstrapSampler,
) -> None:
    """Gegenprobe: Quantil und Bisektion liefern denselben Zielbetrag.

    S3 wird im Betrieb über das empirische Quantil gelöst, weil der Zielbetrag
    die Simulation nicht beeinflusst — er ist nur die Zählschwelle. Die
    Bisektion dient allein als unabhängige Prüfung dieser Abkürzung.

    Args:
        sampler: Ziehungsverfahren.
    """
    kontext = _context(sampler)
    ziel_ = ziel(target_amount=Decimal("400000.00"))

    geschlossen = solve_target(kontext, ziel_, 0.85)
    iterativ = solve_target_by_bisection(kontext, ziel_, 0.85)

    assert geschlossen.iterations == 0
    assert iterativ.iterations > 0
    assert geschlossen.value == pytest.approx(iterativ.value, abs=2.0)
    assert geschlossen.probability >= 0.85


def test_s3_liefert_bei_konflikt_einen_niedrigeren_zielbetrag(
    sampler: BlockBootstrapSampler,
) -> None:
    """Im Konfliktfall liegt der erreichbare Betrag unter dem gewünschten.

    Args:
        sampler: Ziehungsverfahren.
    """
    kontext = _context(sampler)
    ziel_ = ziel(target_amount=Decimal("900000.00"))

    ergebnis = solve_target(kontext, ziel_, 0.90)

    assert ergebnis.value < float(ziel_.target_amount)


def test_alle_drei_stellschrauben_werden_geliefert(sampler: BlockBootstrapSampler) -> None:
    """Der Bericht enthält immer alle drei Stellschrauben in fester Reihenfolge.

    Args:
        sampler: Ziehungsverfahren.
    """
    kontext = _context(sampler)
    hebel = build_levers(kontext, kunde(), ziel(target_amount=Decimal("500000.00")), 0.90)

    assert tuple(h.lever_id for h in hebel) == (
        LeverId.SPARRATE,
        LeverId.HORIZONT,
        LeverId.ZIELBETRAG,
    )
    for einzeln in hebel:
        assert einzeln.name != ""
        assert einzeln.unit != ""
        assert einzeln.method != ""


# --- Diagnose, Trace und Formatierung --------------------------------------


def test_diagnose_besteht_aus_stichpunkten_mit_zahlen(sampler: BlockBootstrapSampler) -> None:
    """Die Diagnose nennt die bindende Achse und die beiden Wahrscheinlichkeiten.

    Args:
        sampler: Ziehungsverfahren.
    """
    ergebnis = _abgleich(sampler, target_amount=Decimal("900000.00"))
    assert ergebnis.conflict_report is not None
    diagnose = ergebnis.conflict_report.diagnosis

    assert len(diagnose) >= 3
    assert all(punkt.strip() != "" for punkt in diagnose)
    assert any(CONFLICT_LEGAL_BASIS in punkt for punkt in diagnose)


def test_diagnose_verwendet_das_deutsche_dezimalkomma(sampler: BlockBootstrapSampler) -> None:
    """Jede Zahl der Diagnose folgt der deutschen Schreibweise.

    Der Test hat einen echten Fehler gefunden: Die Diagnose gab Beträge als
    ``900000.00 €`` aus. Er prüft deshalb nicht bloß auf das Fehlen eines
    Punktes — der Tausenderpunkt ist ja richtig —, sondern jede Zahl gegen die
    vollständige Grammatik.

    Args:
        sampler: Ziehungsverfahren.
    """
    ergebnis = _abgleich(sampler, target_amount=Decimal("900000.00"))
    assert ergebnis.conflict_report is not None

    for punkt in ergebnis.conflict_report.diagnosis:
        ohne_versionen = VERSION_MUSTER.sub("", punkt)
        for zahl in ZAHL_MUSTER.findall(ohne_versionen):
            bereinigt = zahl.rstrip(".,")
            if not any(zeichen in bereinigt for zeichen in ".,"):
                continue  # Kennungen und blanke Ganzzahlen tragen kein Trennzeichen.
            assert DEUTSCHE_ZAHL.match(bereinigt), f"{zahl!r} in {punkt!r}"


def test_format_probability_liefert_prozent_mit_komma() -> None:
    """Wahrscheinlichkeiten erscheinen als Prozentwert mit Dezimalkomma."""
    ausgabe = format_probability(0.8456)

    assert ausgabe == "84,56 %"


@pytest.mark.parametrize(
    ("wert", "erwartet"),
    [
        (900_000, f"900{TRENNER}000,00 €"),
        (1_234.5, f"1{TRENNER}234,50 €"),
        (0, "0,00 €"),
        (-1_500.25, f"-1{TRENNER}500,25 €"),
    ],
)
def test_euro_formatierung(wert: float, erwartet: str) -> None:
    """Beträge tragen Tausendertrennzeichen und Dezimalkomma.

    Args:
        wert: Zu formatierender Betrag.
        erwartet: Erwartete Ausgabe.
    """
    assert format_de.euro(wert) == erwartet


def test_tausendertrennzeichen_ist_das_geschuetzte_leerzeichen() -> None:
    """Getrennt wird nach DIN 5008 mit geschütztem Leerzeichen, nicht mit Punkt.

    Der Punkt wäre für sich genommen ebenfalls zulässig; entscheidend ist, dass
    das gesamte Projekt dieselbe Wahl trifft. Abbildungen, Statistiktabellen und
    Konvergenzblöcke verwenden bereits das Leerzeichen.
    """
    assert format_de.THOUSANDS_SEPARATOR == "\xa0"
    assert "." not in format_de.euro(1_000_000)


@pytest.mark.parametrize(
    ("wert", "erwartet"),
    [(0.8456, "84,56 %"), (1.0, "100,00 %"), (0.0, "0,00 %"), (0.005, "0,50 %")],
)
def test_prozent_formatierung(wert: float, erwartet: str) -> None:
    """Anteile erscheinen als Prozentangabe ohne Tausenderpunkt.

    Args:
        wert: Zu formatierender Anteil.
        erwartet: Erwartete Ausgabe.
    """
    assert format_de.percent(wert) == erwartet


def test_prozentpunkte_sind_von_prozent_unterschieden() -> None:
    """Prozent und Prozentpunkte tragen verschiedene Einheiten.

    Die Verwechslung ist der häufigste Zahlenfehler in Finanzberichten; die
    Trennung in zwei Funktionen macht sie an der Aufrufstelle sichtbar.
    """
    assert format_de.percent(0.0544) == "5,44 %"
    assert format_de.percentage_points(0.0544) == "5,44 Prozentpunkte"


def test_formatierung_haelt_der_eigenen_grammatikpruefung_stand() -> None:
    """Die Ausgaben des Formatierers bestehen die Grammatikprüfung der Tests.

    Verbindet Formatierer und Prüfmuster: Wäre eines von beiden falsch,
    widersprächen sie sich hier.
    """
    for wert in (0, 1, 999, 1_000, 1_234_567, 48_227):
        assert DEUTSCHE_ZAHL.match(format_de.integer(wert))
        assert DEUTSCHE_ZAHL.match(format_de.euro(wert).removesuffix(" €"))


def test_abgleich_ist_deterministisch(sampler: BlockBootstrapSampler) -> None:
    """Gleiches Profil, gleicher Seed, gleiche Policy ⇒ gleicher Trace-Hash (I4).

    Args:
        sampler: Ziehungsverfahren.
    """
    erst = _abgleich(sampler, target_amount=Decimal("900000.00"))
    zweit = _abgleich(sampler, target_amount=Decimal("900000.00"))

    assert erst.trace_hash == zweit.trace_hash
    assert erst.achieved_probability == zweit.achieved_probability


def test_trace_belegt_jede_stellschraube(sampler: BlockBootstrapSampler) -> None:
    """Jede Stellschraube hinterlässt ein eigenes Ereignis im Trace.

    Args:
        sampler: Ziehungsverfahren.
    """
    ergebnis = _abgleich(sampler, target_amount=Decimal("900000.00"))
    hebel_ereignisse = [e for e in ergebnis.trace if e.rule_id == "REC-LEVER"]

    assert len(hebel_ereignisse) == 3


def test_solve_levers_abschaltbar(sampler: BlockBootstrapSampler) -> None:
    """Ohne Stellschraubensuche entsteht ein Bericht ohne Hebel.

    Die Abschaltung existiert, weil die Suche mehrere Simulationen kostet. Sie
    darf die Sperre der Empfehlung nicht aufheben.

    Args:
        sampler: Ziehungsverfahren.
    """
    kunde_ = kunde()
    ziel_ = ziel(target_amount=Decimal("900000.00"))
    recorder = _recorder(kunde_.client_id, ziel_.goal_id)
    profil = classify(kunde_, ziel_, POLICIES, recorder)

    ergebnis = reconcile(
        kunde_,
        ziel_,
        profil,
        POLICIES,
        sampler,
        recorder,
        asset_keys=ASSET_KEYS,
        paths=TEST_PATHS,
        seed=TEST_SEED,
        solve_levers=False,
    )

    assert ergebnis.outcome is ReconcileOutcome.KONFLIKT
    assert not ergebnis.allows_recommendation
    assert ergebnis.conflict_report is not None
    assert ergebnis.conflict_report.levers == ()


def test_konfliktbericht_kennt_den_abstand_zur_schwelle(
    sampler: BlockBootstrapSampler,
) -> None:
    """Der Bericht weist die Lücke zwischen gefordert und erreicht aus.

    Args:
        sampler: Ziehungsverfahren.
    """
    ergebnis = _abgleich(sampler, target_amount=Decimal("900000.00"))
    assert ergebnis.conflict_report is not None
    bericht = ergebnis.conflict_report

    assert bericht.gap > 0.0
    assert bericht.gap == pytest.approx(bericht.required_probability - bericht.achieved_probability)
    assert ergebnis.margin < 0.0


def test_build_diagnosis_nennt_die_bindende_achse() -> None:
    """Die Diagnose benennt, welche Achse die Risikoklasse bindet."""
    kunde_ = kunde()
    ziel_ = ziel(target_amount=Decimal("900000.00"))
    recorder = _recorder()
    profil: RiskProfile = classify(kunde_, ziel_, POLICIES, recorder)

    diagnose = build_diagnosis(ziel_, profil, 0.90, 0.6, 50_000.0)

    assert any(profil.binding_axis.value in punkt for punkt in diagnose)


# --- Das Ergebnisobjekt selbst ---------------------------------------------


def test_bisection_result_ist_unveraenderlich() -> None:
    """Ergebnisobjekte sind eingefroren — kein nachträgliches Verstellen."""
    ergebnis = BisectionResult(1.0, 0.9, 3, reachable=True)

    with pytest.raises(AttributeError):
        ergebnis.value = 2.0  # type: ignore[misc]


def test_zusammenfassung_enthaelt_die_kernzahlen(sampler: BlockBootstrapSampler) -> None:
    """``as_summary`` liefert die Kennzahlen für die Weiterverarbeitung.

    Args:
        sampler: Ziehungsverfahren.
    """
    ergebnis = _abgleich(sampler, target_amount=Decimal("900000.00"))
    zusammenfassung = ergebnis.as_summary()

    for schluessel in ("ausgang", "gefordert", "erreicht", "risikoklasse", "trace_hash"):
        assert schluessel in zusammenfassung


def test_stichtag_der_builder_bleibt_der_projektstichtag() -> None:
    """Die Testbasis rechnet auf dem Stichtag des Projekts."""
    assert date(2026, 7, 26) == STICHTAG


# --- Das Faktenblatt -------------------------------------------------------


@pytest.fixture(scope="module")
def referenzfall() -> Reconciliation:
    """Rechnet den Referenz-Konfliktfall einmal je Modul.

    Der Lauf kostet rund fünf Sekunden, weil er mit den vollen 10 000 Pfaden des
    Regelwerks rechnet. Er wird deshalb geteilt.

    Returns:
        Das Ergebnis des Referenzfalls.
    """
    return reference_conflict_case()


def test_referenzfall_ist_und_bleibt_ein_konflikt(referenzfall: Reconciliation) -> None:
    """Der Referenzfall des Faktenblatts muss ein Konflikt sein.

    Das ist keine Formalie: Änderten sich Snapshot, Kostenannahmen oder
    Regelwerk so, dass der Fall plötzlich erreichbar wäre, hätte das Faktenblatt
    keinen Gegenstand mehr. Der Test macht diesen Bruch sichtbar, statt ihn beim
    nächsten ``make factsheet`` als Fehlermeldung erscheinen zu lassen.

    Args:
        referenzfall: Das Ergebnis des Referenzfalls.
    """
    assert referenzfall.outcome is ReconcileOutcome.KONFLIKT
    assert referenzfall.conflict_report is not None
    assert len(referenzfall.conflict_report.levers) == 3


def test_faktenblatt_enthaelt_alle_abschnitte(referenzfall: Reconciliation) -> None:
    """Das Faktenblatt trägt alle sieben Abschnitte.

    Args:
        referenzfall: Das Ergebnis des Referenzfalls.
    """
    blatt = render_factsheet(referenzfall)

    for ueberschrift in (
        "## 1. Ausgangslage",
        "## 2. Risikoprofil",
        "## 3. Zielabgleich",
        "## 4. Befund",
        "## 5. Die drei Stellschrauben",
        "## 6. Einordnung",
        "## 7. Reproduktion",
    ):
        assert ueberschrift in blatt


def test_faktenblatt_nennt_die_rechtsgrundlage(referenzfall: Reconciliation) -> None:
    """Die Sperre der Empfehlung ist im Faktenblatt begründet.

    Args:
        referenzfall: Das Ergebnis des Referenzfalls.
    """
    blatt = render_factsheet(referenzfall)

    assert CONFLICT_LEGAL_BASIS in blatt
    assert referenzfall.trace_hash[:16] in blatt


def test_faktenblatt_schreibt_jede_zahl_deutsch(referenzfall: Reconciliation) -> None:
    """Keine Zahl des Faktenblatts trägt einen englischen Dezimalpunkt.

    Geprüft wird das gesamte Dokument, nicht nur die Diagnose — es ist die
    Fassung, die in das Kundendokument übernommen wird.

    Args:
        referenzfall: Das Ergebnis des Referenzfalls.
    """
    blatt = render_factsheet(referenzfall)
    for zeile in blatt.splitlines():
        if _ist_ausgenommen(zeile):
            continue
        bereinigte_zeile = zeile
        for muster in (VERSION_MUSTER, DATUM_MUSTER, VERWEIS_MUSTER):
            bereinigte_zeile = muster.sub("", bereinigte_zeile)
        for zahl in ZAHL_MUSTER.findall(bereinigte_zeile):
            bereinigt = zahl.rstrip(".,")
            if not any(zeichen in bereinigt for zeichen in ".,"):
                continue
            assert DEUTSCHE_ZAHL.match(bereinigt), f"{zahl!r} in {zeile!r}"


def test_faktenblatt_ist_reproduzierbar(referenzfall: Reconciliation) -> None:
    """Zwei Darstellungen desselben Ergebnisses sind zeichengleich.

    Args:
        referenzfall: Das Ergebnis des Referenzfalls.
    """
    assert render_factsheet(referenzfall) == render_factsheet(referenzfall)


def test_faktenblatt_verweigert_den_erreichbaren_fall(
    sampler: BlockBootstrapSampler,
) -> None:
    """Ohne Konflikt gibt es kein Faktenblatt — mit klarer Fehlermeldung.

    Ein Faktenblatt über einen Fall ohne Konflikt wäre irreführend. Der Fehler
    ist deshalb laut und nicht still (Prinzip P5).

    Args:
        sampler: Ziehungsverfahren.
    """
    erreichbar = _abgleich(sampler, target_amount=Decimal("60000.00"))

    with pytest.raises(ValueError, match="nicht im Konflikt"):
        render_factsheet(erreichbar)


def test_faktenblatt_kommandozeile_schreibt_die_datei(tmp_path: Path) -> None:
    """``make factsheet`` schreibt das Blatt an den angegebenen Ort.

    Args:
        tmp_path: Temporäres Verzeichnis.
    """
    ziel_datei = tmp_path / "unter" / "konfliktfall.md"

    rueckgabe = factsheet_main(["--write", str(ziel_datei)])

    assert rueckgabe == 0
    assert ziel_datei.exists()
    assert "## 5. Die drei Stellschrauben" in ziel_datei.read_text(encoding="utf-8")
