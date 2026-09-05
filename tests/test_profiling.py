"""Tests der Risikoprofilierung.

Geprüft wird das fachliche Verhalten der drei Achsen und ihrer Zusammenführung:

* Das K.-o.-Kriterium wirkt unabhängig von allen anderen Kriterien.
* Verlustszenarien werden in Euro dargestellt und in dieser Form protokolliert.
* Widersprüche werden ausgewiesen und **nicht** in die Punktzahl eingerechnet.
* Kenntnisse deckeln, sie heben nie an.
* Zusammengeführt wird über das Minimum, niemals über einen Mittelwert.

Die Invarianten I1 bis I3 sind gesondert in ``test_invariants.py`` geprüft.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from candor.data.policy_loader import load_policy_set
from candor.domain.client import (
    InstrumentCategory,
    InstrumentExperience,
    KnowledgeLevel,
    RiskAttitude,
)
from candor.domain.errors import IncompleteProfileError, PolicyError
from candor.domain.policy import PolicySet
from candor.domain.result import Axis
from candor.engine.glidepath import apply_glidepath, max_class_for_remaining_months
from candor.profiling.capacity import assess_capacity
from candor.profiling.classify import classify
from candor.profiling.knowledge import assess_knowledge, knowledge_rank
from candor.profiling.scoring import planned_investment
from candor.profiling.tolerance import assess_tolerance, find_inconsistencies, scenario_amounts
from candor.trace.recorder import TraceMetadata, TraceRecorder
from tests.builders import (
    ANTWORTEN_HOCH,
    ANTWORTEN_MITTE,
    ANTWORTEN_NIEDRIG,
    haushalt,
    kenntnisse,
    kenntnisse_fuer_klasse,
    kunde,
    ziel,
)


@pytest.fixture(scope="module")
def policies() -> PolicySet:
    """Lädt das ausgelieferte Regelwerk einmal je Modul.

    Returns:
        Das geprüfte Regelwerk.
    """
    return load_policy_set()


def recorder(policies: PolicySet) -> TraceRecorder:
    """Baut einen leeren Recorder für einen Testlauf.

    Args:
        policies: Regelwerk, dessen Versionen in den Kopf wandern.

    Returns:
        Der Recorder.
    """
    return TraceRecorder(
        TraceMetadata(
            client_id="K-TEST",
            goal_id="Z-TEST",
            policy_versions=dict(policies.versions),
            package_version="0.1.0",
        )
    )


def regel_ids(rec: TraceRecorder) -> list[str]:
    """Liefert die Regel-Kennungen der aufgezeichneten Ereignisse.

    Args:
        rec: Der Recorder.

    Returns:
        Die Kennungen in Aufzeichnungsreihenfolge.
    """
    return [ereignis.rule_id for ereignis in rec.events]


# --- Risikotragfähigkeit ---------------------------------------------------


def test_jedes_kriterium_schreibt_ein_trace_ereignis(policies: PolicySet) -> None:
    """Alle sechs Kriterien, das K.-o. und die Klassenbildung sind protokolliert.

    Args:
        policies: Das geladene Regelwerk.
    """
    rec = recorder(policies)

    assess_capacity(kunde(), ziel(), policies.capacity, rec)

    assert regel_ids(rec) == [
        "CAP-KO-LIQ",
        "CAP-LIQ",
        "CAP-INC",
        "CAP-SAV",
        "CAP-HOR",
        "CAP-DEBT",
        "CAP-CONC",
        "CAP-CLASS",
    ]


def test_grundfall_ergibt_erwartete_punktzahl(policies: PolicySet) -> None:
    """Der Grundfall ist von Hand nachgerechnet und ergibt 72,00 Punkte.

    Rechnung je Kriterium (Punkte · Gewicht):

    * Liquiditätsreserve 4,0 Monatsausgaben  -> Band bis 6,0    ->  70 · 0,25 = 17,50
    * unbefristet angestellt                 -> Ausprägung      ->  85 · 0,20 = 17,00
    * freie Sparquote 600/4200 = 0,1429      -> Band bis 0,20   ->  70 · 0,20 = 14,00
    * Horizont 239 Monate                    -> Band bis 240    ->  90 · 0,15 = 13,50
    * Kapitaldienst 300/4200 = 0,0714        -> Band bis 0,10   ->  80 · 0,10 =  8,00
    * Klumpenrisiko 139.500/164.500 = 0,8480 -> Band bis 0,95   ->  20 · 0,10 =  2,00

    Summe 72,00 -> Klasse 4 (Schwelle 65).

    Der Horizont liegt mit 239 Monaten knapp **unter** der Bandgrenze von 240 und
    erhält deshalb 90 statt 100 Punkte. Der Fall ist bewusst so stehen geblieben:
    Er zeigt, dass die Bandgrenzen einschließlich zu lesen sind.

    Args:
        policies: Das geladene Regelwerk.
    """
    ergebnis = assess_capacity(kunde(), ziel(), policies.capacity, recorder(policies))

    einzeln = {eintrag.criterion: eintrag.weighted_points for eintrag in ergebnis.scores}
    assert einzeln == {
        "liquiditaetsreserve": Decimal("17.50"),
        "einkommensstabilitaet": Decimal("17.00"),
        "freie_sparquote": Decimal("14.00"),
        "anlagehorizont": Decimal("13.50"),
        "verschuldung": Decimal("8.00"),
        "vermoegensanteil": Decimal("2.00"),
    }
    assert ergebnis.total_points == Decimal("72.00")
    assert ergebnis.risk_class == 4
    assert not ergebnis.ko_triggered


def test_ko_kriterium_setzt_klasse_eins_unabhaengig_von_allem(policies: PolicySet) -> None:
    """Unter der Mindestreserve gilt Klasse 1, auch bei sonst besten Werten.

    Args:
        policies: Das geladene Regelwerk.
    """
    bestfall = haushalt(
        liquidity_reserve_months=Decimal("2.9"),
        monthly_debt_service=Decimal("0.00"),
        free_savings_amount=Decimal("1800.00"),
        other_assets=Decimal("5000000.00"),
    )
    ergebnis = assess_capacity(
        kunde(household=bestfall), ziel(), policies.capacity, recorder(policies)
    )

    assert ergebnis.ko_triggered
    assert ergebnis.risk_class == 1
    assert ergebnis.ko_reason is not None
    # Die uebrigen Kriterien werden trotzdem vollstaendig ausgewertet.
    assert len(ergebnis.scores) == 6
    assert ergebnis.total_points > Decimal("50")


def test_ko_greift_genau_unterhalb_der_schwelle(policies: PolicySet) -> None:
    """Bei exakt der Mindestreserve greift das K.-o.-Kriterium nicht.

    Args:
        policies: Das geladene Regelwerk.
    """
    schwelle = policies.capacity.mindestreserve_monate

    knapp_drunter = assess_capacity(
        kunde(household=haushalt(liquidity_reserve_months=schwelle - Decimal("0.1"))),
        ziel(),
        policies.capacity,
        recorder(policies),
    )
    genau_drauf = assess_capacity(
        kunde(household=haushalt(liquidity_reserve_months=schwelle)),
        ziel(),
        policies.capacity,
        recorder(policies),
    )

    assert knapp_drunter.ko_triggered
    assert not genau_drauf.ko_triggered


def test_klumpenrisiko_senkt_die_tragfaehigkeit(policies: PolicySet) -> None:
    """Ein hoher Vermögensanteil verschlechtert die Bewertung.

    Args:
        policies: Das geladene Regelwerk.
    """
    breit = assess_capacity(
        kunde(household=haushalt(other_assets=Decimal("2000000.00"))),
        ziel(),
        policies.capacity,
        recorder(policies),
    )
    konzentriert = assess_capacity(
        kunde(household=haushalt(other_assets=Decimal("0.00"))),
        ziel(),
        policies.capacity,
        recorder(policies),
    )

    assert breit.total_points > konzentriert.total_points


# --- Risikobereitschaft ----------------------------------------------------


def test_verlustszenarien_werden_in_euro_dargestellt(policies: PolicySet) -> None:
    """Die Szenarien sind Eurobeträge, berechnet aus dem eingezahlten Kapital.

    Args:
        policies: Das geladene Regelwerk.
    """
    anlageziel = ziel()
    betraege = scenario_amounts(anlageziel, policies.tolerance)
    bezug = planned_investment(anlageziel)

    assert betraege["S0"] == Decimal("0.00")
    assert betraege["S3"] == (bezug * Decimal("0.20")).quantize(Decimal("0.01"))
    assert betraege["S5"] > betraege["S3"] > betraege["S1"]


def test_vorgelegter_betrag_steht_im_trace(policies: PolicySet) -> None:
    """Der dem Kunden gezeigte Eurobetrag ist protokolliert, nicht der Prozentwert.

    Args:
        policies: Das geladene Regelwerk.
    """
    rec = recorder(policies)
    assess_tolerance(kunde(), ziel(), policies.tolerance, rec)

    szenario = next(
        ereignis
        for ereignis in rec.events
        if ereignis.rule_id == "TOL-ITEM"
        and ereignis.inputs.get("item") == "verlustszenario_betrag"
    )

    assert "vorgelegter_verlustbetrag_eur" in szenario.inputs
    assert "prozent" not in str(szenario.inputs).lower()


def test_widerspruch_wird_erkannt_und_nicht_verrechnet(policies: PolicySet) -> None:
    """Ein Widerspruch erscheint im Ergebnis, die Punktzahl bleibt unverändert.

    Args:
        policies: Das geladene Regelwerk.
    """
    widerspruechlich = dict(ANTWORTEN_MITTE)
    widerspruechlich["selbsteinschaetzung"] = "CHANCENORIENTIERT"
    widerspruechlich["verlustszenario_betrag"] = "S1"

    ohne_widerspruch = dict(widerspruechlich)
    ohne_widerspruch["selbsteinschaetzung"] = "AUSGEWOGEN"

    rec = recorder(policies)
    mit = assess_tolerance(
        kunde(risk_attitude=RiskAttitude(answers=widerspruechlich)),
        ziel(),
        policies.tolerance,
        rec,
    )

    assert not mit.is_consistent
    assert mit.inconsistencies[0].rule_id == "TOL-K1"
    assert "TOL-CONFLICT" in regel_ids(rec)

    # Die Punktzahl entspricht exakt der Summe der Einzelantworten — es wurde
    # nichts abgezogen und nichts geglaettet.
    erwartet = sum(eintrag.weighted_points for eintrag in mit.scores)
    assert mit.total_points == erwartet


def test_widerspruchsfreie_antworten_erzeugen_keinen_befund(policies: PolicySet) -> None:
    """Der Grundfall ist widerspruchsfrei.

    Args:
        policies: Das geladene Regelwerk.
    """
    ergebnis = assess_tolerance(kunde(), ziel(), policies.tolerance, recorder(policies))

    assert ergebnis.is_consistent
    assert ergebnis.inconsistencies == ()


@pytest.mark.parametrize(
    ("antworten", "regel"),
    [
        (
            {
                "reaktion_auf_rueckgang": "NACHKAUFEN",
                "erfahrung_mit_schwankungen": "ERLEBT_UND_VERKAUFT",
            },
            "TOL-K2",
        ),
        (
            {
                "entnahmedisziplin": "SEHR_WAHRSCHEINLICH",
                "selbsteinschaetzung": "CHANCENORIENTIERT",
            },
            "TOL-K3",
        ),
        ({"verlustszenario_betrag": "S5", "reaktion_auf_rueckgang": "SOFORT_VERKAUFEN"}, "TOL-K4"),
    ],
)
def test_jede_konsistenzregel_greift(
    policies: PolicySet, antworten: dict[str, str], regel: str
) -> None:
    """Jede im Regelwerk hinterlegte Konsistenzregel löst tatsächlich aus.

    Args:
        policies: Das geladene Regelwerk.
        antworten: Die widersprüchlichen Antworten.
        regel: Erwartete Kennung der auslösenden Regel.
    """
    kombiniert = dict(ANTWORTEN_MITTE) | antworten

    befunde = find_inconsistencies(kombiniert, policies.tolerance)

    assert regel in {befund.rule_id for befund in befunde}


def test_fehlende_antwort_wird_nicht_mit_null_bewertet(policies: PolicySet) -> None:
    """Eine fehlende Antwort bricht ab, statt still null Punkte zu vergeben.

    Args:
        policies: Das geladene Regelwerk.
    """
    unvollstaendig = dict(ANTWORTEN_MITTE)
    del unvollstaendig["entnahmedisziplin"]

    with pytest.raises(IncompleteProfileError, match="keine Antwort"):
        assess_tolerance(
            kunde(risk_attitude=RiskAttitude(answers=unvollstaendig)),
            ziel(),
            policies.tolerance,
            recorder(policies),
        )


def test_unbekannte_antwort_wird_abgewiesen(policies: PolicySet) -> None:
    """Eine im Regelwerk nicht vorgesehene Antwort führt zum Fehler.

    Args:
        policies: Das geladene Regelwerk.
    """
    falsch = dict(ANTWORTEN_MITTE)
    falsch["selbsteinschaetzung"] = "TOTAL_RISIKOFREUDIG"

    with pytest.raises(PolicyError, match="unbekannte Ausprägung"):
        assess_tolerance(
            kunde(risk_attitude=RiskAttitude(answers=falsch)),
            ziel(),
            policies.tolerance,
            recorder(policies),
        )


def test_antwort_zu_unbekanntem_item_wird_abgewiesen(policies: PolicySet) -> None:
    """Eine Antwort zu einem nicht existierenden Item führt zum Fehler.

    Args:
        policies: Das geladene Regelwerk.
    """
    zusatz = dict(ANTWORTEN_MITTE) | {"lieblingsfarbe": "BLAU"}

    with pytest.raises(IncompleteProfileError, match="unbekannten Items"):
        assess_tolerance(
            kunde(risk_attitude=RiskAttitude(answers=zusatz)),
            ziel(),
            policies.tolerance,
            recorder(policies),
        )


# --- Kenntnisse und Erfahrungen -------------------------------------------


def test_kenntnisse_liefern_deckel_statt_punktzahl(policies: PolicySet) -> None:
    """Das Ergebnis ist eine Obergrenze und trägt keinen Score.

    Args:
        policies: Das geladene Regelwerk.
    """
    ergebnis = assess_knowledge(kunde(), policies.knowledge, recorder(policies))

    assert ergebnis.max_risk_class == 4
    assert ergebnis.qualifying_categories == ("ETF",)
    assert not hasattr(ergebnis, "total_points")


def test_nicht_erhobene_gattung_schaltet_nichts_frei(policies: PolicySet) -> None:
    """Gattungen ohne Angaben zählen als nicht erfüllt.

    Args:
        policies: Das geladene Regelwerk.
    """
    ergebnis = assess_knowledge(kunde(), policies.knowledge, recorder(policies))

    assert "DERIVATE" in ergebnis.failed_categories
    assert "nichts erhoben" in ergebnis.failed_categories["DERIVATE"]


def test_ohne_jeden_nachweis_gilt_der_vorgabewert(policies: PolicySet) -> None:
    """Erfüllt keine Gattung die Anforderungen, greift der Vorgabewert.

    Args:
        policies: Das geladene Regelwerk.
    """
    anfaenger = kenntnisse(
        entries=(
            InstrumentExperience(
                category=InstrumentCategory.AKTIEN,
                knowledge_level=KnowledgeLevel.KEINE,
                experience_years=Decimal("0.0"),
                transactions_per_year=0,
            ),
        )
    )
    ergebnis = assess_knowledge(kunde(knowledge=anfaenger), policies.knowledge, recorder(policies))

    assert ergebnis.max_risk_class == policies.knowledge.ohne_nachweis_max_risikoklasse
    assert ergebnis.qualifying_categories == ()


def test_zu_kurze_erfahrung_schaltet_nicht_frei(policies: PolicySet) -> None:
    """Alle drei Anforderungen müssen erfüllt sein, nicht nur der Kenntnisstand.

    Args:
        policies: Das geladene Regelwerk.
    """
    frisch = kenntnisse(
        entries=(
            InstrumentExperience(
                category=InstrumentCategory.ETF,
                knowledge_level=KnowledgeLevel.UMFASSEND,
                experience_years=Decimal("0.5"),
                transactions_per_year=50,
            ),
        )
    )
    ergebnis = assess_knowledge(kunde(knowledge=frisch), policies.knowledge, recorder(policies))

    assert "ETF" in ergebnis.failed_categories
    assert "Erfahrungsdauer" in ergebnis.failed_categories["ETF"]


def test_rangfolge_der_kenntnisstufen() -> None:
    """Die Skala ist geordnet: keine < Grund < erweitert < umfassend."""
    stufen = [
        KnowledgeLevel.KEINE,
        KnowledgeLevel.GRUNDKENNTNISSE,
        KnowledgeLevel.ERWEITERT,
        KnowledgeLevel.UMFASSEND,
    ]
    raenge = [knowledge_rank(stufe) for stufe in stufen]

    assert raenge == sorted(raenge)
    assert len(set(raenge)) == len(raenge)


# --- Zusammenführung -------------------------------------------------------


def test_minimum_bindet_nicht_der_mittelwert(policies: PolicySet) -> None:
    """Bei hoher Tragfähigkeit und niedriger Bereitschaft bindet die Bereitschaft.

    Ein Mittelwert läge bei (4+1)/2 = 2,5 und damit über dem Ergebnis. Genau
    diese Anhebung findet nicht statt.

    Args:
        policies: Das geladene Regelwerk.
    """
    profil = classify(
        kunde(risk_attitude=RiskAttitude(answers=dict(ANTWORTEN_NIEDRIG))),
        ziel(),
        policies,
        recorder(policies),
    )

    assert profil.capacity.risk_class == 4
    assert profil.tolerance.risk_class == 1
    assert profil.risk_class == 1
    assert profil.binding_axis is Axis.BEREITSCHAFT


def test_kenntnisse_deckeln_auch_bei_hohen_achsen(policies: PolicySet) -> None:
    """Ein niedriger Kenntnisdeckel bindet trotz hoher Tragfähigkeit und Bereitschaft.

    Args:
        policies: Das geladene Regelwerk.
    """
    profil = classify(
        kunde(
            knowledge=kenntnisse_fuer_klasse(2),
            risk_attitude=RiskAttitude(answers=dict(ANTWORTEN_HOCH)),
        ),
        ziel(),
        policies,
        recorder(policies),
    )

    assert profil.knowledge.max_risk_class == 2
    assert profil.risk_class == 2
    assert profil.binding_axis is Axis.KENNTNISSE


def test_divergenz_wird_ab_zwei_stufen_ausgewiesen(policies: PolicySet) -> None:
    """Liegen die Achsen um mindestens zwei Stufen auseinander, gibt es einen Befund.

    Args:
        policies: Das geladene Regelwerk.
    """
    rec = recorder(policies)
    profil = classify(
        kunde(risk_attitude=RiskAttitude(answers=dict(ANTWORTEN_NIEDRIG))),
        ziel(),
        policies,
        rec,
    )

    assert profil.has_divergence
    assert profil.divergence is not None
    assert profil.divergence.gap == 3
    assert profil.divergence.binding_axis is Axis.BEREITSCHAFT
    assert "CLS-DIV" in regel_ids(rec)


def test_ohne_divergenz_kein_befund(policies: PolicySet) -> None:
    """Liegen die Achsen dicht beieinander, entsteht kein Divergenzbefund.

    Args:
        policies: Das geladene Regelwerk.
    """
    rec = recorder(policies)
    profil = classify(kunde(), ziel(), policies, rec)

    assert not profil.has_divergence
    assert "CLS-DIV" not in regel_ids(rec)


def test_profil_traegt_den_vollstaendigen_trace(policies: PolicySet) -> None:
    """Das Ergebnisobjekt führt Klasse, Achse, Punktestände und Trace mit.

    Args:
        policies: Das geladene Regelwerk.
    """
    rec = recorder(policies)
    profil = classify(kunde(), ziel(), policies, rec)

    assert profil.trace == rec.events
    assert profil.trace_hash == rec.content_hash()
    assert len(profil.trace) >= 20
    zusammenfassung = profil.as_summary()
    assert zusammenfassung["risikoklasse"] == profil.risk_class
    assert zusammenfassung["bindende_achse"] == profil.binding_axis.value
    assert profil.policy_versions == policies.versions


def test_ko_bindet_und_wird_als_solches_ausgewiesen(policies: PolicySet) -> None:
    """Greift das K.-o.-Kriterium, wird es als bindende Achse benannt.

    Args:
        policies: Das geladene Regelwerk.
    """
    profil = classify(
        kunde(household=haushalt(liquidity_reserve_months=Decimal("0.5"))),
        ziel(),
        policies,
        recorder(policies),
    )

    assert profil.risk_class == 1
    assert profil.binding_axis is Axis.KO_LIQUIDITAET


# --- Glide-Path ------------------------------------------------------------


@pytest.mark.parametrize(
    ("monate", "erwartet"),
    [(360, 5), (120, 5), (119, 4), (84, 4), (83, 3), (48, 3), (47, 2), (24, 2), (23, 1), (0, 1)],
)
def test_glidepath_stufen(policies: PolicySet, monate: int, erwartet: int) -> None:
    """Die Stufen des Regelwerks greifen an den vorgesehenen Grenzen.

    Args:
        policies: Das geladene Regelwerk.
        monate: Restlaufzeit in Monaten.
        erwartet: Erwartete Obergrenze.
    """
    assert max_class_for_remaining_months(monate, policies.glidepath) == erwartet


def test_glidepath_deckelt_hart(policies: PolicySet) -> None:
    """Bei kurzer Restlaufzeit sinkt die Klasse unabhängig vom Profil.

    Args:
        policies: Das geladene Regelwerk.
    """
    rec = recorder(policies)
    profil = classify(kunde(), ziel(), policies, rec)
    gedeckelt = apply_glidepath(profil, 12, policies.glidepath, rec)

    assert profil.risk_class == 4
    assert gedeckelt.risk_class == 1
    assert gedeckelt.binding_axis is Axis.GLIDEPATH
    assert gedeckelt.glidepath_class == 1
    assert "GLD-CAP" in regel_ids(rec)


def test_glidepath_hebt_niemals_an(policies: PolicySet) -> None:
    """Eine lange Restlaufzeit verändert ein niedriges Profil nicht.

    Args:
        policies: Das geladene Regelwerk.
    """
    rec = recorder(policies)
    profil = classify(
        kunde(risk_attitude=RiskAttitude(answers=dict(ANTWORTEN_NIEDRIG))),
        ziel(),
        policies,
        rec,
    )
    danach = apply_glidepath(profil, 600, policies.glidepath, rec)

    assert danach.risk_class == profil.risk_class
    assert danach.binding_axis is profil.binding_axis
    assert danach.glidepath_class == 5


def test_negative_restlaufzeit_wird_abgewiesen(policies: PolicySet) -> None:
    """Eine negative Restlaufzeit ist unzulässig.

    Args:
        policies: Das geladene Regelwerk.
    """
    with pytest.raises(PolicyError, match="nicht negativ"):
        max_class_for_remaining_months(-1, policies.glidepath)
