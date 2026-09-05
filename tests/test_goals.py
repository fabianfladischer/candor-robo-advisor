"""Tests des Zielmodells und der Renditerechnung.

Der Kern sind **sieben von Hand nachgerechnete Kontrollfälle** (K1 bis K7). Jeder
ist im Test mit vollständigem Rechenweg kommentiert, sodass die Zahlen ohne den
Code nachvollziehbar sind.

Ergänzend geprüft werden die Randfälle der Renditerechnung sowie der
Zielkatalog und die Zuteilung der Sparrate auf mehrere Ziele.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from candor.data.policy_loader import load_policy_set
from candor.domain.goal import Goal, GoalPriority, GoalType
from candor.domain.policy import PolicySet
from candor.domain.result import ReturnCase, SolutionMethod, deflate
from candor.goals.model import (
    add_months,
    allocate_savings,
    allocation_order,
    apply_allocation,
    build_goal,
    template_for,
)
from candor.goals.required_return import (
    annualize,
    deannualize,
    future_value,
    inflate,
    required_return,
    solve_monthly_rate,
)
from candor.trace.recorder import TraceMetadata, TraceRecorder
from tests.builders import STICHTAG

#: Toleranz beim Vergleich numerisch bestimmter Renditen.
TOLERANZ = Decimal("0.000002")


@pytest.fixture(scope="module")
def policies() -> PolicySet:
    """Lädt das ausgelieferte Regelwerk einmal je Modul.

    Returns:
        Das geprüfte Regelwerk.
    """
    return load_policy_set()


def recorder(policies: PolicySet) -> TraceRecorder:
    """Baut einen leeren Recorder.

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


def ziel_mit(**felder: object) -> Goal:
    """Baut ein Ziel mit den übergebenen Feldern.

    Args:
        **felder: Abweichungen vom Grundfall.

    Returns:
        Das erzeugte Ziel.
    """
    grund: dict[str, object] = {
        "goal_id": "Z-K",
        "name": "Kontrollfall",
        "goal_type": GoalType.VERMOEGENSAUFBAU,
        "priority": GoalPriority.WUNSCH,
        "reference_date": date(2026, 1, 1),
        "target_date": date(2036, 1, 1),
        "target_amount": Decimal("20000.00"),
        "initial_capital": Decimal("10000.00"),
        "monthly_contribution": Decimal("0.00"),
    }
    grund.update(felder)
    return Goal(**grund)  # type: ignore[arg-type]


# ===========================================================================
# Von Hand nachgerechnete Kontrollfälle
# ===========================================================================


def test_k1_nur_startkapital_geschlossene_loesung() -> None:
    """K1 — nur Startkapital, keine Sparrate. Geschlossene Lösung nach (9).

    Gegeben:  PV = 10.000 €, PMT = 0 €, n = 120 Monate, Z = 20.000 €

    Rechnung:
        Ohne Sparrate gilt Z = PV · (1+i)^n, also
            (1+i)^120 = 20.000 / 10.000 = 2
            i         = 2^(1/120) − 1

        2^(1/120): ln 2 = 0,693147; 0,693147 / 120 = 0,00577623
                   e^0,00577623 = 1,00579295
            i = 0,005793  (auf sechs Stellen)

        Annualisiert nach (12):
            r = (1+i)^12 − 1 = 2^(12/120) − 1 = 2^0,1 − 1
        2^0,1: 0,693147 · 0,1 = 0,0693147; e^0,0693147 = 1,07177346
            r = 0,071773  =  7,1773 %

    Probe: 10.000 · 2 = 20.000 ✓
    """
    rate, fall, verfahren = solve_monthly_rate(
        Decimal("20000.00"), Decimal("10000.00"), Decimal("0.00"), 120
    )

    assert verfahren is SolutionMethod.GESCHLOSSEN
    assert fall is ReturnCase.ERREICHBAR
    assert abs(rate - Decimal("0.005793")) <= TOLERANZ
    assert abs(annualize(rate) - Decimal("0.071773")) <= Decimal("0.00001")


def test_k2_zielbetrag_gleich_einzahlungssumme() -> None:
    """K2 — Zielbetrag entspricht exakt der Einzahlungssumme. Exakt nach (11).

    Gegeben:  PV = 1.000 €, PMT = 100 €, n = 10 Monate, Z = 2.000 €

    Rechnung:
        Endwert ohne Rendite nach (7):
            FV(0) = PV + PMT · n = 1.000 + 100 · 10 = 2.000 €
        Das ist genau der Zielbetrag, also
            i = 0  und  r = 0

    Es wird keine Rendite benötigt; die Einzahlungen allein genügen.
    """
    rate, fall, verfahren = solve_monthly_rate(
        Decimal("2000.00"), Decimal("1000.00"), Decimal("100.00"), 10
    )

    assert rate == Decimal(0)
    assert fall is ReturnCase.OHNE_RENDITE_ERREICHBAR
    assert verfahren is SolutionMethod.EXAKT_NULL
    assert annualize(rate) == Decimal(0)


def test_k3_eine_periode_geschlossene_loesung() -> None:
    """K3 — genau ein Monat. Geschlossene Lösung nach (10).

    Gegeben:  PV = 1.000 €, PMT = 100 €, n = 1 Monat, Z = 1.200 €

    Rechnung:
        Bei n = 1 wird die einzige Rate am Monatsende eingezahlt und nicht mehr
        verzinst. Aus (6) mit n = 1:
            Z = PV · (1+i) + PMT
            1.200 = 1.000 · (1+i) + 100
            1.100 = 1.000 · (1+i)
            1+i   = 1,1
            i     = 0,1  =  10 % je Monat

        Annualisiert nach (12):
            r = 1,1^12 − 1
        1,1^2 = 1,21;  1,1^4 = 1,4641;  1,1^8 = 2,14358881
        1,1^12 = 1,1^8 · 1,1^4 = 2,14358881 · 1,4641 = 3,13842838
            r = 2,138428  =  213,84 %
    """
    rate, fall, verfahren = solve_monthly_rate(
        Decimal("1200.00"), Decimal("1000.00"), Decimal("100.00"), 1
    )

    assert verfahren is SolutionMethod.GESCHLOSSEN
    assert fall is ReturnCase.ERREICHBAR
    assert rate == Decimal("0.100000")
    assert abs(annualize(rate) - Decimal("2.138428")) <= Decimal("0.00001")


def test_k4_reiner_sparplan_numerisch() -> None:
    """K4 — reiner Sparplan ohne Startkapital. Numerisch über Brent.

    Gegeben:  PV = 0 €, PMT = 100 €, n = 12 Monate

    Vorwärtsprobe zu i = 0,01 nach (4):
        1,01^12: 1,01^2 = 1,0201;  1,01^4 = 1,04060401
                 1,01^8 = 1,04060401^2 = 1,08285671
                 1,01^12 = 1,08285671 · 1,04060401 = 1,12682503
        FV = 100 · (1,12682503 − 1) / 0,01
           = 100 · 12,682503
           = 1.268,25 €

    Rückwärts muss Z = 1.268,25 € wieder i ≈ 0,01 liefern.

        Annualisiert: r = 1,01^12 − 1 = 0,126825 = 12,6825 %
    """
    probe = future_value(Decimal("0.00"), Decimal("100.00"), Decimal("0.01"), 12)
    assert abs(probe - Decimal("1268.25")) < Decimal("0.01")

    rate, fall, verfahren = solve_monthly_rate(
        Decimal("1268.25"), Decimal("0.00"), Decimal("100.00"), 12
    )

    assert verfahren is SolutionMethod.BRENT
    assert fall is ReturnCase.ERREICHBAR
    assert abs(rate - Decimal("0.01")) <= Decimal("0.00001")
    assert abs(annualize(rate) - Decimal("0.126825")) <= Decimal("0.0001")


def test_k5_startkapital_und_sparplan_numerisch() -> None:
    """K5 — Startkapital und Sparplan zusammen. Numerisch über Brent.

    Gegeben:  PV = 5.000 €, PMT = 200 €, n = 24 Monate

    Vorwärtsprobe zu i = 0,005 nach (6):
        1,005^2  = 1,010025
        1,005^4  = 1,010025^2      = 1,02015050
        1,005^8  = 1,02015050^2    = 1,04070704
        1,005^16 = 1,04070704^2    = 1,08307116
        1,005^24 = 1,08307116 · 1,04070704 = 1,12715978

        Zinsteil des Startkapitals:
            5.000 · 1,12715978 = 5.635,80 €
        Rentenendwert der Raten:
            200 · (1,12715978 − 1) / 0,005 = 200 · 25,431956 = 5.086,39 €
        Summe:
            FV = 5.635,80 + 5.086,39 = 10.722,19 €

    Rückwärts muss Z = 10.722,19 € wieder i ≈ 0,005 liefern.

        Annualisiert: r = 1,005^12 − 1
        1,005^12 = 1,04070704 · 1,02015050 = 1,06167781
            r = 0,061678  =  6,1678 %
    """
    probe = future_value(Decimal("5000.00"), Decimal("200.00"), Decimal("0.005"), 24)
    assert abs(probe - Decimal("10722.19")) < Decimal("0.01")

    rate, fall, verfahren = solve_monthly_rate(
        Decimal("10722.19"), Decimal("5000.00"), Decimal("200.00"), 24
    )

    assert fall is ReturnCase.ERREICHBAR
    assert verfahren is SolutionMethod.BRENT
    assert abs(rate - Decimal("0.005")) <= Decimal("0.00001")
    assert abs(annualize(rate) - Decimal("0.061678")) <= Decimal("0.0001")


def test_k6_zielbetrag_bereits_erreicht_negative_rendite() -> None:
    """K6 — Startkapital übersteigt den Zielbetrag. Negative Rendite.

    Gegeben:  PV = 100.000 €, PMT = 0 €, n = 12 Monate, Z = 90.000 €

    Rechnung:
        (1+i)^12 = 90.000 / 100.000 = 0,9
        Da n genau zwölf Monate sind, ist die Jahresrendite unmittelbar
            r = 0,9 − 1 = −0,10 = −10 %

        Monatlich: i = 0,9^(1/12) − 1
        ln 0,9 = −0,10536052;  / 12 = −0,00878004
        e^(−0,00878004) = 0,99125840
            i = −0,008742

    Der Fall ist kein Fehler: Wer sein Ziel bereits übertroffen hat, braucht
    keine Rendite, sondern Kapitalerhalt.
    """
    rate, fall, verfahren = solve_monthly_rate(
        Decimal("90000.00"), Decimal("100000.00"), Decimal("0.00"), 12
    )

    assert verfahren is SolutionMethod.GESCHLOSSEN
    assert fall is ReturnCase.BEREITS_ERREICHT
    assert abs(rate - Decimal("-0.008742")) <= TOLERANZ
    assert abs(annualize(rate) - Decimal("-0.100000")) <= Decimal("0.00001")


def test_k7_kaufkraft_und_fisher() -> None:
    """K7 — Inflationsrechnung nach (15) und Fisher-Umrechnung nach (14).

    Gegeben:  Z = 100.000 €, n = 120 Monate (10 Jahre), π = 0,02

    Kaufkrafterhaltender Zielbetrag nach (15):
        1,02^2  = 1,0404
        1,02^4  = 1,0404^2        = 1,08243216
        1,02^8  = 1,08243216^2    = 1,17165938
        1,02^10 = 1,17165938 · 1,0404 = 1,21899442
        Z_Kaufkraft = 100.000 · 1,21899442 = 121.899,44 €

    Reale Rendite bei nominal r = 0,05 nach (14):
        r_real = 1,05 / 1,02 − 1
               = 1,02941176 − 1
               = 0,029412  =  2,9412 %

        Die Näherung r − π ergäbe 3,0000 % — um 0,0588 Prozentpunkte zu hoch.
    """
    assert inflate(Decimal("100000.00"), Decimal("0.02"), 120) == Decimal("121899.44")
    assert deflate(Decimal("0.05"), Decimal("0.02")) == Decimal("0.029412")


# ===========================================================================
# Randfälle der Renditerechnung
# ===========================================================================


def test_kein_fall_liefert_nan(policies: PolicySet) -> None:
    """Keiner der Randfälle liefert ``NaN`` oder ``None``.

    Args:
        policies: Das geladene Regelwerk.
    """
    faelle = [
        ziel_mit(target_amount=Decimal("20000.00")),
        ziel_mit(target_amount=Decimal("5000.00")),
        ziel_mit(target_amount=Decimal("0.01")),
        ziel_mit(
            target_amount=Decimal("99999999.00"),
            initial_capital=Decimal("1.00"),
            monthly_contribution=Decimal("1.00"),
        ),
        ziel_mit(
            initial_capital=Decimal("0.00"),
            monthly_contribution=Decimal("100.00"),
            target_amount=Decimal("15000.00"),
        ),
    ]

    for anlageziel in faelle:
        ergebnis = required_return(anlageziel, policies.goals)
        assert ergebnis.nominal.monthly_rate.is_finite()
        assert ergebnis.nominal.annual_rate.is_finite()
        assert ergebnis.purchasing_power.annual_rate.is_finite()
        assert ergebnis.real_annual.is_finite()


def test_unerreichbarer_zielbetrag_wird_als_schranke_ausgewiesen(policies: PolicySet) -> None:
    """Ein unerreichbarer Zielbetrag liefert die Suchgrenze, kein Ergebnis.

    Args:
        policies: Das geladene Regelwerk.
    """
    # Zwei Monate Horizont: Selbst bei der oberen Suchgrenze von 100 % je Monat
    # erreicht der Endwert nur 1 · 2² + 1 · (2² − 1)/1 = 7 €.
    ergebnis = required_return(
        ziel_mit(
            reference_date=date(2026, 1, 1),
            target_date=date(2026, 3, 1),
            target_amount=Decimal("1000.00"),
            initial_capital=Decimal("1.00"),
            monthly_contribution=Decimal("1.00"),
        ),
        policies.goals,
    )

    assert ergebnis.nominal.case is ReturnCase.NICHT_ERREICHBAR
    assert ergebnis.nominal.method is SolutionMethod.SCHRANKE
    assert any("Suchbereichsgrenze" in hinweis for hinweis in ergebnis.warnings)


def test_uebererfuellter_zielbetrag_wird_als_schranke_ausgewiesen() -> None:
    """Ein Zielbetrag unter der letzten Rate ist nicht mehr sinnvoll lösbar."""
    rate, fall, verfahren = solve_monthly_rate(
        Decimal("0.01"), Decimal("0.00"), Decimal("500.00"), 24
    )

    assert fall is ReturnCase.UEBERERFUELLT
    assert verfahren is SolutionMethod.SCHRANKE
    assert rate == Decimal("-0.99")


def test_kurzer_horizont_wird_als_hinweis_ausgewiesen(policies: PolicySet) -> None:
    """Bei einem Horizont unter einem Jahr wird die Hochrechnung benannt.

    Kontrolle der Annualisierung: n = 6, PV = 1.000, PMT = 0, Z = 1.100.
        (1+i)^6 = 1,1  ⇒  (1+i)^12 = 1,1² = 1,21  ⇒  r = 21 % exakt.

    Args:
        policies: Das geladene Regelwerk.
    """
    kurz = ziel_mit(
        reference_date=date(2026, 1, 1),
        target_date=date(2026, 7, 1),
        initial_capital=Decimal("1000.00"),
        monthly_contribution=Decimal("0.00"),
        target_amount=Decimal("1100.00"),
    )
    ergebnis = required_return(kurz, policies.goals)

    assert kurz.horizon_months == 6
    assert abs(ergebnis.nominal.annual_rate - Decimal("0.21")) <= Decimal("0.00001")
    assert any("Extrapolation" in hinweis for hinweis in ergebnis.warnings)


def test_nullsparrate_wird_benannt(policies: PolicySet) -> None:
    """Ohne Sparrate wird der fehlende Ausgleichshebel benannt.

    Args:
        policies: Das geladene Regelwerk.
    """
    ergebnis = required_return(ziel_mit(monthly_contribution=Decimal("0.00")), policies.goals)

    assert any("Ohne Sparrate" in hinweis for hinweis in ergebnis.warnings)


def test_ohne_kapital_und_ohne_sparrate_wird_abgewiesen() -> None:
    """Ohne jeden Weg zum Ziel bricht die Rechnung ab, statt null zu liefern."""
    with pytest.raises(ValueError, match="kein Zielbetrag erreichbar"):
        solve_monthly_rate(Decimal("1000.00"), Decimal("0.00"), Decimal("0.00"), 12)


def test_horizont_unter_einem_monat_wird_abgewiesen() -> None:
    """Ein Horizont unter einem Monat ist nicht auswertbar."""
    with pytest.raises(ValueError, match="mindestens einen Monat"):
        solve_monthly_rate(Decimal("1000.00"), Decimal("100.00"), Decimal("0.00"), 0)


def test_endwert_weist_unmoegliche_rendite_ab() -> None:
    """Eine Monatsrendite bei oder unter −100 % ist keine sinnvolle Eingabe."""
    with pytest.raises(ValueError, match="über −100 %"):
        future_value(Decimal("1000.00"), Decimal("0.00"), Decimal("-1.0"), 12)


def test_annualisierung_und_umkehrung_sind_konsistent() -> None:
    """Annualisieren und Deannualisieren heben einander auf."""
    for monatsrate in ("0.005", "0.01", "-0.002"):
        rate = Decimal(monatsrate)
        zurueck = deannualize(annualize(rate))
        assert abs(zurueck - rate) < Decimal("0.000001")


def test_nominal_und_real_werden_getrennt_ausgewiesen(policies: PolicySet) -> None:
    """Der kaufkrafterhaltende Zielbetrag verlangt eine höhere Rendite.

    Args:
        policies: Das geladene Regelwerk.
    """
    ergebnis = required_return(ziel_mit(), policies.goals)

    assert ergebnis.purchasing_power.target_amount > ergebnis.nominal.target_amount
    assert ergebnis.purchasing_power.annual_rate > ergebnis.nominal.annual_rate
    assert ergebnis.real_annual < ergebnis.nominal.annual_rate
    assert ergebnis.inflation_annual == policies.goals.inflationsannahme_jaehrlich


def test_abweichende_inflationsannahme_wirkt(policies: PolicySet) -> None:
    """Eine übergebene Inflationsannahme überschreibt die des Regelwerks.

    Args:
        policies: Das geladene Regelwerk.
    """
    ohne = required_return(ziel_mit(), policies.goals, inflation_annual=Decimal("0.00"))
    mit = required_return(ziel_mit(), policies.goals, inflation_annual=Decimal("0.05"))

    assert ohne.purchasing_power.target_amount == ohne.nominal.target_amount
    assert mit.purchasing_power.target_amount > ohne.purchasing_power.target_amount


def test_renditerechnung_schreibt_ein_trace_ereignis(policies: PolicySet) -> None:
    """Die Renditerechnung ist im Trace nachvollziehbar.

    Args:
        policies: Das geladene Regelwerk.
    """
    rec = recorder(policies)
    required_return(ziel_mit(), policies.goals, recorder=rec)

    ereignis = rec.events[0]
    assert ereignis.rule_id == "GOAL-RET"
    assert ereignis.inputs["rentenformel"] == "nachschüssig"
    assert "erforderliche_rendite_real_pa" in str(ereignis.output)


# ===========================================================================
# Zielkatalog
# ===========================================================================


@pytest.mark.parametrize("zieltyp", list(GoalType))
def test_jeder_zieltyp_hat_vorbelegungen(policies: PolicySet, zieltyp: GoalType) -> None:
    """Der Zielkatalog deckt jeden Zieltyp ab.

    Args:
        policies: Das geladene Regelwerk.
        zieltyp: Der geprüfte Zieltyp.
    """
    vorlage = template_for(zieltyp, policies.goals)

    assert vorlage.standardhorizont_monate >= 1
    assert vorlage.standardprioritaet in set(GoalPriority)
    assert len(vorlage.bezeichnung) >= 3


def test_vorbelegungen_werden_uebernommen(policies: PolicySet) -> None:
    """Ohne Angabe gelten Standardhorizont und Standardpriorität des Zieltyps.

    Args:
        policies: Das geladene Regelwerk.
    """
    vorlage = template_for(GoalType.ALTERSVORSORGE, policies.goals)
    anlageziel = build_goal(
        "Z-1",
        GoalType.ALTERSVORSORGE,
        Decimal("300000.00"),
        STICHTAG,
        policies.goals,
        monthly_contribution=Decimal("500.00"),
    )

    assert anlageziel.priority is vorlage.standardprioritaet
    assert anlageziel.target_date == add_months(STICHTAG, vorlage.standardhorizont_monate)
    assert anlageziel.name == vorlage.bezeichnung
    assert anlageziel.goal_type is GoalType.ALTERSVORSORGE


def test_vorbelegungen_lassen_sich_ueberschreiben(policies: PolicySet) -> None:
    """Jede Vorbelegung weicht einer ausdrücklichen Angabe.

    Args:
        policies: Das geladene Regelwerk.
    """
    anlageziel = build_goal(
        "Z-2",
        GoalType.ALTERSVORSORGE,
        Decimal("300000.00"),
        STICHTAG,
        policies.goals,
        name="Eigene Bezeichnung",
        target_date=date(2040, 1, 1),
        priority=GoalPriority.WUNSCH,
        monthly_contribution=Decimal("500.00"),
    )

    assert anlageziel.name == "Eigene Bezeichnung"
    assert anlageziel.target_date == date(2040, 1, 1)
    assert anlageziel.priority is GoalPriority.WUNSCH


def test_zieltyp_hebt_die_mindestwahrscheinlichkeit_an(policies: PolicySet) -> None:
    """Die Altersvorsorge verlangt mehr als der MUSS-Grundwert.

    Args:
        policies: Das geladene Regelwerk.
    """
    grundwert = policies.goals.minimum_success_probability(GoalPriority.MUSS)
    mit_zieltyp = policies.goals.minimum_success_probability(
        GoalPriority.MUSS, None, GoalType.ALTERSVORSORGE
    )

    assert mit_zieltyp > grundwert
    assert mit_zieltyp == Decimal("0.90")


def test_eigene_vorgabe_kann_nur_anheben(policies: PolicySet) -> None:
    """Eine niedrigere Eigenvorgabe bleibt wirkungslos.

    Args:
        policies: Das geladene Regelwerk.
    """
    schwelle = policies.goals.minimum_success_probability(
        GoalPriority.MUSS, Decimal("0.10"), GoalType.ALTERSVORSORGE
    )
    angehoben = policies.goals.minimum_success_probability(
        GoalPriority.MUSS, Decimal("0.99"), GoalType.ALTERSVORSORGE
    )

    assert schwelle == Decimal("0.90")
    assert angehoben == Decimal("0.99")


def test_zielkatalog_schreibt_ein_trace_ereignis(policies: PolicySet) -> None:
    """Die Übernahme der Vorbelegungen ist im Trace nachvollziehbar.

    Args:
        policies: Das geladene Regelwerk.
    """
    rec = recorder(policies)
    build_goal(
        "Z-3",
        GoalType.LIQUIDITAETSPUFFER,
        Decimal("12000.00"),
        STICHTAG,
        policies.goals,
        monthly_contribution=Decimal("400.00"),
        recorder=rec,
    )

    assert rec.events[0].rule_id == "GOAL-TPL"
    assert rec.events[0].inputs["termin_ueberschrieben"] is False


# ===========================================================================
# Mehrzielfähigkeit
# ===========================================================================


def drei_ziele(policies: PolicySet) -> tuple[Goal, ...]:
    """Baut drei Ziele mit unterschiedlicher Priorität und Fristigkeit.

    Args:
        policies: Das geladene Regelwerk.

    Returns:
        Die drei Ziele in absichtlich ungünstiger Eingabereihenfolge.
    """
    wunsch = build_goal(
        "Z-WUNSCH",
        GoalType.VERMOEGENSAUFBAU,
        Decimal("50000.00"),
        STICHTAG,
        policies.goals,
        monthly_contribution=Decimal("300.00"),
    )
    muss_spaet = build_goal(
        "Z-ALTER",
        GoalType.ALTERSVORSORGE,
        Decimal("300000.00"),
        STICHTAG,
        policies.goals,
        monthly_contribution=Decimal("400.00"),
    )
    muss_frueh = build_goal(
        "Z-PUFFER",
        GoalType.LIQUIDITAETSPUFFER,
        Decimal("12000.00"),
        STICHTAG,
        policies.goals,
        monthly_contribution=Decimal("500.00"),
    )
    return (wunsch, muss_spaet, muss_frueh)


def test_reihenfolge_muss_vor_wunsch_dann_frist(policies: PolicySet) -> None:
    """MUSS vor WUNSCH, innerhalb der Priorität der frühere Termin zuerst.

    Args:
        policies: Das geladene Regelwerk.
    """
    sortiert = allocation_order(drei_ziele(policies), policies.goals)

    assert [ziel.goal_id for ziel in sortiert] == ["Z-PUFFER", "Z-ALTER", "Z-WUNSCH"]


def test_reihenfolge_ist_bei_gleichstand_deterministisch(policies: PolicySet) -> None:
    """Bei gleicher Priorität und gleichem Termin entscheidet die Kennung.

    Args:
        policies: Das geladene Regelwerk.
    """
    gemeinsam = {
        "goal_type": GoalType.VERMOEGENSAUFBAU,
        "target_amount": Decimal("10000.00"),
        "reference_date": STICHTAG,
        "target_date": date(2036, 1, 1),
        "priority": GoalPriority.WUNSCH,
        "monthly_contribution": Decimal("100.00"),
        "initial_capital": Decimal("0.00"),
    }
    b = Goal(goal_id="Z-B", name="B", **gemeinsam)  # type: ignore[arg-type]
    a = Goal(goal_id="Z-A", name="A", **gemeinsam)  # type: ignore[arg-type]

    assert [z.goal_id for z in allocation_order((b, a), policies.goals)] == ["Z-A", "Z-B"]
    assert [z.goal_id for z in allocation_order((a, b), policies.goals)] == ["Z-A", "Z-B"]


def test_muss_ziele_werden_zuerst_bedient(policies: PolicySet) -> None:
    """Bei knappem Budget erhalten MUSS-Ziele die Sparrate zuerst.

    Gewünscht sind 500 + 400 + 300 = 1.200 €, verfügbar sind 800 €.
    Erwartet: Puffer 500 € voll, Altersvorsorge 300 € (Rest), Wunsch 0 €.

    Args:
        policies: Das geladene Regelwerk.
    """
    ergebnis = allocate_savings(drei_ziele(policies), Decimal("800.00"), policies.goals)
    nach_kennung = {eintrag.goal_id: eintrag for eintrag in ergebnis.allocations}

    assert nach_kennung["Z-PUFFER"].allocated == Decimal("500.00")
    assert nach_kennung["Z-ALTER"].allocated == Decimal("300.00")
    assert nach_kennung["Z-WUNSCH"].allocated == Decimal("0.00")
    assert ergebnis.allocated_total == Decimal("800.00")
    assert ergebnis.unallocated == Decimal("0.00")
    assert ergebnis.underfunded == ("Z-ALTER", "Z-WUNSCH")
    assert not ergebnis.is_complete


def test_ausreichendes_budget_bedient_alle_ziele(policies: PolicySet) -> None:
    """Reicht das Budget, erhält jedes Ziel seine gewünschte Rate.

    Args:
        policies: Das geladene Regelwerk.
    """
    ergebnis = allocate_savings(drei_ziele(policies), Decimal("1500.00"), policies.goals)

    assert ergebnis.is_complete
    assert ergebnis.allocated_total == Decimal("1200.00")
    assert ergebnis.unallocated == Decimal("300.00")


def test_zuteilung_schreibt_ein_trace_ereignis(policies: PolicySet) -> None:
    """Die Zuteilung ist samt Reihenfolge im Trace nachvollziehbar.

    Args:
        policies: Das geladene Regelwerk.
    """
    rec = recorder(policies)
    allocate_savings(drei_ziele(policies), Decimal("800.00"), policies.goals, recorder=rec)

    ereignis = rec.events[0]
    assert ereignis.rule_id == "GOAL-ALLOC"
    assert ereignis.inputs["reihenfolge"] == ["Z-PUFFER", "Z-ALTER", "Z-WUNSCH"]


def test_zuteilung_wird_in_die_ziele_uebernommen(policies: PolicySet) -> None:
    """Die zugeteilten Raten ersetzen die gewünschten.

    Args:
        policies: Das geladene Regelwerk.
    """
    ziele = drei_ziele(policies)
    ergebnis = allocate_savings(ziele, Decimal("800.00"), policies.goals)
    angepasst = apply_allocation(ziele, ergebnis)

    # Z-WUNSCH erhaelt nichts und hat kein Startkapital, faellt also weg.
    assert [ziel.goal_id for ziel in angepasst] == ["Z-PUFFER", "Z-ALTER"]
    assert angepasst[1].monthly_contribution == Decimal("300.00")


def test_leere_zielliste_wird_abgewiesen(policies: PolicySet) -> None:
    """Ohne Ziele gibt es nichts zu verteilen.

    Args:
        policies: Das geladene Regelwerk.
    """
    with pytest.raises(ValueError, match="keine Ziele"):
        allocate_savings((), Decimal("500.00"), policies.goals)


def test_negatives_budget_wird_abgewiesen(policies: PolicySet) -> None:
    """Ein negatives Budget ist keine sinnvolle Eingabe.

    Args:
        policies: Das geladene Regelwerk.
    """
    with pytest.raises(ValueError, match="nicht negativ"):
        allocate_savings(drei_ziele(policies), Decimal("-1.00"), policies.goals)


def test_negative_monatszahl_wird_abgewiesen() -> None:
    """Die Kalenderrechnung weist negative Monatszahlen ab."""
    with pytest.raises(ValueError, match="nicht negativ"):
        add_months(STICHTAG, -1)
