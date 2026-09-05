"""Tests des Domänenkerns: Kundenprofil und Anlageziel.

Geprüft wird vor allem, dass die Validierung **greift** — also dass unzulässige
Profile abgewiesen werden und nicht stillschweigend mit Vorgabewerten
durchlaufen (Prinzip P5). Jeder Fehlerfall wird einzeln geprüft, weil ein
Sammeltest nicht zeigen würde, welche Prüfung tatsächlich angeschlagen hat.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from candor.domain.client import (
    EmploymentType,
    InstrumentCategory,
    InstrumentExperience,
    KnowledgeLevel,
    SustainabilityComponent,
    SustainabilityPreferences,
    parse_client,
)
from candor.domain.errors import IncompleteProfileError, InconsistentProfileError
from candor.domain.goal import GoalPriority, parse_goal
from tests.builders import haushalt, kenntnisse, kunde, ziel

# --- Haushalt --------------------------------------------------------------


def test_tragfaehiger_haushalt_wird_angenommen() -> None:
    """Ein stimmiger Haushalt lässt sich anlegen."""
    assert haushalt().employment_type is EmploymentType.UNBEFRISTET


def test_sparquote_ueber_ueberschuss_wird_abgewiesen() -> None:
    """Eine Sparrate über dem verfügbaren Überschuss ist ein Widerspruch."""
    with pytest.raises(InconsistentProfileError, match="übersteigt den verfügbaren Überschuss"):
        haushalt(free_savings_amount=Decimal("2000.00"))


def test_schuldendienst_ueber_einkommen_wird_abgewiesen() -> None:
    """Ein Kapitaldienst über dem Nettoeinkommen ist ein Widerspruch."""
    with pytest.raises(InconsistentProfileError, match="Schuldendienst"):
        haushalt(
            monthly_debt_service=Decimal("5000.00"),
            fixed_monthly_costs=Decimal("0.00"),
            free_savings_amount=Decimal("0.00"),
        )


def test_negativer_betrag_wird_abgewiesen() -> None:
    """Negative Geldbeträge sind unzulässig."""
    with pytest.raises(Exception, match="greater than or equal to 0"):
        haushalt(net_monthly_income=Decimal("-1.00"))


def test_haushalt_ist_unveraenderlich() -> None:
    """Ein geprüfter Haushalt lässt sich nachträglich nicht ändern."""
    modell = haushalt()
    with pytest.raises(ValidationError, match=r"frozen"):
        modell.net_monthly_income = Decimal("1.00")  # type: ignore[misc]


# --- Kenntnisse und Erfahrungen -------------------------------------------


def test_fehlende_kenntnisangaben_nennen_artikel_55() -> None:
    """Ohne Angaben zu Kenntnissen bricht die Prüfung mit Rechtsbezug ab."""
    with pytest.raises(IncompleteProfileError) as fehler:
        kenntnisse(entries=())

    assert fehler.value.field == "knowledge.entries"
    assert fehler.value.legal_basis is not None
    assert "Art. 55" in fehler.value.legal_basis


def test_doppelte_gattung_wird_abgewiesen() -> None:
    """Dieselbe Instrumentengattung darf nur einmal vorkommen."""
    eintrag = InstrumentExperience(
        category=InstrumentCategory.AKTIEN,
        knowledge_level=KnowledgeLevel.GRUNDKENNTNISSE,
        experience_years=Decimal("1.0"),
        transactions_per_year=1,
    )
    with pytest.raises(InconsistentProfileError, match="Mehrfache Angaben"):
        kenntnisse(entries=(eintrag, eintrag))


def test_erfahrung_ohne_kenntnisse_wird_abgewiesen() -> None:
    """Erfahrungsjahre bei zugleich 'keine Kenntnisse' sind nicht auswertbar."""
    with pytest.raises(InconsistentProfileError, match="keine Kenntnisse"):
        InstrumentExperience(
            category=InstrumentCategory.DERIVATE,
            knowledge_level=KnowledgeLevel.KEINE,
            experience_years=Decimal("3.0"),
            transactions_per_year=0,
        )


def test_nicht_erhobene_gattung_liefert_none() -> None:
    """Zu einer nicht erhobenen Gattung liegt ausdrücklich nichts vor."""
    stand = kenntnisse()

    assert stand.for_category(InstrumentCategory.ETF) is not None
    assert stand.for_category(InstrumentCategory.DERIVATE) is None


# --- Nachhaltigkeitspräferenzen -------------------------------------------


def test_praeferenz_ohne_auspraegung_wird_abgewiesen() -> None:
    """Erklärte Präferenzen ohne gewählte Ausprägung sind widersprüchlich."""
    with pytest.raises(InconsistentProfileError, match="keine der drei Ausprägungen"):
        SustainabilityPreferences(has_preferences=True)


def test_auspraegung_ohne_erklaerte_praeferenz_wird_abgewiesen() -> None:
    """Eine Ausprägung ohne erklärte Präferenz ist widersprüchlich."""
    with pytest.raises(InconsistentProfileError, match="keine Nachhaltigkeitspräferenzen"):
        SustainabilityPreferences(
            has_preferences=False,
            taxonomy_aligned=SustainabilityComponent(requested=True, minimum_share=Decimal("0.30")),
        )


def test_verlangte_auspraegung_ohne_mindestanteil_wird_abgewiesen() -> None:
    """Eine verlangte Ausprägung ohne Mindestanteil ist nicht umsetzbar."""
    with pytest.raises(InconsistentProfileError, match="Mindestanteil fehlt"):
        SustainabilityComponent(requested=True)


def test_mindestanteil_ohne_verlangen_wird_abgewiesen() -> None:
    """Ein Mindestanteil ohne Verlangen ist widersprüchlich."""
    with pytest.raises(InconsistentProfileError, match="nicht verlangt"):
        SustainabilityComponent(requested=False, minimum_share=Decimal("0.20"))


def test_dreistufige_praeferenz_wird_angenommen() -> None:
    """Alle drei Ausprägungen der DelVO 2021/1253 lassen sich nebeneinander setzen."""
    praeferenz = SustainabilityPreferences(
        has_preferences=True,
        taxonomy_aligned=SustainabilityComponent(requested=True, minimum_share=Decimal("0.20")),
        sfdr_sustainable=SustainabilityComponent(requested=True, minimum_share=Decimal("0.40")),
        pai_considered=SustainabilityComponent(requested=True, minimum_share=Decimal("0.60")),
    )

    assert praeferenz.taxonomy_aligned.minimum_share == Decimal("0.20")
    assert praeferenz.pai_considered.minimum_share == Decimal("0.60")


# --- parse_client ----------------------------------------------------------


def test_fehlendes_pflichtfeld_nennt_feld_und_rechtsgrundlage() -> None:
    """Eine fehlende Pflichtangabe wird mit Feldpfad und Rechtsgrundlage gemeldet."""
    rohdaten = kunde().model_dump()
    del rohdaten["household"]["liquidity_reserve_months"]

    with pytest.raises(IncompleteProfileError) as fehler:
        parse_client(rohdaten)

    assert fehler.value.field == "household.liquidity_reserve_months"
    assert fehler.value.legal_basis is not None
    assert "Art. 54" in fehler.value.legal_basis
    assert "Rechtsgrundlage" in str(fehler.value)


def test_unbekanntes_feld_wird_abgewiesen() -> None:
    """Ein unbekanntes Feld im Profil führt zum Abbruch, nicht zum Verwerfen."""
    rohdaten = kunde().model_dump()
    rohdaten["household"]["nettoeinkomen"] = 1000

    with pytest.raises(IncompleteProfileError):
        parse_client(rohdaten)


def test_vollstaendiges_profil_wird_angenommen() -> None:
    """Ein vollständiges Profil lässt sich aus Rohdaten aufbauen."""
    geprueft = parse_client(kunde().model_dump())

    assert geprueft.client_id == "K-0001"
    assert geprueft.knowledge.entries[0].category is InstrumentCategory.ETF


# --- Anlageziel ------------------------------------------------------------


def test_zieltermin_in_der_vergangenheit_wird_abgewiesen() -> None:
    """Ein Zieltermin vor dem Stichtag ist unzulässig."""
    with pytest.raises(InconsistentProfileError, match="liegt nicht nach dem Stichtag"):
        ziel(target_date=date(2020, 1, 1))


def test_ziel_ohne_kapital_und_ohne_sparrate_wird_abgewiesen() -> None:
    """Ohne Startkapital und ohne Sparrate ist kein Ziel erreichbar."""
    with pytest.raises(IncompleteProfileError) as fehler:
        ziel(initial_capital=Decimal("0.00"), monthly_contribution=Decimal("0.00"))

    assert fehler.value.field == "goal.monthly_contribution"


def test_zielbetrag_null_wird_abgewiesen() -> None:
    """Ein Zielbetrag von null ist unzulässig."""
    with pytest.raises(Exception, match="greater than 0"):
        ziel(target_amount=Decimal("0.00"))


@pytest.mark.parametrize(
    ("stichtag", "zieltermin", "erwartet"),
    [
        (date(2026, 1, 15), date(2027, 1, 15), 12),
        (date(2026, 1, 15), date(2027, 1, 14), 11),
        (date(2026, 1, 31), date(2026, 3, 1), 1),
        (date(2026, 7, 26), date(2046, 7, 1), 239),
    ],
)
def test_horizont_wird_in_vollen_monaten_gemessen(
    stichtag: date, zieltermin: date, erwartet: int
) -> None:
    """Angebrochene Monate zählen nicht mit.

    Args:
        stichtag: Stichtag der Erhebung.
        zieltermin: Zieltermin.
        erwartet: Erwartete Zahl voller Monate.
    """
    assert ziel(reference_date=stichtag, target_date=zieltermin).horizon_months == erwartet


def test_horizont_in_jahren_wird_gerundet() -> None:
    """Der Horizont in Jahren wird auf zwei Stellen gerundet ausgewiesen."""
    assert ziel(
        reference_date=date(2026, 1, 1), target_date=date(2028, 1, 1)
    ).horizon_years == Decimal("2.00")


def test_prioritaet_wird_gefuehrt() -> None:
    """MUSS- und WUNSCH-Ziele sind unterscheidbar."""
    assert ziel().priority is GoalPriority.MUSS
    assert ziel(priority=GoalPriority.WUNSCH).priority is GoalPriority.WUNSCH


def test_parse_goal_meldet_fehlende_angabe() -> None:
    """``parse_goal`` übersetzt einen Schemafehler in einen fachlichen Fehler."""
    rohdaten = ziel().model_dump()
    del rohdaten["target_amount"]

    with pytest.raises(IncompleteProfileError) as fehler:
        parse_goal(rohdaten)

    assert fehler.value.field == "goal.target_amount"
