"""Gemeinsame Bausteine für Kundenprofile und Anlageziele in den Tests.

Die Bausteine liefern einen tragfähigen Grundfall, von dem einzelne Felder über
Schlüsselwortargumente abweichen können. So bleibt in jedem Test sichtbar,
welche **eine** Angabe den geprüften Unterschied ausmacht.

Der Grundfall ist bewusst unauffällig gewählt: Er löst kein K.-o.-Kriterium aus,
enthält keine Widersprüche und liegt in keiner Dimension am Rand. Tests, die
Randfälle prüfen, weichen ausdrücklich davon ab.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Final

from candor.domain.client import (
    Client,
    EducationLevel,
    EmploymentType,
    Household,
    InstrumentCategory,
    InstrumentExperience,
    KnowledgeAndExperience,
    KnowledgeLevel,
    RiskAttitude,
    SustainabilityPreferences,
)
from candor.domain.goal import Goal, GoalPriority

#: Stichtag aller Testfälle. Fest gewählt, damit Ergebnisse reproduzierbar sind.
STICHTAG: Final = date(2026, 7, 26)

#: Widerspruchsfreie Antworten mittlerer Ausprägung (70,00 Punkte, Klasse 4).
ANTWORTEN_MITTE: Final[dict[str, str]] = {
    "verlustszenario_betrag": "S3",
    "reaktion_auf_rueckgang": "ABWARTEN",
    "erfahrung_mit_schwankungen": "ERLEBT_UND_GEHALTEN",
    "selbsteinschaetzung": "AUSGEWOGEN",
    "entnahmedisziplin": "UNWAHRSCHEINLICH",
}

#: Antworten mit durchgehend niedriger Bereitschaft (Klasse 1).
ANTWORTEN_NIEDRIG: Final[dict[str, str]] = {
    "verlustszenario_betrag": "S0",
    "reaktion_auf_rueckgang": "SOFORT_VERKAUFEN",
    "erfahrung_mit_schwankungen": "ERLEBT_UND_VERKAUFT",
    "selbsteinschaetzung": "SEHR_SICHERHEITSORIENTIERT",
    "entnahmedisziplin": "SEHR_WAHRSCHEINLICH",
}

#: Antworten mit durchgehend hoher Bereitschaft (Klasse 5).
ANTWORTEN_HOCH: Final[dict[str, str]] = {
    "verlustszenario_betrag": "S5",
    "reaktion_auf_rueckgang": "NACHKAUFEN",
    "erfahrung_mit_schwankungen": "ERLEBT_UND_NACHGEKAUFT",
    "selbsteinschaetzung": "CHANCENORIENTIERT",
    "entnahmedisziplin": "AUSGESCHLOSSEN",
}


def haushalt(**abweichungen: Any) -> Household:
    """Baut einen tragfähigen Haushalt.

    Args:
        **abweichungen: Felder, die vom Grundfall abweichen sollen.

    Returns:
        Der erzeugte Haushalt.
    """
    felder: dict[str, Any] = {
        "net_monthly_income": Decimal("4200.00"),
        "fixed_monthly_costs": Decimal("2400.00"),
        "monthly_debt_service": Decimal("300.00"),
        "free_savings_amount": Decimal("600.00"),
        "liquidity_reserve_months": Decimal("4.0"),
        "total_debt": Decimal("18000.00"),
        "other_assets": Decimal("25000.00"),
        "employment_type": EmploymentType.UNBEFRISTET,
    }
    felder.update(abweichungen)
    return Household(**felder)


def kenntnisse(**abweichungen: Any) -> KnowledgeAndExperience:
    """Baut einen Kenntnisstand, der die Gattung ETF freischaltet.

    Args:
        **abweichungen: Felder, die vom Grundfall abweichen sollen.

    Returns:
        Der erzeugte Kenntnisstand.
    """
    felder: dict[str, Any] = {
        "education_level": EducationLevel.HOCHSCHULABSCHLUSS,
        "profession_finance_related": False,
        "entries": (
            InstrumentExperience(
                category=InstrumentCategory.ETF,
                knowledge_level=KnowledgeLevel.ERWEITERT,
                experience_years=Decimal("5.0"),
                transactions_per_year=6,
            ),
        ),
    }
    felder.update(abweichungen)
    return KnowledgeAndExperience(**felder)


def kenntnisse_fuer_klasse(klasse: int) -> KnowledgeAndExperience:
    """Baut einen Kenntnisstand, der genau die gewünschte Klasse freischaltet.

    Args:
        klasse: Angestrebte Obergrenze von 1 bis 5.

    Returns:
        Der erzeugte Kenntnisstand.
    """
    gattung = {
        1: InstrumentCategory.GELDMARKT,
        2: InstrumentCategory.ANLEIHEN_IG,
        3: InstrumentCategory.FONDS_UCITS,
        4: InstrumentCategory.ETF,
        5: InstrumentCategory.AKTIEN,
    }[klasse]
    stufe = {
        1: KnowledgeLevel.GRUNDKENNTNISSE,
        2: KnowledgeLevel.GRUNDKENNTNISSE,
        3: KnowledgeLevel.GRUNDKENNTNISSE,
        4: KnowledgeLevel.ERWEITERT,
        5: KnowledgeLevel.ERWEITERT,
    }[klasse]
    return kenntnisse(
        entries=(
            InstrumentExperience(
                category=gattung,
                knowledge_level=stufe,
                experience_years=Decimal("10.0"),
                transactions_per_year=20,
            ),
        )
    )


def kunde(**abweichungen: Any) -> Client:
    """Baut ein vollständiges Kundenprofil.

    Args:
        **abweichungen: Felder, die vom Grundfall abweichen sollen.

    Returns:
        Das erzeugte Kundenprofil.
    """
    felder: dict[str, Any] = {
        "client_id": "K-0001",
        "household": haushalt(),
        "knowledge": kenntnisse(),
        "sustainability": SustainabilityPreferences(has_preferences=False),
        "risk_attitude": RiskAttitude(answers=dict(ANTWORTEN_MITTE)),
    }
    felder.update(abweichungen)
    return Client(**felder)


def ziel(**abweichungen: Any) -> Goal:
    """Baut ein zulässiges Anlageziel mit rund zwanzig Jahren Horizont.

    Args:
        **abweichungen: Felder, die vom Grundfall abweichen sollen.

    Returns:
        Das erzeugte Ziel.
    """
    felder: dict[str, Any] = {
        "goal_id": "Z-0001",
        "name": "Ruhestand",
        "priority": GoalPriority.MUSS,
        "reference_date": STICHTAG,
        "target_date": date(2046, 7, 1),
        "target_amount": Decimal("250000.00"),
        "initial_capital": Decimal("20000.00"),
        "monthly_contribution": Decimal("500.00"),
    }
    felder.update(abweichungen)
    return Goal(**felder)


def add_months(start: date, months: int) -> date:
    """Addiert Monate auf ein Datum und behält den Tag bei, soweit möglich.

    Args:
        start: Ausgangsdatum.
        months: Zahl der zu addierenden Monate.

    Returns:
        Das verschobene Datum; der Tag wird auf den 1. gesetzt, damit kein
        Monatsende-Sonderfall entsteht.
    """
    gesamt = start.year * 12 + (start.month - 1) + months
    return date(gesamt // 12, gesamt % 12 + 1, 1)
