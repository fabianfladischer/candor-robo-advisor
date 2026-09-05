"""Drei durchgerechnete Beispielfälle des Systems.

Die Fälle sind so gewählt, dass sie die drei Ausgänge des Systems abdecken und
sich in genau den Größen unterscheiden, auf die es ankommt:

======================  ====================================================
``konservativ``         Risikoarme Verhältnisse, bescheidenes Ziel. Bindend ist
                        die Risikobereitschaft. Erwarteter Ausgang: Empfehlung.
``ausgewogen``          Mittlere Verhältnisse, erreichbares Ziel, zusätzlich mit
                        Nachhaltigkeitspräferenz (PAI). Erwarteter Ausgang:
                        Empfehlung mit gefiltertem Universum.
``konflikt``            Der Referenzfall des Systems: Altersvorsorge über
                        300 000 €, die unter der zulässigen Klasse nicht mit der
                        geforderten Wahrscheinlichkeit erreichbar ist.
                        Erwarteter Ausgang: keine Empfehlung.
======================  ====================================================

Die Fälle sind **fest verdrahtet und nicht zufällig erzeugt**. Sie gehen in den
Golden-Master-Test; er verlangt, dass sie
sich nicht von Lauf zu Lauf ändern.

Aufruf:
    python -m candor.report.examples --write
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Final

from candor.data.loader import snapshot_info
from candor.data.policy_loader import load_policy_set
from candor.data.universe_meta import ASSET_KEYS
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
    SustainabilityComponent,
    SustainabilityPreferences,
)
from candor.domain.goal import Goal, GoalType
from candor.domain.policy import PolicySet
from candor.engine.advise import Advice, advise
from candor.engine.convergence import load_history
from candor.engine.simulate import BlockBootstrapSampler
from candor.goals.model import build_goal

#: Stichtag aller Beispielfälle.
EXAMPLE_DATE: Final = date(2026, 7, 26)

#: Antworten mit vorsichtiger Risikobereitschaft.
ANSWERS_VORSICHTIG: Final[dict[str, str]] = {
    "verlustszenario_betrag": "S1",
    "reaktion_auf_rueckgang": "TEILWEISE_VERKAUFEN",
    "erfahrung_mit_schwankungen": "NIE_ERLEBT",
    "selbsteinschaetzung": "SICHERHEITSORIENTIERT",
    "entnahmedisziplin": "MOEGLICH",
}

#: Antworten mittlerer Ausprägung.
ANSWERS_MITTE: Final[dict[str, str]] = {
    "verlustszenario_betrag": "S3",
    "reaktion_auf_rueckgang": "ABWARTEN",
    "erfahrung_mit_schwankungen": "ERLEBT_UND_GEHALTEN",
    "selbsteinschaetzung": "AUSGEWOGEN",
    "entnahmedisziplin": "UNWAHRSCHEINLICH",
}


def _konservativ(policies: PolicySet) -> tuple[Client, Goal]:
    """Baut den konservativen Fall.

    Ein Haushalt mit auskömmlichen Verhältnissen, aber geringer
    Risikobereitschaft und wenig Erfahrung. Das Ziel ist bewusst bescheiden
    gewählt, damit es unter der niedrigen Klasse erreichbar bleibt.

    Args:
        policies: Regelwerk mit dem Zielkatalog.

    Returns:
        Kundenprofil und Anlageziel.
    """
    kunde = Client(
        client_id="K-2026-0101",
        household=Household(
            net_monthly_income=Decimal("3600.00"),
            fixed_monthly_costs=Decimal("2100.00"),
            monthly_debt_service=Decimal("0.00"),
            free_savings_amount=Decimal("450.00"),
            liquidity_reserve_months=Decimal("8.0"),
            total_debt=Decimal("0.00"),
            other_assets=Decimal("40000.00"),
            employment_type=EmploymentType.VERBEAMTET,
        ),
        knowledge=KnowledgeAndExperience(
            education_level=EducationLevel.BERUFSAUSBILDUNG,
            profession_finance_related=False,
            entries=(
                InstrumentExperience(
                    category=InstrumentCategory.GELDMARKT,
                    knowledge_level=KnowledgeLevel.GRUNDKENNTNISSE,
                    experience_years=Decimal("6.0"),
                    transactions_per_year=2,
                ),
                InstrumentExperience(
                    category=InstrumentCategory.FONDS_UCITS,
                    knowledge_level=KnowledgeLevel.GRUNDKENNTNISSE,
                    experience_years=Decimal("3.0"),
                    transactions_per_year=2,
                ),
            ),
        ),
        sustainability=SustainabilityPreferences(has_preferences=False),
        risk_attitude=RiskAttitude(answers=dict(ANSWERS_VORSICHTIG)),
    )
    ziel = build_goal(
        "Z-2026-0101-A",
        GoalType.VERMOEGENSAUFBAU,
        Decimal("58000.00"),
        EXAMPLE_DATE,
        policies.goals,
        name="Rücklage für später",
        initial_capital=Decimal("15000.00"),
        monthly_contribution=Decimal("400.00"),
    )
    return kunde, ziel


def _ausgewogen(policies: PolicySet) -> tuple[Client, Goal]:
    """Baut den ausgewogenen Fall mit Nachhaltigkeitspräferenz.

    Gewählt ist die PAI-Ausprägung. Sie ist die einzige der drei, die mit den
    Bandbreiten aller Risikoklassen vereinbar ist — Taxonomie und SFDR
    kollidieren mit den vorgeschriebenen Untergrenzen für Staatsanleihen. Der
    Fall zeigt damit einen wirksamen Filter, ohne in den Konflikt zu laufen.

    Args:
        policies: Regelwerk mit dem Zielkatalog.

    Returns:
        Kundenprofil und Anlageziel.
    """
    kunde = Client(
        client_id="K-2026-0233",
        household=Household(
            net_monthly_income=Decimal("5200.00"),
            fixed_monthly_costs=Decimal("2600.00"),
            monthly_debt_service=Decimal("250.00"),
            free_savings_amount=Decimal("900.00"),
            liquidity_reserve_months=Decimal("6.0"),
            total_debt=Decimal("12000.00"),
            other_assets=Decimal("65000.00"),
            employment_type=EmploymentType.UNBEFRISTET,
        ),
        knowledge=KnowledgeAndExperience(
            education_level=EducationLevel.HOCHSCHULABSCHLUSS,
            profession_finance_related=False,
            entries=(
                InstrumentExperience(
                    category=InstrumentCategory.ETF,
                    knowledge_level=KnowledgeLevel.ERWEITERT,
                    experience_years=Decimal("8.0"),
                    transactions_per_year=12,
                ),
                InstrumentExperience(
                    category=InstrumentCategory.AKTIEN,
                    knowledge_level=KnowledgeLevel.ERWEITERT,
                    experience_years=Decimal("6.0"),
                    transactions_per_year=8,
                ),
            ),
        ),
        sustainability=SustainabilityPreferences(
            has_preferences=True,
            pai_considered=SustainabilityComponent(requested=True, minimum_share=Decimal("0.60")),
        ),
        risk_attitude=RiskAttitude(answers=dict(ANSWERS_MITTE)),
    )
    ziel = build_goal(
        "Z-2026-0233-A",
        GoalType.IMMOBILIEN_EIGENKAPITAL,
        Decimal("95000.00"),
        EXAMPLE_DATE,
        policies.goals,
        name="Eigenkapital Immobilienkauf",
        initial_capital=Decimal("35000.00"),
        monthly_contribution=Decimal("800.00"),
    )
    return kunde, ziel


def _konflikt(policies: PolicySet) -> tuple[Client, Goal]:
    """Baut den Konfliktfall, den Referenzfall des Systems.

    Args:
        policies: Regelwerk mit dem Zielkatalog.

    Returns:
        Kundenprofil und Anlageziel.
    """
    kunde = Client(
        client_id="K-2026-0147",
        household=Household(
            net_monthly_income=Decimal("4200.00"),
            fixed_monthly_costs=Decimal("2400.00"),
            monthly_debt_service=Decimal("300.00"),
            free_savings_amount=Decimal("600.00"),
            liquidity_reserve_months=Decimal("4.0"),
            total_debt=Decimal("18000.00"),
            other_assets=Decimal("25000.00"),
            employment_type=EmploymentType.UNBEFRISTET,
        ),
        knowledge=KnowledgeAndExperience(
            education_level=EducationLevel.HOCHSCHULABSCHLUSS,
            profession_finance_related=False,
            entries=(
                InstrumentExperience(
                    category=InstrumentCategory.ETF,
                    knowledge_level=KnowledgeLevel.ERWEITERT,
                    experience_years=Decimal("5.0"),
                    transactions_per_year=6,
                ),
            ),
        ),
        sustainability=SustainabilityPreferences(has_preferences=False),
        risk_attitude=RiskAttitude(answers=dict(ANSWERS_MITTE)),
    )
    ziel = build_goal(
        "Z-2026-0147-A",
        GoalType.ALTERSVORSORGE,
        Decimal("300000.00"),
        EXAMPLE_DATE,
        policies.goals,
        initial_capital=Decimal("20000.00"),
        monthly_contribution=Decimal("500.00"),
    )
    return kunde, ziel


#: Die drei Beispielfälle mit ihren Erzeugern.
EXAMPLES: Final[dict[str, Callable[[PolicySet], tuple[Client, Goal]]]] = {
    "konservativ": _konservativ,
    "ausgewogen": _ausgewogen,
    "konflikt": _konflikt,
}


def build_example(name: str, seed: int | None = None) -> Advice:
    """Rechnet einen Beispielfall vollständig durch.

    Args:
        name: Kennung des Falls.
        seed: Abweichender Seed. Ohne Angabe der Wert aus der Policy.

    Returns:
        Das Ergebnis der Entscheidungskette.

    Raises:
        KeyError: Wenn der Fall unbekannt ist.
    """
    if name not in EXAMPLES:
        msg = f"Unbekannter Beispielfall {name!r}. Bekannt sind: {sorted(EXAMPLES)}."
        raise KeyError(msg)

    policies = load_policy_set()
    historie = load_history(policies)
    sampler = BlockBootstrapSampler(historie, policies.simulation.blocklaenge_monate)
    kunde, ziel = EXAMPLES[name](policies)

    return advise(
        kunde,
        ziel,
        policies,
        sampler,
        historie,
        seed=seed,
        snapshot_hash=str(snapshot_info().get("sha256_short", "")),
        asset_keys=ASSET_KEYS,
    )


def write_all(output_dir: Path) -> list[Path]:
    """Erzeugt alle drei Beispieldokumente.

    Für jeden Fall wird das Dokument erzeugt, das zu seinem Ausgang passt: die
    Geeignetheitserklärung bei einer Empfehlung, der Konfliktbericht sonst. Ein
    Fall, der beides oder keines liefert, ist ein Fehler.

    Args:
        output_dir: Zielverzeichnis.

    Returns:
        Die Pfade der geschriebenen Dateien.
    """
    # Verzoegert: beide Module importieren dieses hier fuer ihre
    # Kommandozeile; ein Import auf Modulebene waere ein Zirkelbezug.
    from candor.report import conflict, suitability  # noqa: PLC0415

    geschrieben: list[Path] = []
    for name in sorted(EXAMPLES):
        advice = build_example(name)
        if advice.is_recommendation:
            ziel = suitability.write_pdf(advice, output_dir / f"{name}_geeignetheitserklaerung.pdf")
        else:
            ziel = conflict.write_pdf(advice, output_dir / f"{name}_konfliktbericht.pdf")
        geschrieben.append(ziel)
    return geschrieben


def main(argv: list[str] | None = None) -> int:
    """Einstiegspunkt der Kommandozeile.

    Args:
        argv: Argumente ohne Programmnamen.

    Returns:
        ``0`` bei Erfolg.
    """
    parser = argparse.ArgumentParser(description="Beispieldokumente für den Anhang erzeugen.")
    parser.add_argument("--out", type=Path, default=Path("beispiele"))
    args = parser.parse_args(argv)

    for pfad in write_all(args.out):
        sys.stdout.write(f"Geschrieben: {pfad}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
