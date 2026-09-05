"""Faktenblatt des Referenz-Konfliktfalls.

Der Konfliktfall ist das Alleinstellungsmerkmal des Systems, und er lässt sich
nur an konkreten Zahlen beurteilen. Dieses Modul rechnet einen festen
Referenzfall durch und gibt ihn als Faktenblatt aus — Zahlen und Stichpunkte,
kein Fließtext.

Der Fall ist bewusst **unspektakulär** gewählt: ein Haushalt mit
durchschnittlichen Verhältnissen, ein übliches Altersvorsorgeziel, eine
Sparrate, die auf den ersten Blick auskömmlich wirkt. Genau dort liegt die
Aussage. Ein Konflikt entsteht nicht erst bei offensichtlicher Überforderung,
sondern schon dann, wenn Ziel und zulässiges Risiko um wenige Prozentpunkte
auseinanderliegen — und das sieht man einem Fall von außen nicht an.

Aufruf:
    python -m candor.report.factsheet [--write ZIELDATEI.md] [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Final

from candor import __version__, format_de
from candor.data.policy_loader import load_policy_set
from candor.data.universe_meta import ASSET_KEYS, get_instrument
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
from candor.domain.goal import Goal, GoalType
from candor.domain.policy import PolicySet
from candor.domain.result import Feasibility, Reconciliation
from candor.engine.convergence import load_history
from candor.engine.glidepath import apply_glidepath_for_goal
from candor.engine.reconcile import reconcile
from candor.engine.simulate import BlockBootstrapSampler
from candor.goals.model import build_goal
from candor.goals.required_return import required_return
from candor.profiling.classify import classify
from candor.trace.recorder import TraceMetadata, TraceRecorder

#: Stichtag des Referenzfalls. Fest gesetzt, damit das Faktenblatt bei jedem
#: Lauf dieselben Fristen ausweist (Prinzip P1).
REFERENCE_DATE: Final = date(2026, 7, 26)

#: Antworten des Referenzkunden zur Risikobereitschaft.
REFERENCE_ANSWERS: Final[dict[str, str]] = {
    "verlustszenario_betrag": "S3",
    "reaktion_auf_rueckgang": "ABWARTEN",
    "erfahrung_mit_schwankungen": "ERLEBT_UND_GEHALTEN",
    "selbsteinschaetzung": "AUSGEWOGEN",
    "entnahmedisziplin": "UNWAHRSCHEINLICH",
}


def reference_client() -> Client:
    """Baut den Referenzkunden des Konfliktfalls.

    Returns:
        Das Kundenprofil.
    """
    return Client(
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
        risk_attitude=RiskAttitude(answers=dict(REFERENCE_ANSWERS)),
    )


def reference_goal(policies: PolicySet) -> Goal:
    """Baut das Anlageziel des Konfliktfalls.

    Args:
        policies: Regelwerk mit dem Zielkatalog.

    Returns:
        Das Anlageziel.
    """
    return build_goal(
        "Z-2026-0147-A",
        GoalType.ALTERSVORSORGE,
        Decimal("300000.00"),
        REFERENCE_DATE,
        policies.goals,
        initial_capital=Decimal("20000.00"),
        monthly_contribution=Decimal("500.00"),
    )


def reference_conflict_case(seed: int | None = None) -> Reconciliation:
    """Rechnet den Referenzfall vollständig durch.

    Args:
        seed: Abweichender Seed. Ohne Angabe der Wert aus der Policy.

    Returns:
        Das Ergebnis des Zielabgleichs samt Konfliktbericht.
    """
    policies = load_policy_set()
    kunde = reference_client()
    ziel = reference_goal(policies)
    sampler = BlockBootstrapSampler(load_history(policies), policies.simulation.blocklaenge_monate)

    recorder = TraceRecorder(
        TraceMetadata(
            client_id=kunde.client_id,
            goal_id=ziel.goal_id,
            policy_versions=dict(policies.versions),
            package_version=__version__,
            seed=seed if seed is not None else policies.simulation.basis_seed,
        )
    )
    profil = classify(kunde, ziel, policies, recorder)
    profil = apply_glidepath_for_goal(profil, ziel, policies.glidepath, recorder)
    return reconcile(
        kunde, ziel, profil, policies, sampler, recorder, asset_keys=ASSET_KEYS, seed=seed
    )


def _de(value: Decimal | float, decimals: int = 2, *, percent: bool = False) -> str:
    """Formatiert eine Zahl mit deutschem Dezimalkomma und Tausendertrennung.

    Leitet an :mod:`candor.format_de` weiter. Die Funktion bleibt als kurze
    Schreibweise für die vielen Tabellenzellen dieses Moduls bestehen, führt aber
    keine eigene Formatierungsregel mehr: Ein zweiter Ort für dieselbe Regel war
    zuvor die Ursache dafür, dass die Konfliktdiagnose englische Dezimalpunkte
    ausgab.

    Args:
        value: Zu formatierender Wert.
        decimals: Zahl der Nachkommastellen.
        percent: Multipliziert mit 100 und hängt ein Prozentzeichen an.

    Returns:
        Der formatierte Wert.
    """
    if percent:
        return format_de.percent(value, decimals)
    return format_de.decimal(value, decimals)


def render_factsheet(result: Reconciliation, seed: int | None = None) -> str:
    """Erzeugt das Faktenblatt zum Konfliktfall.

    Args:
        result: Ergebnis des Zielabgleichs.
        seed: Verwendeter Seed, für die Reproduktionsangabe.

    Returns:
        Das Faktenblatt als Markdown.

    Raises:
        ValueError: Wenn der Fall kein Konflikt ist.
    """
    if result.conflict_report is None:
        msg = f"Der Fall endet mit dem Ausgang {result.outcome.value}, nicht im Konflikt."
        raise ValueError(msg)

    policies = load_policy_set()
    kunde = reference_client()
    ziel = reference_goal(policies)
    bericht = result.conflict_report
    rendite = required_return(ziel, policies.goals)
    haushalt = kunde.household

    zeilen = [
        "# Konfliktfall — Faktenblatt",
        "",
        "_Erzeugt durch `make factsheet`. Nicht von Hand bearbeiten — der Inhalt wird bei",
        "jedem Lauf vollständig neu gerechnet._",
        "",
        "---",
        "",
        "## 1. Ausgangslage",
        "",
        "### 1.1 Haushalt",
        "",
        "| Größe | Wert |",
        "|---|---:|",
        f"| Nettoeinkommen monatlich | {_de(haushalt.net_monthly_income)} € |",
        f"| Fixkosten monatlich | {_de(haushalt.fixed_monthly_costs)} € |",
        f"| Kapitaldienst monatlich | {_de(haushalt.monthly_debt_service)} € |",
        f"| **Freie Sparquote monatlich** | **{_de(haushalt.free_savings_amount)} €** |",
        f"| Liquiditätsreserve | {_de(haushalt.liquidity_reserve_months, 1)} Monatsausgaben |",
        f"| Bestehende Verbindlichkeiten | {_de(haushalt.total_debt)} € |",
        f"| Sonstiges Vermögen | {_de(haushalt.other_assets)} € |",
        f"| Beschäftigungsart | {haushalt.employment_type.value} |",
        "",
        "### 1.2 Anlageziel",
        "",
        "| Größe | Wert |",
        "|---|---:|",
        f"| Zieltyp | {ziel.goal_type.value} |",
        f"| Priorität | {ziel.priority.value} |",
        f"| Zielbetrag | {_de(ziel.target_amount)} € |",
        f"| Zieltermin | {ziel.target_date.strftime('%d.%m.%Y')} |",
        f"| Anlagehorizont | {ziel.horizon_months} Monate ({_de(ziel.horizon_years, 2)} Jahre) |",
        f"| Startkapital | {_de(ziel.initial_capital)} € |",
        f"| Sparrate monatlich | {_de(ziel.monthly_contribution)} € |",
        f"| Summe der Einzahlungen | {_de(rendite.value_without_return)} € |",
        "",
        "### 1.3 Erforderliche Rendite",
        "",
        "| Größe | Wert |",
        "|---|---:|",
        f"| Erforderlich nominal p. a. | {_de(rendite.nominal.annual_rate, 2, percent=True)} |",
        f"| Erforderlich real p. a. | {_de(rendite.real_annual, 2, percent=True)} |",
        f"| Kaufkrafterhaltender Zielbetrag | {_de(rendite.purchasing_power.target_amount)} € |",
        f"| Inflationsannahme | {_de(rendite.inflation_annual, 2, percent=True)} |",
        "",
        "---",
        "",
        "## 2. Risikoprofil",
        "",
        "| Achse | Ergebnis |",
        "|---|---|",
    ]

    profil_quelle = _profile_rows(result)
    zeilen.extend(profil_quelle)

    hebel_s1 = bericht.lever(bericht.levers[0].lever_id)
    zeilen += [
        "",
        "---",
        "",
        "## 3. Zielabgleich",
        "",
        "| Größe | Wert |",
        "|---|---:|",
        f"| Geforderte Erfolgswahrscheinlichkeit | "
        f"**{_de(result.required_probability, 2, percent=True)}** |",
        f"| Erreichte Erfolgswahrscheinlichkeit | "
        f"**{_de(result.achieved_probability, 2, percent=True)}** |",
        f"| Fehlender Abstand | {format_de.percentage_points(-result.margin)} |",
        f"| Pufferband des Regelwerks | {format_de.percentage_points(result.buffer_band)} |",
        f"| **Ausgang** | **{result.outcome.value}** |",
        f"| Empfehlung zulässig | {'ja' if result.allows_recommendation else 'nein'} |",
        f"| Rechtsgrundlage | {bericht.legal_basis} |",
        "",
        "### 3.1 Verteilung des Endwerts",
        "",
        "| Perzentil | Endwert |",
        "|---:|---:|",
    ]
    for stufe, wert in sorted(result.percentiles.items()):
        zeilen.append(f"| {stufe} % | {_de(wert, 0)} € |")
    zeilen += [
        f"| Zielbetrag zum Vergleich | {_de(ziel.target_amount, 0)} € |",
        "",
        f"Erwarteter Fehlbetrag **im Misserfolgsfall**: {_de(result.expected_shortfall, 0)} €. "
        "Gemittelt wird ausschließlich über die Pfade, die das Ziel verfehlen.",
        "",
        f"Medianer größter Rückgang über die Pfade: "
        f"{_de(result.median_max_drawdown, 2, percent=True)}.",
        "",
        "### 3.2 Zielallokation der zulässigen Risikoklasse",
        "",
        "| Assetklasse | Gewicht | Instrument |",
        "|---|---:|---|",
    ]
    for schluessel, gewicht in result.weights.items():
        instrument = get_instrument(schluessel)
        zeilen.append(
            f"| {instrument.label} (`{schluessel}`) | {_de(gewicht, 2, percent=True)} | "
            f"{instrument.fund_name} |"
        )

    zeilen += [
        "",
        "---",
        "",
        "## 4. Befund",
        "",
    ]
    zeilen.extend(f"- {punkt}" for punkt in bericht.diagnosis)

    zeilen += [
        "",
        "---",
        "",
        "## 5. Die drei Stellschrauben",
        "",
        "| | Stellschraube | Ist | Soll | Differenz | Erreichte Wahrsch. | Machbarkeit |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for hebel in bericht.levers:
        stellen = 0 if hebel.unit == "Monate" else 2
        zeilen.append(
            f"| {hebel.lever_id.value} | {hebel.name} "
            f"| {_de(hebel.current_value, stellen)} | {_de(hebel.required_value, stellen)} "
            f"| {_de(hebel.difference, stellen)} "
            f"| {_de(hebel.achieved_probability, 2, percent=True)} "
            f"| {hebel.feasibility.value} |"
        )
    zeilen += ["", "Einheiten: S1 in Euro je Monat, S2 in Monaten, S3 in Euro.", ""]

    for hebel in bericht.levers:
        zeilen += [
            f"**{hebel.lever_id.value} — {hebel.name}**",
            "",
            f"- {hebel.feasibility_note}",
            f"- Verfahren: {hebel.method}; {hebel.iterations} Auswertungen der Zielfunktion.",
            "",
        ]

    umsetzbar = [h.lever_id.value for h in bericht.levers if h.feasibility is Feasibility.UMSETZBAR]
    zeilen += [
        "---",
        "",
        "## 6. Einordnung",
        "",
        f"- Umsetzbare Stellschrauben: {', '.join(umsetzbar) if umsetzbar else 'keine'}.",
        f"- Die Sparrate müsste um {_de(hebel_s1.difference)} € steigen; frei verfügbar sind "
        f"{_de(haushalt.free_savings_amount)} €.",
        "- Keine der drei Stellschrauben ist eine Empfehlung. Sie sind die quantifizierte "
        "Antwort auf die Frage, was sich ändern müsste.",
        "",
        "---",
        "",
        "## 7. Reproduktion",
        "",
        "| Größe | Wert |",
        "|---|---|",
        f"| Seed | {seed if seed is not None else policies.simulation.basis_seed} |",
        f"| Pfade | {_de(float(policies.simulation.standard_pfade), 0)} |",
        f"| Blocklänge | {policies.simulation.blocklaenge_monate} Monate, zirkulär |",
        f"| Trace-Hash | `{result.trace_hash[:32]}…` |",
        f"| Trace-Ereignisse | {len(result.trace)} |",
        "| Policy-Fassungen | "
        + " · ".join(
            f"{name} {version}" for name, version in sorted(result.policy_versions.items())
        )
        + " |",
        f"| Paketversion | {__version__} |",
        "",
        "Reproduktion: `make factsheet`. Bei unverändertem Snapshot, Regelwerk und Seed ist das",
        "Ergebnis bitidentisch (Invariante I4).",
    ]
    return "\n".join(zeilen) + "\n"


def _profile_rows(result: Reconciliation) -> list[str]:
    """Bildet die Zeilen der Risikoprofil-Tabelle.

    Args:
        result: Ergebnis des Zielabgleichs.

    Returns:
        Die Tabellenzeilen.
    """
    bericht = result.conflict_report
    if bericht is None:  # pragma: no cover — im Faktenblatt stets gesetzt
        return []
    return [
        f"| **Bindende Risikoklasse** | **{result.risk_class}** |",
        f"| Bindende Achse | {bericht.binding_axis.value} |",
        "| Zielallokation | siehe Abschnitt 3.2 |",
    ]


def main(argv: list[str] | None = None) -> int:
    """Einstiegspunkt der Kommandozeile.

    Args:
        argv: Argumente ohne Programmnamen.

    Returns:
        ``0`` bei Erfolg, ``1`` wenn der Referenzfall kein Konflikt ist.
    """
    parser = argparse.ArgumentParser(description="Faktenblatt des Referenz-Konfliktfalls.")
    parser.add_argument("--seed", type=int, default=None, help="abweichender Seed")
    parser.add_argument(
        "--write", type=Path, default=None, help="Zieldatei; ohne Angabe Ausgabe auf die Konsole"
    )
    args = parser.parse_args(argv)

    ergebnis = reference_conflict_case(seed=args.seed)
    if ergebnis.conflict_report is None:
        sys.stdout.write(
            f"FEHLER — Der Referenzfall endet mit {ergebnis.outcome.value} und ist damit "
            "kein Konfliktfall.\n"
        )
        return 1

    blatt = render_factsheet(ergebnis, args.seed)
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(blatt, encoding="utf-8")
        sys.stdout.write(f"Faktenblatt geschrieben nach {args.write}\n")
    else:
        sys.stdout.write(blatt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
