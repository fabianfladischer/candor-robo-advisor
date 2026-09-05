"""Zielkatalog und Mehrzielfähigkeit.

Zwei Aufgaben:

1. **Vorbelegungen.** Ein Kunde nennt „Altersvorsorge“, nicht „300 Monate,
   MUSS-Ziel, 90 % Mindesterfolgswahrscheinlichkeit". Der Zielkatalog in
   ``policies/goals_v1.yaml`` übersetzt das eine ins andere. Jede Vorbelegung
   ist überschreibbar; keine steht im Code.
2. **Zuteilung der Sparrate auf mehrere Ziele.** Ein Haushalt hat einen
   monatlichen Überschuss und meist mehr als ein Ziel.

Die Zuteilungsreihenfolge
-------------------------
Sortiert wird nach drei Kriterien, in dieser Reihenfolge:

1. **Priorität** — MUSS vor WUNSCH, gemäß ``zuteilungsreihenfolge`` im
   Regelwerk. Ein MUSS-Ziel wird vollständig bedient, bevor ein WUNSCH-Ziel den
   ersten Euro erhält. Begründung: Die Unterscheidung MUSS/WUNSCH bedeutet, dass
   eine Verfehlung des einen wesentliche Folgen hat und die des anderen tragbar
   ist. Eine anteilige Verteilung würde beide Ziele gleich behandeln und damit
   die Unterscheidung aufheben.
2. **Zieltermin** — der frühere zuerst. Ein Ziel in drei Jahren kann eine
   Unterfinanzierung nicht mehr aufholen, ein Ziel in dreißig Jahren schon.
   Zeitnähe ist der Knappheitsgrund, nicht die Zielhöhe.
3. **Zielkennung** — alphabetisch, ausschließlich zur Sicherung der
   Reproduzierbarkeit. Ohne dieses dritte Kriterium wäre die Reihenfolge bei
   gleicher Priorität und gleichem Termin von der Eingabereihenfolge abhängig
   und das Ergebnis nicht mehr bitidentisch (Prinzip P1).

Ein Ziel wird **vollständig oder teilweise** bedient, aber nie übersprungen, um
ein späteres vollständig zu bedienen. Wer das Budget nicht mehr voll erhält,
bekommt den Rest; die Unterdeckung wird ausgewiesen. Die Alternative — kleinere
Ziele bevorzugen, um mehr davon vollständig zu bedienen — würde die
Prioritätsreihenfolge über die Hintertür aushebeln.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Final

from candor.domain.goal import Goal, GoalPriority, GoalType
from candor.domain.policy import GoalPolicy, GoalTemplate
from candor.domain.result import AllocationResult, GoalAllocation
from candor.trace.events import Rule
from candor.trace.recorder import TraceRecorder

#: Rechtsgrundlage der Zielerhebung.
LEGAL_BASIS: Final = "Art. 54 Abs. 2 Buchst. a DelVO (EU) 2017/565"

#: Monate je Jahr.
MONTHS_PER_YEAR: Final = 12

RULE_TEMPLATE = Rule(
    rule_id="GOAL-TPL",
    description=(
        "Aus dem Zieltyp werden Standardhorizont, Standardpriorität und "
        "Mindest-Erfolgswahrscheinlichkeit aus dem Zielkatalog übernommen, soweit sie "
        "nicht ausdrücklich überschrieben wurden."
    ),
    legal_basis=LEGAL_BASIS,
)
RULE_ALLOCATION = Rule(
    rule_id="GOAL-ALLOC",
    description=(
        "Die verfügbare Sparrate wird auf die Ziele verteilt: MUSS vor WUNSCH, "
        "innerhalb derselben Priorität der frühere Zieltermin zuerst, bei Gleichstand "
        "nach Zielkennung. Ein Ziel wird vollständig oder teilweise bedient, nie "
        "übersprungen."
    ),
    legal_basis=LEGAL_BASIS,
)


def add_months(start: date, months: int) -> date:
    """Verschiebt ein Datum um eine Zahl von Monaten.

    Reine Kalenderrechnung. Der Tag wird auf den Monatsersten gesetzt, damit
    keine Monatsende-Sonderfälle entstehen — der 31. Januar plus einen Monat ist
    kalendarisch nicht eindeutig.

    Args:
        start: Ausgangsdatum.
        months: Zahl der zu addierenden Monate, nicht negativ.

    Returns:
        Das verschobene Datum.

    Raises:
        ValueError: Wenn die Monatszahl negativ ist.
    """
    if months < 0:
        msg = f"Die Monatszahl darf nicht negativ sein, ist aber {months}."
        raise ValueError(msg)
    gesamt = start.year * MONTHS_PER_YEAR + (start.month - 1) + months
    return date(gesamt // MONTHS_PER_YEAR, gesamt % MONTHS_PER_YEAR + 1, 1)


def template_for(goal_type: GoalType, policy: GoalPolicy) -> GoalTemplate:
    """Liefert die Vorbelegungen eines Zieltyps.

    Args:
        goal_type: Der Zieltyp.
        policy: Zielpolicy mit dem Katalog.

    Returns:
        Die Vorbelegungen.

    Raises:
        KeyError: Wenn der Zieltyp im Katalog fehlt. Das Regelwerksmodell
            schließt das aus; die Prüfung bleibt als Absicherung bestehen.
    """
    vorlage = policy.zieltypen.get(goal_type.name)
    if vorlage is None:  # pragma: no cover — durch das Modell ausgeschlossen
        msg = f"Zieltyp {goal_type.name!r} fehlt im Zielkatalog."
        raise KeyError(msg)
    return vorlage


def build_goal(  # noqa: PLR0913 — jedes Feld ist eine eigene fachliche Angabe
    goal_id: str,
    goal_type: GoalType,
    target_amount: Decimal,
    reference_date: date,
    policy: GoalPolicy,
    *,
    name: str | None = None,
    target_date: date | None = None,
    priority: GoalPriority | None = None,
    initial_capital: Decimal = Decimal("0.00"),
    monthly_contribution: Decimal = Decimal("0.00"),
    required_success_probability: Decimal | None = None,
    recorder: TraceRecorder | None = None,
) -> Goal:
    """Legt ein Anlageziel mit den Vorbelegungen seines Zieltyps an.

    Alle Vorbelegungen lassen sich überschreiben. Wird ein Wert übergeben, gilt
    er; sonst greift der Zielkatalog.

    Args:
        goal_id: Fachliche Kennung des Ziels.
        goal_type: Typ des Ziels.
        target_amount: Angestrebter Betrag zum Zieltermin.
        reference_date: Stichtag der Erhebung.
        policy: Zielpolicy mit dem Katalog.
        name: Abweichende Bezeichnung. Ohne Angabe die des Zieltyps.
        target_date: Abweichender Zieltermin. Ohne Angabe Stichtag zuzüglich
            Standardhorizont.
        priority: Abweichende Priorität. Ohne Angabe die des Zieltyps.
        initial_capital: Bereits vorhandenes Kapital.
        monthly_contribution: Geplante monatliche Sparrate.
        required_success_probability: Abweichende Mindestwahrscheinlichkeit.
            Sie kann die Anforderung nur anheben; die Absicherung erfolgt beim
            Nachschlagen in der Policy.
        recorder: Optionaler Recorder für den Trace.

    Returns:
        Das geprüfte Anlageziel.
    """
    vorlage = template_for(goal_type, policy)
    termin = target_date or add_months(reference_date, vorlage.standardhorizont_monate)
    verbindlichkeit = priority or vorlage.standardprioritaet

    ziel = Goal(
        goal_id=goal_id,
        name=name or vorlage.bezeichnung,
        goal_type=goal_type,
        priority=verbindlichkeit,
        reference_date=reference_date,
        target_date=termin,
        target_amount=target_amount,
        initial_capital=initial_capital,
        monthly_contribution=monthly_contribution,
        required_success_probability=required_success_probability,
    )

    if recorder is not None:
        recorder.record(
            RULE_TEMPLATE,
            {
                "zieltyp": goal_type,
                "standardhorizont_monate": vorlage.standardhorizont_monate,
                "standardprioritaet": vorlage.standardprioritaet,
                "termin_ueberschrieben": target_date is not None,
                "prioritaet_ueberschrieben": priority is not None,
            },
            {
                "zieltermin": termin,
                "prioritaet": verbindlichkeit,
                "horizont_monate": ziel.horizon_months,
                "mindest_erfolgswahrscheinlichkeit": policy.minimum_success_probability(
                    verbindlichkeit, required_success_probability, goal_type
                ),
            },
            policy.version,
        )

    return ziel


def allocation_order(goals: tuple[Goal, ...], policy: GoalPolicy) -> tuple[Goal, ...]:
    """Sortiert Ziele nach der Zuteilungsreihenfolge.

    Args:
        goals: Die zu sortierenden Ziele.
        policy: Zielpolicy mit der Prioritätsreihenfolge.

    Returns:
        Die Ziele in Zuteilungsreihenfolge.
    """
    rang = {name: index for index, name in enumerate(policy.zuteilungsreihenfolge)}
    return tuple(
        sorted(
            goals,
            key=lambda ziel: (
                rang.get(ziel.priority.name, len(rang)),
                ziel.target_date,
                ziel.goal_id,
            ),
        )
    )


def allocate_savings(
    goals: tuple[Goal, ...],
    monthly_budget: Decimal,
    policy: GoalPolicy,
    recorder: TraceRecorder | None = None,
) -> AllocationResult:
    """Verteilt die verfügbare Sparrate auf mehrere Ziele.

    Args:
        goals: Die Ziele des Kunden.
        monthly_budget: Monatlich verfügbarer Betrag.
        policy: Zielpolicy mit der Zuteilungsreihenfolge.
        recorder: Optionaler Recorder für den Trace.

    Returns:
        Die Zuteilung je Ziel samt Unterdeckungen.

    Raises:
        ValueError: Wenn keine Ziele übergeben werden oder das Budget negativ
            ist. Beides ist keine sinnvolle Eingabe.
    """
    if not goals:
        msg = "Es wurden keine Ziele zur Zuteilung übergeben."
        raise ValueError(msg)
    if monthly_budget < 0:
        msg = f"Das Budget darf nicht negativ sein, ist aber {monthly_budget}."
        raise ValueError(msg)

    sortiert = allocation_order(goals, policy)
    verbleibend = monthly_budget
    zuteilungen: list[GoalAllocation] = []

    for rang, ziel in enumerate(sortiert, start=1):
        gewuenscht = ziel.monthly_contribution
        zugeteilt = min(gewuenscht, verbleibend)
        verbleibend -= zugeteilt
        zuteilungen.append(
            GoalAllocation(
                goal_id=ziel.goal_id,
                priority=ziel.priority.name,
                rank=rang,
                requested=gewuenscht,
                allocated=zugeteilt,
            )
        )

    ergebnis = AllocationResult(
        budget=monthly_budget,
        allocations=tuple(zuteilungen),
        policy_version=policy.version,
    )

    if recorder is not None:
        recorder.record(
            RULE_ALLOCATION,
            {
                "budget_monatlich": monthly_budget,
                "reihenfolge": [ziel.goal_id for ziel in sortiert],
                "prioritaetsreihenfolge": list(policy.zuteilungsreihenfolge),
                "gewuenscht_gesamt": sum(
                    (ziel.monthly_contribution for ziel in sortiert), Decimal(0)
                ),
            },
            {
                "zuteilungen": {
                    eintrag.goal_id: eintrag.allocated for eintrag in ergebnis.allocations
                },
                "zugeteilt_gesamt": ergebnis.allocated_total,
                "nicht_zugeteilt": ergebnis.unallocated,
                "unterdeckt": list(ergebnis.underfunded),
                "vollstaendig": ergebnis.is_complete,
            },
            policy.version,
        )

    return ergebnis


def apply_allocation(goals: tuple[Goal, ...], result: AllocationResult) -> tuple[Goal, ...]:
    """Setzt die zugeteilten Sparraten in die Ziele ein.

    Ziele, die weder Startkapital noch eine zugeteilte Sparrate haben, werden
    **weggelassen**: Für sie besteht kein Weg zum Ziel, und ein Anlageziel ohne
    jeden Weg dorthin ist keine tragfähige Grundlage für eine Empfehlung
    (Prinzip P5). Sie erscheinen weiterhin in der Zuteilung als unterdeckt.

    Args:
        goals: Die ursprünglichen Ziele.
        result: Ergebnis der Zuteilung.

    Returns:
        Die Ziele mit angepasster Sparrate, in Zuteilungsreihenfolge.
    """
    nach_kennung = {ziel.goal_id: ziel for ziel in goals}
    angepasst: list[Goal] = []
    for eintrag in result.allocations:
        ziel = nach_kennung[eintrag.goal_id]
        if eintrag.allocated == 0 and ziel.initial_capital == 0:
            continue
        angepasst.append(
            Goal.model_validate({**ziel.model_dump(), "monthly_contribution": eintrag.allocated})
        )
    return tuple(angepasst)
