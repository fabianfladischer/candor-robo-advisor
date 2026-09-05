"""Risikobereitschaft — die subjektive Achse.

Gemessen wird, welche Schwankung ein Anleger **tragen will**. Was er tragen
**kann**, ermittelt :mod:`candor.profiling.capacity`.

Warum Eurobeträge und keine Prozentangaben
------------------------------------------
Die Verlustszenarien werden dem Kunden ausschließlich als konkrete Eurobeträge
vorgelegt: nicht „ein Rückgang um 20 %“, sondern „Ihr Vermögen von 60.000 €
fällt auf 48.000 €". Dafür sprechen drei Gründe:

1. **Framing.** Dieselbe Größe wird je nach Darstellung unterschiedlich bewertet.
   „20 % Verlust“ wirkt auf viele Anleger abstrakt und beherrschbar; „12.000 €
   weniger" ist der Betrag, an dem sich die tatsächliche Reaktion entscheidet.
   Wer nach der prozentualen Angabe eine hohe Bereitschaft erklärt und beim
   Eurobetrag zurückweicht, hat nicht seine Meinung geändert — er hat die Frage
   beim ersten Mal nicht auf seine Lage bezogen.
2. **Bezug zur eigenen Lage.** Ein Prozentsatz ist für alle Kunden gleich, der
   Eurobetrag nicht. 20 % sind bei 5.000 € Anlagesumme etwas anderes als bei
   500.000 €. Die Frage nach der Tragbarkeit ist nur in der zweiten Form
   beantwortbar.
3. **Nachweisbarkeit.** Der vorgelegte Betrag steht im Trace. Im Streitfall ist
   belegt, welche Zahl der Kunde tatsächlich vor Augen hatte, als er seine
   Bereitschaft erklärte.

Die Prozentanteile stehen weiterhin im Regelwerk — sie werden gebraucht, um aus
der Bezugsgröße die Beträge zu berechnen. Der Kunde sieht sie nie.

Bezugsgröße ist das über die Laufzeit eingezahlte Kapital, undiskontiert. Der
daraus berechnete **Verlustbetrag** geht in keine Bewertung ein — die Punkte
hängen allein an der gewählten Stufe S0 bis S5, nicht an ihrer Beschriftung.

Die **Bezugsgröße selbst** ist dagegen sehr wohl bewertungsrelevant, nur auf der
anderen Achse: :func:`candor.profiling.scoring.planned_investment` ist zugleich
Zähler des Klumpenrisikos in :mod:`candor.profiling.capacity`. Ein pauschaler
Satz, der Wert diene „allein der Darstellung und gehe in keine Bewertung ein",
ist für den Verlustbetrag richtig und für die Bezugsgröße falsch.

Widersprüche werden ausgewiesen, nicht weggerechnet
---------------------------------------------------
Wer sich als chancenorientiert bezeichnet und zugleich schon fünf Prozent
Rückgang nicht aushalten würde, hat nicht „im Mittel eine mittlere
Risikobereitschaft". Er hat zwei unvereinbare Angaben gemacht, und die
wahrscheinlichste Erklärung ist, dass die Selbsteinschätzung nicht trägt.

Eine Mittelung würde den Widerspruch zum Verschwinden bringen und ein Ergebnis
erzeugen, das keine der beiden Angaben abbildet. Deshalb: Der Widerspruch wird
als Befund im Trace und im Ergebnis geführt, die Punktzahl bleibt unverändert.
Die Auflösung ist eine Aufgabe des Beraters, nicht des Algorithmus — genau das
ist die Haltung, die diesem Projekt den Namen gibt.

Die Punktzahl bleibt aus einem zweiten Grund unangetastet: Ein Abschlag für
Widersprüche würde die Monotonie der Achse brechen (Invariante I2). Höhere
Antwortpunkte könnten dann zu einer niedrigeren Klasse führen, und die Achse
wäre nicht mehr prüfbar.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Final

from candor.domain.client import Client
from candor.domain.errors import IncompleteProfileError
from candor.domain.goal import Goal
from candor.domain.policy import TolerancePolicy
from candor.domain.result import CriterionScore, Inconsistency, ToleranceResult
from candor.profiling.scoring import (
    planned_investment,
    resolve_option,
    risk_class_for_points,
    total_points,
    weighted,
)
from candor.trace.events import Rule
from candor.trace.recorder import TraceRecorder

#: Rechtsgrundlage der subjektiven Achse.
LEGAL_BASIS: Final = "Art. 54 Abs. 2 Buchst. a und Abs. 5 DelVO (EU) 2017/565"

#: Rundung der dargestellten Eurobeträge.
EURO_QUANT: Final = Decimal("0.01")

RULE_SCENARIOS = Rule(
    rule_id="TOL-SCEN",
    description=(
        "Aus der Bezugsgröße werden die Stufen der Verlustszenario-Frage als konkrete "
        "Eurobeträge berechnet und dem Kunden ausschließlich in dieser Form vorgelegt. "
        "Für diese Frage wird keine Prozentangabe gezeigt. Die übrigen Items der "
        "Bereitschaftserhebung nennen weiterhin die festen Prozentwerte, die das "
        "Regelwerk für sie vorgibt."
    ),
    legal_basis=LEGAL_BASIS,
)
RULE_ITEM = Rule(
    rule_id="TOL-ITEM",
    description=(
        "Die Antwort auf ein Item der Risikobereitschaft wird über die Ausprägungen "
        "der zugehörigen Dimension bewertet."
    ),
    legal_basis=LEGAL_BASIS,
)
RULE_CONFLICT = Rule(
    rule_id="TOL-CONFLICT",
    description=(
        "Eine Konsistenzregel des Regelwerks hat einen Widerspruch zwischen zwei "
        "Antworten erkannt. Der Befund wird ausgewiesen; die Punktzahl bleibt "
        "unverändert."
    ),
    legal_basis=LEGAL_BASIS,
)
RULE_CLASS = Rule(
    rule_id="TOL-CLASS",
    description=(
        "Die gewichtete Gesamtpunktzahl wird über die Klassenschwellen des Regelwerks "
        "auf die Risikoklasse der Bereitschaft abgebildet."
    ),
    legal_basis=LEGAL_BASIS,
)


def scenario_amounts(goal: Goal, policy: TolerancePolicy) -> dict[str, Decimal]:
    """Berechnet die dem Kunden vorzulegenden Verlustbeträge in Euro.

    Args:
        goal: Anlageziel, aus dem sich die Bezugsgröße ergibt.
        policy: Punktetabelle der Bereitschaft mit den Szenariostufen.

    Returns:
        Zuordnung Stufenkennung auf den Verlustbetrag in Euro.
    """
    bezugsbetrag = planned_investment(goal)
    return {
        stufe.id: (bezugsbetrag * stufe.verlust_anteil).quantize(EURO_QUANT, rounding=ROUND_HALF_UP)
        for stufe in policy.verlustszenarien.stufen
    }


def find_inconsistencies(
    answers: dict[str, str], policy: TolerancePolicy
) -> tuple[Inconsistency, ...]:
    """Prüft die Antworten gegen die Konsistenzregeln des Regelwerks.

    Args:
        answers: Antworten des Kunden je Item.
        policy: Punktetabelle mit den Konsistenzregeln.

    Returns:
        Die erkannten Widersprüche in Reihenfolge des Regelwerks.
    """
    befunde: list[Inconsistency] = []
    for regel in policy.konsistenzregeln:
        beteiligt = {
            dimension: answers[dimension] for dimension in regel.bedingungen if dimension in answers
        }
        if len(beteiligt) != len(regel.bedingungen):
            continue
        if all(answers[dimension] in optionen for dimension, optionen in regel.bedingungen.items()):
            befunde.append(
                Inconsistency(
                    rule_id=regel.id,
                    description=regel.beschreibung,
                    answers=beteiligt,
                )
            )
    return tuple(befunde)


def assess_tolerance(
    client: Client,
    goal: Goal,
    policy: TolerancePolicy,
    recorder: TraceRecorder,
) -> ToleranceResult:
    """Ermittelt die Risikobereitschaft und protokolliert jeden Schritt.

    Args:
        client: Geprüftes Kundenprofil mit den Antworten zur Risikobereitschaft.
        goal: Geprüftes Anlageziel, liefert die Bezugsgröße der Szenarien.
        policy: Punktetabelle der Bereitschaft.
        recorder: Recorder, in den jedes Item ein Ereignis schreibt.

    Returns:
        Das Ergebnis der subjektiven Achse.

    Raises:
        IncompleteProfileError: Wenn zu einem Item des Regelwerks keine Antwort
            vorliegt. Eine fehlende Antwort wird nicht mit null Punkten
            bewertet — das wäre eine erfundene Angabe (Prinzip P5).
        PolicyError: Wenn eine Antwort im Regelwerk nicht vorgesehen ist.
    """
    antworten = dict(client.risk_attitude.answers)
    betraege = scenario_amounts(goal, policy)
    bezugsbetrag = planned_investment(goal)

    recorder.record(
        RULE_SCENARIOS,
        {
            "bezugsgroesse": policy.verlustszenarien.bezugsgroesse,
            "bezugsbetrag_eur": bezugsbetrag,
            "dimension": policy.verlustszenarien.dimension,
        },
        {"verlustbetraege_eur": betraege},
        policy.version,
    )

    fehlend = sorted(set(policy.dimensionen) - set(antworten))
    if fehlend:
        raise IncompleteProfileError(
            f"risk_attitude.answers.{fehlend[0]}",
            f"Zu {len(fehlend)} Item(s) der Risikobereitschaft liegt keine Antwort vor: "
            f"{', '.join(fehlend)}.",
            LEGAL_BASIS,
        )
    unbekannt = sorted(set(antworten) - set(policy.dimensionen))
    if unbekannt:
        raise IncompleteProfileError(
            f"risk_attitude.answers.{unbekannt[0]}",
            f"Antworten zu unbekannten Items: {', '.join(unbekannt)}. Das Regelwerk "
            f"{policy.version} kennt diese Items nicht.",
            LEGAL_BASIS,
        )

    ergebnisse: list[CriterionScore] = []
    for name in policy.dimensionen:
        dimension = policy.dimensionen[name]
        antwort = antworten[name]
        punkte, grundlage = resolve_option(dimension, antwort, name)
        beitrag = weighted(punkte, dimension.gewicht)

        eingaben: dict[str, object] = {
            "item": name,
            "antwort": antwort,
            "gewicht": dimension.gewicht,
        }
        if name == policy.verlustszenarien.dimension:
            eingaben["vorgelegter_verlustbetrag_eur"] = betraege[antwort]

        recorder.record(
            RULE_ITEM,
            eingaben,
            {"punkte": punkte, "gewichteter_beitrag": beitrag},
            policy.version,
        )
        ergebnisse.append(
            CriterionScore(
                criterion=name,
                measured_value=Decimal(0),
                unit="Antwortoption",
                points=punkte,
                weight=dimension.gewicht,
                weighted_points=beitrag,
                basis=grundlage,
            )
        )

    widersprueche = find_inconsistencies(antworten, policy)
    for befund in widersprueche:
        recorder.record(
            RULE_CONFLICT,
            {"konsistenzregel": befund.rule_id, "beteiligte_antworten": befund.answers},
            {
                "widerspruch": befund.description,
                "punktzahl_angepasst": False,
                "hinweis": (
                    "Der Widerspruch wird ausgewiesen, nicht weggerechnet. Die Auflösung "
                    "ist Aufgabe des Beraters."
                ),
            },
            policy.version,
        )

    summe = total_points([eintrag.weighted_points for eintrag in ergebnisse])
    klasse = risk_class_for_points(policy.klassenschwellen, summe)

    recorder.record(
        RULE_CLASS,
        {"gesamtpunkte": summe, "anzahl_widersprueche": len(widersprueche)},
        {"risikoklasse": klasse},
        policy.version,
    )

    return ToleranceResult(
        risk_class=klasse,
        total_points=summe,
        scores=tuple(ergebnisse),
        inconsistencies=widersprueche,
        scenario_amounts=betraege,
        reference_amount=bezugsbetrag,
        presented_amount=betraege[antworten[policy.verlustszenarien.dimension]],
        policy_version=policy.version,
    )
