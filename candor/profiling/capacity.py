"""Risikotragfähigkeit — die objektive Achse.

Gemessen wird, welche Schwankung ein Haushalt **tragen kann**, unabhängig davon,
welche er tragen **will**. Die zweite Frage beantwortet
:mod:`candor.profiling.tolerance`; verknüpft werden beide über ein bindendes
Minimum in :mod:`candor.profiling.classify`.

Sechs Kriterien, Struktur angelehnt an DIN 77230 (Basis-Finanzanalyse für
Privathaushalte). Die Norm wurde nicht eingesehen; übernommen ist ausschließlich
die Idee, die finanzielle Belastbarkeit über mehrere getrennte Dimensionen statt
über eine Kennzahl zu erfassen. Sämtliche Punktwerte stammen aus
``policies/capacity_v1.yaml``.

Das K.-o.-Kriterium
-------------------
Liegt die Liquiditätsreserve unter der im Regelwerk gesetzten Mindestreserve,
gilt Risikoklasse 1 — **ohne Verrechnung mit den übrigen fünf Kriterien**. Der
Grund ist kein Vorsichtsreflex, sondern ein Zusammenhang: Wer keine Rücklage hat,
muss im Notfall die Anlage auflösen. Der Notfall tritt typischerweise dann ein,
wenn auch die Märkte fallen — Kurzarbeit und Arbeitsplatzverlust häufen sich in
der Rezession. Der erzwungene Verkauf fällt damit gerade in die Phase, in der er
am teuersten ist.

Ein hohes Einkommen oder ein langer Horizont können das nicht ausgleichen: Sie
beschreiben Verhältnisse, die im Notfall gerade nicht mehr gelten. Eine
Verrechnung würde genau diese Abhängigkeit unterschlagen. Deshalb ein Ausschluss
und keine Punktabwertung.

Die übrigen fünf Kriterien werden auch bei ausgelöstem K.-o. vollständig
ausgewertet und im Trace festgehalten. Der Kunde soll sehen, was ohne die
fehlende Rücklage möglich gewesen wäre — das macht die Rücklage zur konkreten,
erreichbaren Handlungsempfehlung statt zu einer bloßen Absage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from candor.domain.client import Client
from candor.domain.errors import PolicyError
from candor.domain.goal import Goal
from candor.domain.policy import CapacityPolicy
from candor.domain.result import CapacityResult, CriterionScore
from candor.profiling.scoring import (
    planned_investment,
    resolve_band,
    resolve_option,
    risk_class_for_points,
    share,
    total_points,
    weighted,
)
from candor.trace.events import Rule
from candor.trace.recorder import TraceRecorder

#: Rechtsgrundlage der gesamten Achse.
LEGAL_BASIS: Final = "Art. 54 Abs. 2 Buchst. b und Abs. 4 DelVO (EU) 2017/565"

RULE_KO = Rule(
    rule_id="CAP-KO-LIQ",
    description=(
        "Liegt die Liquiditätsreserve unter der Mindestreserve des Regelwerks, wird "
        "Risikoklasse 1 gesetzt — ohne Verrechnung mit den übrigen Kriterien."
    ),
    legal_basis=LEGAL_BASIS,
)
RULE_CLASS = Rule(
    rule_id="CAP-CLASS",
    description=(
        "Die gewichtete Gesamtpunktzahl wird über die Klassenschwellen des Regelwerks "
        "auf die Risikoklasse der Tragfähigkeit abgebildet; ein ausgelöstes "
        "K.-o.-Kriterium überschreibt das Ergebnis."
    ),
    legal_basis=LEGAL_BASIS,
)


@dataclass(frozen=True, slots=True)
class _Criterion:
    """Ein Bewertungskriterium der Tragfähigkeit.

    Attributes:
        key: Schlüssel der Dimension im Regelwerk.
        rule: Regel, unter der das Kriterium in den Trace geht.
        unit: Maßeinheit des Messwerts.
        measure: Ermittelt den Messwert aus Kunde und Ziel.
        categorical: Ob die Bewertung über Ausprägungen statt Bänder erfolgt.
    """

    key: str
    rule: Rule
    unit: str
    measure: Callable[[Client, Goal], Decimal]
    categorical: bool = False


def _liquiditaetsreserve(client: Client, goal: Goal) -> Decimal:  # noqa: ARG001
    """Liefert die Liquiditätsreserve in Monatsausgaben.

    Args:
        client: Kundenprofil.
        goal: Anlageziel, hier ohne Bedeutung.

    Returns:
        Die Reserve in Monatsausgaben.
    """
    return client.household.liquidity_reserve_months


def _freie_sparquote(client: Client, goal: Goal) -> Decimal:  # noqa: ARG001
    """Liefert die freie Sparquote als Anteil am Nettoeinkommen.

    Args:
        client: Kundenprofil.
        goal: Anlageziel, hier ohne Bedeutung.

    Returns:
        Der Anteil zwischen 0 und 1.
    """
    return share(client.household.free_savings_amount, client.household.net_monthly_income)


def _anlagehorizont(client: Client, goal: Goal) -> Decimal:  # noqa: ARG001
    """Liefert den Anlagehorizont in Monaten.

    Args:
        client: Kundenprofil, hier ohne Bedeutung.
        goal: Anlageziel.

    Returns:
        Der Horizont in vollen Monaten.
    """
    return Decimal(goal.horizon_months)


def _verschuldungsgrad(client: Client, goal: Goal) -> Decimal:  # noqa: ARG001
    """Liefert den Kapitaldienst als Anteil am Nettoeinkommen.

    Args:
        client: Kundenprofil.
        goal: Anlageziel, hier ohne Bedeutung.

    Returns:
        Der Anteil zwischen 0 und 1.
    """
    return share(client.household.monthly_debt_service, client.household.net_monthly_income)


def _vermoegensanteil(client: Client, goal: Goal) -> Decimal:
    """Liefert den Anteil der geplanten Anlage am Gesamtvermögen (Klumpenrisiko).

    Bezugsgröße ist das über die Laufzeit eingezahlte Kapital zuzüglich des
    sonstigen Vermögens. Beide Größen sind nominal und undiskontiert; für die
    Frage, wie stark ein Haushalt auf ein einzelnes Ziel setzt, genügt das.

    Args:
        client: Kundenprofil.
        goal: Anlageziel.

    Returns:
        Der Anteil zwischen 0 und 1.
    """
    anlage = planned_investment(goal)
    return share(anlage, anlage + client.household.other_assets)


def _einkommensstabilitaet(client: Client, goal: Goal) -> Decimal:  # noqa: ARG001
    """Platzhalter für die kategoriale Dimension.

    Der Messwert ist die Beschäftigungsart und damit kein Zahlenwert; er wird in
    :func:`assess_capacity` gesondert behandelt. Diese Funktion liefert nur den
    im Trace ausgewiesenen Zahlenwert null.

    Args:
        client: Kundenprofil, hier ohne Bedeutung.
        goal: Anlageziel, hier ohne Bedeutung.

    Returns:
        Stets null.
    """
    return Decimal(0)


#: Die sechs Kriterien in Auswertungsreihenfolge. Die Schlüssel entsprechen den
#: Dimensionen in ``policies/capacity_v1.yaml``.
CRITERIA: Final[tuple[_Criterion, ...]] = (
    _Criterion(
        key="liquiditaetsreserve",
        rule=Rule(
            "CAP-LIQ",
            "Die Liquiditätsreserve in Monatsausgaben wird über die Punktbänder der "
            "Dimension 'liquiditaetsreserve' bewertet.",
            LEGAL_BASIS,
        ),
        unit="Monatsausgaben",
        measure=_liquiditaetsreserve,
    ),
    _Criterion(
        key="einkommensstabilitaet",
        rule=Rule(
            "CAP-INC",
            "Die Beschäftigungsart wird über die Ausprägungen der Dimension "
            "'einkommensstabilitaet' bewertet.",
            LEGAL_BASIS,
        ),
        unit="Beschäftigungsart",
        measure=_einkommensstabilitaet,
        categorical=True,
    ),
    _Criterion(
        key="freie_sparquote",
        rule=Rule(
            "CAP-SAV",
            "Die freie Sparquote als Anteil am Nettoeinkommen wird über die Punktbänder "
            "der Dimension 'freie_sparquote' bewertet.",
            LEGAL_BASIS,
        ),
        unit="Anteil am Nettoeinkommen",
        measure=_freie_sparquote,
    ),
    _Criterion(
        key="anlagehorizont",
        rule=Rule(
            "CAP-HOR",
            "Der Anlagehorizont in Monaten bis zum Zieltermin wird über die Punktbänder "
            "der Dimension 'anlagehorizont' bewertet.",
            LEGAL_BASIS,
        ),
        unit="Monate",
        measure=_anlagehorizont,
    ),
    _Criterion(
        key="verschuldung",
        rule=Rule(
            "CAP-DEBT",
            "Der monatliche Kapitaldienst als Anteil am Nettoeinkommen wird über die "
            "Punktbänder der Dimension 'verschuldung' bewertet.",
            LEGAL_BASIS,
        ),
        unit="Anteil am Nettoeinkommen",
        measure=_verschuldungsgrad,
    ),
    _Criterion(
        key="vermoegensanteil",
        rule=Rule(
            "CAP-CONC",
            "Der Anteil der geplanten Anlage am Gesamtvermögen (Klumpenrisiko) wird über "
            "die Punktbänder der Dimension 'vermoegensanteil' bewertet.",
            LEGAL_BASIS,
        ),
        unit="Anteil am Gesamtvermögen",
        measure=_vermoegensanteil,
    ),
)


def assess_capacity(
    client: Client,
    goal: Goal,
    policy: CapacityPolicy,
    recorder: TraceRecorder,
) -> CapacityResult:
    """Ermittelt die Risikotragfähigkeit und protokolliert jeden Schritt.

    Args:
        client: Geprüftes Kundenprofil.
        goal: Geprüftes Anlageziel.
        policy: Punktetabelle der Tragfähigkeit.
        recorder: Recorder, in den jedes Kriterium ein Ereignis schreibt.

    Returns:
        Das Ergebnis der objektiven Achse.

    Raises:
        PolicyError: Wenn das Regelwerk eine benötigte Dimension nicht führt.
    """
    reserve = client.household.liquidity_reserve_months
    ko_ausgeloest = reserve < policy.mindestreserve_monate
    ko_grund = (
        f"Die Liquiditätsreserve von {reserve} Monatsausgaben liegt unter der "
        f"Mindestreserve von {policy.mindestreserve_monate} Monatsausgaben."
        if ko_ausgeloest
        else None
    )
    recorder.record(
        RULE_KO,
        {
            "liquiditaetsreserve_monate": reserve,
            "mindestreserve_monate": policy.mindestreserve_monate,
        },
        {"ausgeloest": ko_ausgeloest, "begruendung": ko_grund},
        policy.version,
    )

    ergebnisse: list[CriterionScore] = []
    for kriterium in CRITERIA:
        dimension = policy.dimensionen.get(kriterium.key)
        if dimension is None:
            raise PolicyError(
                f"Die Dimension {kriterium.key!r} fehlt in capacity_v1.yaml, wird aber "
                "für die Tragfähigkeit benötigt."
            )

        if kriterium.categorical:
            auspraegung = client.household.employment_type.name
            punkte, grundlage = resolve_option(dimension, auspraegung, kriterium.key)
            messwert = Decimal(0)
            eingaben: dict[str, object] = {"auspraegung": auspraegung}
        else:
            messwert = kriterium.measure(client, goal)
            punkte, grundlage = resolve_band(dimension, messwert)
            eingaben = {"messwert": messwert, "einheit": kriterium.unit}

        beitrag = weighted(punkte, dimension.gewicht)
        eingaben["gewicht"] = dimension.gewicht
        eingaben["grundlage"] = grundlage
        recorder.record(
            kriterium.rule,
            eingaben,
            {"punkte": punkte, "gewichteter_beitrag": beitrag},
            policy.version,
        )
        ergebnisse.append(
            CriterionScore(
                criterion=kriterium.key,
                measured_value=messwert,
                unit=kriterium.unit,
                points=punkte,
                weight=dimension.gewicht,
                weighted_points=beitrag,
                basis=grundlage,
            )
        )

    summe = total_points([eintrag.weighted_points for eintrag in ergebnisse])
    klasse_aus_punkten = risk_class_for_points(policy.klassenschwellen, summe)
    endklasse = 1 if ko_ausgeloest else klasse_aus_punkten

    recorder.record(
        RULE_CLASS,
        {
            "gesamtpunkte": summe,
            "klasse_aus_punkten": klasse_aus_punkten,
            "ko_ausgeloest": ko_ausgeloest,
        },
        {"risikoklasse": endklasse, "begruendung": ko_grund},
        policy.version,
    )

    return CapacityResult(
        risk_class=endklasse,
        class_from_points=klasse_aus_punkten,
        total_points=summe,
        scores=tuple(ergebnisse),
        ko_triggered=ko_ausgeloest,
        ko_reason=ko_grund,
        policy_version=policy.version,
    )
