"""Kenntnisse und Erfahrungen — Art. 55 DelVO (EU) 2017/565.

Das Ergebnis ist **kein Punktwert und keine dritte Achse**, sondern eine
Obergrenze. Das ist keine Feinheit der Umsetzung, sondern folgt aus der
Vorschrift: Art. 55 verlangt Angaben zu Kenntnissen und Erfahrungen, damit die
Firma beurteilen kann, ob der Kunde die **Risiken versteht**. Verstehen lässt
sich nicht gegen ein hohes Einkommen aufrechnen.

Daraus folgt die Rechenrichtung: Kenntnisse können eine Risikoklasse **senken**,
niemals anheben. Ein Kunde mit umfassenden Derivatekenntnissen, aber ohne
Rücklage und ohne Risikobereitschaft bleibt in Klasse 1. Ein Kunde mit hoher
Tragfähigkeit und Bereitschaft, der noch nie ein Wertpapier gehalten hat, wird
gedeckelt.

Auswertung
----------
Eine Instrumentengattung „schaltet frei“, wenn **alle drei** Anforderungen des
Regelwerks erfüllt sind: Mindestkenntnisstand, Mindesterfahrungsdauer und
Mindestzahl an Geschäften je Jahr. Maßgeblich ist die höchste so
freigeschaltete Risikoklasse.

Gattungen, zu denen nichts erhoben wurde, zählen als nicht erfüllt. Was nicht
erhoben wurde, darf nicht zur Freigabe komplexerer Produkte führen — die
Beweislast liegt beim Nachweis, nicht beim Zweifel.

Die Rangfolge der Kenntnisstufen (``KEINE`` < ``GRUNDKENNTNISSE`` <
``ERWEITERT`` < ``UMFASSEND``) steht hier und nicht im Regelwerk. Sie ist die
Ordnungsstruktur der Skala selbst und keine Festlegung des Anbieters: Eine
Policy, die „Grundkenntnisse“ über „umfassende Kenntnisse“ stellt, wäre nicht
streng, sondern falsch.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final

from candor.domain.client import Client, InstrumentCategory, KnowledgeLevel
from candor.domain.policy import KnowledgePolicy
from candor.domain.result import KnowledgeResult
from candor.trace.events import Rule
from candor.trace.recorder import TraceRecorder

#: Rechtsgrundlage der Achse.
LEGAL_BASIS: Final = "Art. 55 DelVO (EU) 2017/565"

#: Rangfolge der Kenntnisstufen. Ordnungsstruktur der Skala, keine Bewertung.
KNOWLEDGE_ORDER: Final[tuple[KnowledgeLevel, ...]] = (
    KnowledgeLevel.KEINE,
    KnowledgeLevel.GRUNDKENNTNISSE,
    KnowledgeLevel.ERWEITERT,
    KnowledgeLevel.UMFASSEND,
)

RULE_CATEGORY = Rule(
    rule_id="KNW-CAT",
    description=(
        "Für eine Instrumentengattung wird geprüft, ob Kenntnisstand, Erfahrungsdauer "
        "und Transaktionshäufigkeit die Mindestanforderungen des Regelwerks erfüllen."
    ),
    legal_basis=LEGAL_BASIS,
)
RULE_CAP = Rule(
    rule_id="KNW-CAP",
    description=(
        "Aus den freigeschalteten Instrumentengattungen wird der Deckel auf die "
        "zulässige Produktkomplexität gebildet; ohne Nachweis gilt der Vorgabewert "
        "des Regelwerks."
    ),
    legal_basis=LEGAL_BASIS,
)


def knowledge_rank(level: KnowledgeLevel) -> int:
    """Liefert den Rang einer Kenntnisstufe.

    Args:
        level: Die Kenntnisstufe.

    Returns:
        Der Rang, beginnend bei null für ``KEINE``.
    """
    return KNOWLEDGE_ORDER.index(level)


def assess_knowledge(
    client: Client,
    policy: KnowledgePolicy,
    recorder: TraceRecorder,
) -> KnowledgeResult:
    """Ermittelt den Deckel auf die zulässige Produktkomplexität.

    Args:
        client: Geprüftes Kundenprofil.
        policy: Regelwerk der Kenntnisse und Erfahrungen.
        recorder: Recorder, in den jede Gattung ein Ereignis schreibt.

    Returns:
        Das Ergebnis als Obergrenze, nicht als Punktwert.
    """
    freigeschaltet: list[str] = []
    gescheitert: dict[str, str] = {}

    for gattung in InstrumentCategory:
        anforderung = policy.gattungen.get(gattung.name)
        if anforderung is None:  # pragma: no cover — Modell erzwingt Vollständigkeit
            continue

        eintrag = client.knowledge.for_category(gattung)
        if eintrag is None:
            gescheitert[gattung.name] = "Zu dieser Gattung wurde nichts erhoben."
            recorder.record(
                RULE_CATEGORY,
                {
                    "gattung": gattung.name,
                    "erhoben": False,
                    "mindestkenntnis": anforderung.mindestkenntnis,
                    "mindesterfahrung_jahre": anforderung.mindesterfahrung_jahre,
                    "mindesttransaktionen_pro_jahr": (anforderung.mindesttransaktionen_pro_jahr),
                },
                {"erfuellt": False, "begruendung": gescheitert[gattung.name]},
                policy.version,
            )
            continue

        maengel: list[str] = []
        if knowledge_rank(eintrag.knowledge_level) < knowledge_rank(anforderung.mindestkenntnis):
            maengel.append(
                f"Kenntnisstand '{eintrag.knowledge_level.value}' bleibt unter "
                f"'{anforderung.mindestkenntnis.value}'"
            )
        if eintrag.experience_years < anforderung.mindesterfahrung_jahre:
            maengel.append(
                f"Erfahrungsdauer {eintrag.experience_years} Jahre bleibt unter "
                f"{anforderung.mindesterfahrung_jahre} Jahren"
            )
        if eintrag.transactions_per_year < anforderung.mindesttransaktionen_pro_jahr:
            maengel.append(
                f"{eintrag.transactions_per_year} Geschäfte je Jahr bleiben unter "
                f"{anforderung.mindesttransaktionen_pro_jahr}"
            )

        erfuellt = not maengel
        if erfuellt:
            freigeschaltet.append(gattung.name)
        else:
            gescheitert[gattung.name] = "; ".join(maengel)

        recorder.record(
            RULE_CATEGORY,
            {
                "gattung": gattung.name,
                "erhoben": True,
                "kenntnisstand": eintrag.knowledge_level,
                "erfahrung_jahre": eintrag.experience_years,
                "transaktionen_pro_jahr": eintrag.transactions_per_year,
                "mindestkenntnis": anforderung.mindestkenntnis,
                "mindesterfahrung_jahre": anforderung.mindesterfahrung_jahre,
                "mindesttransaktionen_pro_jahr": anforderung.mindesttransaktionen_pro_jahr,
            },
            {
                "erfuellt": erfuellt,
                "schaltet_risikoklasse_frei": (
                    anforderung.schaltet_risikoklasse_frei if erfuellt else None
                ),
                "begruendung": None if erfuellt else gescheitert[gattung.name],
            },
            policy.version,
        )

    if freigeschaltet:
        deckel = max(policy.gattungen[name].schaltet_risikoklasse_frei for name in freigeschaltet)
        begruendung = (
            f"Höchste freigeschaltete Klasse aus den Gattungen: {', '.join(freigeschaltet)}."
        )
    else:
        deckel = policy.ohne_nachweis_max_risikoklasse
        begruendung = (
            "Keine Instrumentengattung erfüllt die Mindestanforderungen; es gilt der "
            "Vorgabewert ohne Nachweis."
        )

    recorder.record(
        RULE_CAP,
        {
            "freigeschaltete_gattungen": freigeschaltet,
            "ohne_nachweis_max_risikoklasse": policy.ohne_nachweis_max_risikoklasse,
        },
        {"max_risikoklasse": deckel, "begruendung": begruendung},
        policy.version,
    )

    return KnowledgeResult(
        max_risk_class=deckel,
        qualifying_categories=tuple(freigeschaltet),
        failed_categories=gescheitert,
        policy_version=policy.version,
    )


def experience_summary(client: Client) -> dict[str, Decimal]:
    """Verdichtet die Erfahrungsangaben für Anzeige und Bericht.

    Args:
        client: Geprüftes Kundenprofil.

    Returns:
        Zuordnung Gattungsname auf Erfahrungsdauer in Jahren.
    """
    return {eintrag.category.name: eintrag.experience_years for eintrag in client.knowledge.entries}
