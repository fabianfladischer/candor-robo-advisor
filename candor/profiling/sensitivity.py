"""Sensitivitätsanalyse der Risikoprofilierung.

Die Frage
---------
Welche einzelne Antwort verändert die Risikoklasse am stärksten? Die Antwort
entscheidet darüber, wo ein Fragebogen sorgfältig sein muss und wo nicht — und
sie deckt auf, ob eine einzelne Angabe unangemessen viel Gewicht trägt.

Verfahren
---------
Ein Basisprofil wird festgehalten. Für jedes Item wird **eine** Angabe über
ihren gesamten zulässigen Bereich variiert, alles andere bleibt unverändert.
Gemessen wird die resultierende **Ergebnisklasse** — nicht der Punktwert der
Achse. Das ist der Unterschied, auf den es ankommt: Ein Item kann viele Punkte
bewegen und trotzdem folgenlos bleiben, wenn seine Achse nicht die bindende ist.
Umgekehrt kann ein gering gewichtetes Item die Klasse umwerfen, wenn es genau an
einer Schwelle sitzt.

Die Analyse ist damit ausdrücklich eine **Ein-Faktor-Analyse**. Sie erfasst keine
Wechselwirkungen: Dass zwei Antworten gemeinsam mehr bewegen als einzeln, bleibt
unsichtbar. Für die Frage nach dem größten Hebel je Item genügt das; für eine
Aussage über den gesamten Eingaberaum nicht.

Warum das Ergebnis vom Basisprofil abhängt
------------------------------------------
Ein Item, dessen Achse im Basisprofil nicht bindet, zeigt geringen Hebel — bei
einem anderen Basisprofil kann derselbe Hebel groß sein. Die Analyse wird
deshalb über **mehrere** Basisprofile gerechnet und der Hebel als Spanne über
alle ausgewiesen, nicht als eine Zahl.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from candor import __version__
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
from candor.domain.errors import DomainError
from candor.domain.goal import Goal
from candor.domain.policy import PolicySet
from candor.profiling.classify import classify
from candor.trace.recorder import TraceMetadata, TraceRecorder

#: Kategorien, deren Achse die Tragfähigkeit ist.
AXIS_CAPACITY: Final = "Risikotragfähigkeit"

#: Kategorien, deren Achse die Bereitschaft ist.
AXIS_TOLERANCE: Final = "Risikobereitschaft"

#: Kategorien, deren Achse die Kenntnisse sind.
AXIS_KNOWLEDGE: Final = "Kenntnisse und Erfahrungen"


class SensitivityError(DomainError):
    """Die Sensitivitätsanalyse lässt sich für dieses Item nicht durchführen."""


@dataclass(frozen=True, slots=True)
class ItemSensitivity:
    """Hebel eines einzelnen Items auf die Ergebnisklasse.

    Attributes:
        item: Kennung des Items.
        label: Bezeichnung für Tabellen und Abbildungen.
        axis: Achse, auf die das Item einzahlt.
        baseline_class: Ergebnisklasse im Basisprofil.
        min_class: Niedrigste erreichbare Ergebnisklasse.
        max_class: Höchste erreichbare Ergebnisklasse.
        min_value: Ausprägung, die zur niedrigsten Klasse führt.
        max_value: Ausprägung, die zur höchsten Klasse führt.
        variants: Zahl der geprüften Ausprägungen.
    """

    item: str
    label: str
    axis: str
    baseline_class: int
    min_class: int
    max_class: int
    min_value: str
    max_value: str
    variants: int

    @property
    def span(self) -> int:
        """Spannweite der Ergebnisklasse über alle Ausprägungen."""
        return self.max_class - self.min_class

    @property
    def downside(self) -> int:
        """Klassenstufen, die das Item gegenüber dem Basisprofil abwärts bewegt."""
        return self.baseline_class - self.min_class

    @property
    def upside(self) -> int:
        """Klassenstufen, die das Item gegenüber dem Basisprofil aufwärts bewegt."""
        return self.max_class - self.baseline_class


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    """Ergebnis der Sensitivitätsanalyse für ein Basisprofil.

    Attributes:
        profile_name: Kennung des Basisprofils.
        baseline_class: Ergebnisklasse des unveränderten Profils.
        binding_axis: Bindende Achse des unveränderten Profils.
        items: Hebel je Item, nach Spannweite absteigend geordnet.
    """

    profile_name: str
    baseline_class: int
    binding_axis: str
    items: tuple[ItemSensitivity, ...]

    @property
    def strongest(self) -> ItemSensitivity:
        """Das Item mit dem größten Hebel.

        Raises:
            SensitivityError: Wenn kein Item ausgewertet wurde.
        """
        if not self.items:
            msg = "Die Analyse enthält kein einziges Item."
            raise SensitivityError(msg)
        return self.items[0]


def _recorder(policies: PolicySet) -> TraceRecorder:
    """Baut einen Recorder für einen einzelnen Profilierungslauf.

    Der Trace wird hier nicht ausgewertet — die Profilierung verlangt ihn aber,
    und sie soll es: Auch ein Lauf, der nur einer Analyse dient, hinterlässt
    einen vollständigen Entscheidungspfad.

    Args:
        policies: Regelwerk, dessen Fassung in den Trace wandert.

    Returns:
        Der Recorder.
    """
    return TraceRecorder(
        TraceMetadata(
            client_id="K-SENS",
            goal_id="Z-SENS",
            policy_versions=dict(policies.versions),
            package_version=__version__,
        )
    )


def _klasse(client: Client, goal: Goal, policies: PolicySet) -> int:
    """Ermittelt die Ergebnisklasse eines Profils.

    Args:
        client: Kundenprofil.
        goal: Anlageziel.
        policies: Regelwerk.

    Returns:
        Die bindende Risikoklasse.
    """
    return classify(client, goal, policies, _recorder(policies)).risk_class


def _mit_haushalt(client: Client, **felder: object) -> Client:
    """Erzeugt ein Kundenprofil mit geändertem Haushalt.

    Args:
        client: Ausgangsprofil.
        **felder: Zu ändernde Haushaltsfelder.

    Returns:
        Das geänderte Profil.
    """
    haushalt = client.household.model_copy(update=felder)
    return client.model_copy(update={"household": haushalt})


def _mit_antwort(client: Client, item: str, wert: str) -> Client:
    """Erzeugt ein Kundenprofil mit geänderter Antwort zur Risikobereitschaft.

    Args:
        client: Ausgangsprofil.
        item: Kennung des Items.
        wert: Neue Ausprägung.

    Returns:
        Das geänderte Profil.
    """
    antworten = dict(client.risk_attitude.answers)
    antworten[item] = wert
    return client.model_copy(update={"risk_attitude": RiskAttitude(answers=antworten)})


def _mit_kenntnissen(client: Client, level: KnowledgeLevel) -> Client:
    """Erzeugt ein Kundenprofil mit geändertem Kenntnisstand.

    Alle erfassten Gattungen erhalten denselben Stand. Die Variation zielt auf
    den Deckel, nicht auf die Feinstruktur der Gattungen.

    Beim Stand ``KEINE`` werden Erfahrungsjahre und Transaktionszahl auf null
    gesetzt. Der Domänenkern weist die Kombination „keine Kenntnisse bei
    vorhandener Erfahrung" als widersprüchlich zurück — zu Recht: Sie ist nicht
    auswertbar. Die Variation muss diese Zusammengehörigkeit nachvollziehen,
    statt sie zu umgehen.

    Args:
        client: Ausgangsprofil.
        level: Neuer Kenntnisstand.

    Returns:
        Das geänderte Profil.
    """
    ohne_kenntnisse = level is KnowledgeLevel.KEINE
    eintraege = tuple(
        InstrumentExperience(
            category=eintrag.category,
            knowledge_level=level,
            experience_years=Decimal("0.0") if ohne_kenntnisse else eintrag.experience_years,
            transactions_per_year=0 if ohne_kenntnisse else eintrag.transactions_per_year,
        )
        for eintrag in client.knowledge.entries
    )
    kenntnisse = client.knowledge.model_copy(update={"entries": eintraege})
    return client.model_copy(update={"knowledge": kenntnisse})


def _tolerance_items(policies: PolicySet) -> dict[str, tuple[str, ...]]:
    """Liest die Ausprägungen aller Items der Risikobereitschaft.

    Args:
        policies: Regelwerk.

    Returns:
        Zuordnung Itemkennung auf ihre Ausprägungen.
    """
    return {
        name: tuple(sorted(dimension.optionen or {}))
        for name, dimension in policies.tolerance.dimensionen.items()
    }


#: Kurzbezeichnung je Item der Risikobereitschaft. Die ``beschreibung`` im
#: Regelwerk ist ein erklärender Satz und für eine Achsenbeschriftung zu lang;
#: eine automatische Kürzung schnitte mitten im Wort ab.
TOLERANCE_LABELS: Final[dict[str, str]] = {
    "verlustszenario_betrag": "Hingenommener Verlustbetrag",
    "reaktion_auf_rueckgang": "Reaktion auf 20 % Rückgang",
    "erfahrung_mit_schwankungen": "Erlebte Schwankungen",
    "selbsteinschaetzung": "Selbsteinschätzung",
    "entnahmedisziplin": "Vorzeitige Entnahme",
}

#: Variationsbereiche der stetigen Haushaltsangaben. Die Werte spannen den in
#: der Beratung realistischen Bereich auf; sie sind eine Setzung.
HOUSEHOLD_RANGES: Final[dict[str, tuple[str, tuple[Decimal, ...]]]] = {
    "liquidity_reserve_months": (
        "Liquiditätsreserve (K.-o.)",
        (Decimal("0.0"), Decimal("2.0"), Decimal("3.0"), Decimal("6.0"), Decimal("12.0")),
    ),
    "free_savings_amount": (
        "Freier Sparbetrag je Monat",
        (Decimal("50.00"), Decimal("200.00"), Decimal("450.00"), Decimal("800.00")),
    ),
    "total_debt": (
        "Bestehende Verbindlichkeiten",
        (Decimal("0.00"), Decimal("20000.00"), Decimal("60000.00"), Decimal("150000.00")),
    ),
    "other_assets": (
        "Sonstiges Vermögen",
        (Decimal("0.00"), Decimal("25000.00"), Decimal("100000.00"), Decimal("300000.00")),
    ),
}


def analyse(
    client: Client,
    goal: Goal,
    policies: PolicySet,
    profile_name: str = "Basisprofil",
) -> SensitivityReport:
    """Führt die Ein-Faktor-Sensitivitätsanalyse durch.

    Args:
        client: Basisprofil.
        goal: Anlageziel; bleibt über alle Varianten unverändert.
        policies: Regelwerk.
        profile_name: Kennung des Basisprofils für die Ausgabe.

    Returns:
        Der Bericht, nach Hebel absteigend geordnet.
    """
    basisprofil = classify(client, goal, policies, _recorder(policies))
    basis = basisprofil.risk_class
    ergebnisse: list[ItemSensitivity] = []

    def _erfassen(item: str, label: str, axis: str, varianten: dict[str, Client]) -> None:
        """Wertet eine Variantenschar aus und legt das Ergebnis ab.

        Args:
            item: Kennung des Items.
            label: Bezeichnung für die Ausgabe.
            axis: Achse, auf die das Item einzahlt.
            varianten: Zuordnung Ausprägung auf Kundenprofil.
        """
        klassen = {name: _klasse(variante, goal, policies) for name, variante in varianten.items()}
        niedrigste = min(klassen, key=lambda name: (klassen[name], name))
        hoechste = max(klassen, key=lambda name: (klassen[name], name))
        ergebnisse.append(
            ItemSensitivity(
                item=item,
                label=label,
                axis=axis,
                baseline_class=basis,
                min_class=klassen[niedrigste],
                max_class=klassen[hoechste],
                min_value=niedrigste,
                max_value=hoechste,
                variants=len(klassen),
            )
        )

    for item, auspraegungen in _tolerance_items(policies).items():
        _erfassen(
            item,
            TOLERANCE_LABELS.get(item, item),
            AXIS_TOLERANCE,
            {wert: _mit_antwort(client, item, wert) for wert in auspraegungen},
        )

    for feld, (label, werte) in HOUSEHOLD_RANGES.items():
        _erfassen(
            feld,
            label,
            AXIS_CAPACITY,
            {str(wert): _mit_haushalt(client, **{feld: wert}) for wert in werte},
        )

    _erfassen(
        "employment_type",
        "Beschäftigungsverhältnis",
        AXIS_CAPACITY,
        {art.name: _mit_haushalt(client, employment_type=art) for art in EmploymentType},
    )

    _erfassen(
        "knowledge_level",
        "Kenntnisstand (Deckel)",
        AXIS_KNOWLEDGE,
        {stand.name: _mit_kenntnissen(client, stand) for stand in KnowledgeLevel},
    )

    ergebnisse.sort(key=lambda eintrag: (-eintrag.span, eintrag.label))
    return SensitivityReport(
        profile_name=profile_name,
        baseline_class=basis,
        binding_axis=basisprofil.binding_axis.value,
        items=tuple(ergebnisse),
    )


def baseline_clients() -> dict[str, Client]:
    """Baut drei Basisprofile für die Analyse.

    Die Profile unterscheiden sich in der bindenden Achse. Nur so wird sichtbar,
    dass der Hebel eines Items davon abhängt, ob seine Achse gerade bindet.

    Returns:
        Zuordnung Profilname auf Kundenprofil.
    """

    def _bauen(
        name: str,
        reserve: Decimal,
        antworten: dict[str, str],
        stand: KnowledgeLevel,
    ) -> tuple[str, Client]:
        """Baut ein Basisprofil.

        Args:
            name: Kennung des Profils.
            reserve: Liquiditätsreserve in Monatsausgaben.
            antworten: Antworten zur Risikobereitschaft.
            stand: Kenntnisstand.

        Returns:
            Name und Kundenprofil.
        """
        return name, Client(
            client_id=f"K-SENS-{name}",
            household=Household(
                net_monthly_income=Decimal("4200.00"),
                fixed_monthly_costs=Decimal("2400.00"),
                monthly_debt_service=Decimal("300.00"),
                free_savings_amount=Decimal("600.00"),
                liquidity_reserve_months=reserve,
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
                        knowledge_level=stand,
                        experience_years=Decimal("5.0"),
                        transactions_per_year=6,
                    ),
                ),
            ),
            sustainability=SustainabilityPreferences(has_preferences=False),
            risk_attitude=RiskAttitude(answers=antworten),
        )

    mitte = {
        "verlustszenario_betrag": "S3",
        "reaktion_auf_rueckgang": "ABWARTEN",
        "erfahrung_mit_schwankungen": "ERLEBT_UND_GEHALTEN",
        "selbsteinschaetzung": "AUSGEWOGEN",
        "entnahmedisziplin": "UNWAHRSCHEINLICH",
    }
    vorsichtig = {
        **mitte,
        "verlustszenario_betrag": "S1",
        "selbsteinschaetzung": "SICHERHEITSORIENTIERT",
    }
    offensiv = {
        "verlustszenario_betrag": "S5",
        "reaktion_auf_rueckgang": "NACHKAUFEN",
        "erfahrung_mit_schwankungen": "ERLEBT_UND_NACHGEKAUFT",
        "selbsteinschaetzung": "CHANCENORIENTIERT",
        "entnahmedisziplin": "AUSGESCHLOSSEN",
    }

    return dict(
        (
            _bauen("ausgewogen", Decimal("4.0"), mitte, KnowledgeLevel.ERWEITERT),
            _bauen("vorsichtig", Decimal("6.0"), vorsichtig, KnowledgeLevel.GRUNDKENNTNISSE),
            _bauen("offensiv", Decimal("8.0"), offensiv, KnowledgeLevel.UMFASSEND),
        )
    )
