"""Demo-Oberfläche für reproduzierbare Bildschirmaufnahmen.

Zweck und ausdrückliche Nicht-Ziele
-----------------------------------
Diese Oberfläche dient **einem** Zweck: nachvollziehbare Bildschirmaufnahmen der
Entscheidungskette. Sie ist **kein Produktionscode**. Es fehlen bewusst:

* Authentifizierung, Sitzungsverwaltung, Mandantentrennung,
* Persistenz — jede Eingabe ist nach dem Neuladen verloren,
* Barrierefreiheit über das hinaus, was Streamlit mitbringt,
* jede Fehlerbehandlung, die über das Anzeigen der Meldung hinausgeht.

Wer sie in Betrieb nähme, verstieße gegen mehrere aufsichtsrechtliche Vorgaben,
angefangen bei der Aufbewahrungspflicht für Geeignetheitserklärungen.

Was sie richtig macht
---------------------
Sie ruft **denselben** Entscheidungspfad auf wie jede andere Ausgabe des
Systems: :func:`candor.engine.advise.advise`. Es gibt keinen zweiten Rechenweg
für die Oberfläche. Ein Ergebnis auf dem Bildschirm ist damit dasselbe wie eines
im PDF und eines im Test — bei gleichem Profil und Seed bitidentisch
(Invariante I4).

Aufruf:
    make app
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Final

import streamlit as st

from candor import format_de
from candor.data.loader import snapshot_info
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
    SustainabilityComponent,
    SustainabilityPreferences,
)
from candor.domain.errors import DomainError
from candor.domain.goal import Goal, GoalType
from candor.domain.policy import PolicySet, TolerancePolicy
from candor.engine.advise import Advice, advise
from candor.engine.convergence import load_history
from candor.engine.simulate import BlockBootstrapSampler, FloatArray
from candor.goals.model import build_goal
from candor.profiling.tolerance import scenario_amounts
from candor.report import conflict, suitability

#: Stichtag der Demo. Fest gesetzt, damit Bildschirmaufnahmen reproduzierbar
#: bleiben — mit dem heutigen Datum änderte sich der Anlagehorizont täglich.
DEMO_DATE: Final = date(2026, 7, 26)

#: Beschriftung der vier Schritte.
STEPS: Final[tuple[str, ...]] = (
    "Schritt 1 · Anlageziel",
    "Schritt 2 · Finanzlage",
    "Schritt 3 · Risikobereitschaft",
    "Schritt 4 · Kenntnisse und Nachhaltigkeit",
)


@dataclass(frozen=True, slots=True)
class GoalInput:
    """Die im ersten Schritt erhobenen Zielangaben.

    Attributes:
        goal_type: Gewählter Zieltyp.
        target_amount: Angestrebter Betrag.
        initial_capital: Bereits vorhandenes Kapital.
        monthly_contribution: Geplante monatliche Sparrate.
        years: Anlagehorizont in Jahren.
    """

    goal_type: GoalType
    target_amount: Decimal
    initial_capital: Decimal
    monthly_contribution: Decimal
    years: int


#: Wortlaut der Verlustszenario-Frage. Ihre Antwortmöglichkeiten stehen bewusst
#: **nicht** in :data:`TOLERANCE_CHOICES`: Sie sind kein Text, sondern ein
#: Rechenergebnis aus dem Anlageziel und werden von
#: :func:`verlustszenario_optionen` erzeugt.
SCENARIO_QUESTION: Final = (
    "Welchen zwischenzeitlichen Rückgang Ihres Anlagebetrags würden Sie hinnehmen?"
)

#: Fragen und Auswahlmöglichkeiten der Bereitschaftserhebung — ohne die
#: Verlustszenario-Frage, deren Beschriftung berechnet wird.
TOLERANCE_CHOICES: Final[dict[str, tuple[str, dict[str, str]]]] = {
    "reaktion_auf_rueckgang": (
        "Ihr Depot verliert binnen weniger Monate 20 %. Was tun Sie?",
        {
            "SOFORT_VERKAUFEN": "sofort verkaufen",
            "TEILWEISE_VERKAUFEN": "teilweise verkaufen",
            "ABWARTEN": "abwarten",
            "NACHKAUFEN": "nachkaufen",
        },
    ),
    "erfahrung_mit_schwankungen": (
        "Haben Sie einen Rückgang von mehr als 20 % bereits selbst erlebt?",
        {
            "NIE_ERLEBT": "nie erlebt",
            "ERLEBT_UND_VERKAUFT": "erlebt und verkauft",
            "ERLEBT_UND_GEHALTEN": "erlebt und gehalten",
            "ERLEBT_UND_NACHGEKAUFT": "erlebt und nachgekauft",
        },
    ),
    "selbsteinschaetzung": (
        "Wie schätzen Sie Ihre Risikoneigung selbst ein?",
        {
            "SEHR_SICHERHEITSORIENTIERT": "sehr sicherheitsorientiert",
            "SICHERHEITSORIENTIERT": "sicherheitsorientiert",
            "AUSGEWOGEN": "ausgewogen",
            "WACHSTUMSORIENTIERT": "wachstumsorientiert",
            "CHANCENORIENTIERT": "chancenorientiert",
        },
    ),
    "entnahmedisziplin": (
        "Wie wahrscheinlich ist eine Entnahme vor dem Zieltermin?",
        {
            "SEHR_WAHRSCHEINLICH": "sehr wahrscheinlich",
            "MOEGLICH": "möglich",
            "UNWAHRSCHEINLICH": "unwahrscheinlich",
            "AUSGESCHLOSSEN": "ausgeschlossen",
        },
    ),
}


def verlustszenario_optionen(goal: Goal, policy: TolerancePolicy) -> dict[str, str]:
    """Beschriftet die Verlustszenario-Frage mit den vorgelegten Eurobeträgen.

    Die Beträge kommen ausschließlich aus
    :func:`candor.profiling.tolerance.scenario_amounts` — derselben Funktion,
    die der Entscheidungspfad aufruft und die den vorgelegten Betrag in den
    Trace schreibt. Eine eigene Rechnung an dieser Stelle wäre ein zweiter
    Rechenweg und ergäbe abweichende Beträge, weil der Anlagehorizont
    angebrochene Monate abschneidet.

    Warum Beträge und keine Prozentangaben: siehe Modul-Docstring von
    :mod:`candor.profiling.tolerance`. Prozentwerte an dieser Stelle wären ein
    Bruch mit Regelwerk, Trace und Geeignetheitserklärung, die übereinstimmend
    Eurobeträge ausweisen.

    Args:
        goal: Das erhobene Anlageziel, aus dem sich die Bezugsgröße ergibt.
        policy: Punktetabelle der Bereitschaft mit den Szenariostufen.

    Returns:
        Zuordnung Stufenkennung auf die Beschriftung in Euro.
    """
    return {
        stufe: format_de.euro(betrag) for stufe, betrag in scenario_amounts(goal, policy).items()
    }


@st.cache_resource
def _regelwerk() -> PolicySet:
    """Lädt das Regelwerk einmal je Sitzung.

    Returns:
        Das geprüfte Regelwerk.
    """
    return load_policy_set()


@st.cache_resource
def _sampler() -> BlockBootstrapSampler:
    """Baut das Ziehungsverfahren einmal je Sitzung.

    Returns:
        Der Block-Bootstrap-Sampler.
    """
    policies = _regelwerk()
    return BlockBootstrapSampler(load_history(policies), policies.simulation.blocklaenge_monate)


@st.cache_data
def _historie() -> FloatArray:
    """Lädt die Renditematrix einmal je Sitzung.

    Returns:
        Die historischen Monatsrenditen.
    """
    return load_history(_regelwerk())


def _schritt_ziel() -> GoalInput:
    """Erfasst das Anlageziel.

    Returns:
        Die erhobenen Angaben.
    """
    st.subheader(STEPS[0])
    links, rechts = st.columns(2)
    with links:
        zieltyp = st.selectbox(
            "Wofür sparen Sie?",
            options=list(GoalType),
            format_func=lambda typ: typ.value,
            index=list(GoalType).index(GoalType.ALTERSVORSORGE),
        )
        zielbetrag = st.number_input(
            "Zielbetrag in Euro", min_value=1_000, max_value=5_000_000, value=300_000, step=5_000
        )
    with rechts:
        startkapital = st.number_input(
            "Bereits vorhandenes Kapital in Euro",
            min_value=0,
            max_value=2_000_000,
            value=20_000,
            step=1_000,
        )
        sparrate = st.number_input(
            "Monatliche Sparrate in Euro", min_value=0, max_value=10_000, value=500, step=25
        )
    jahre = st.slider("Anlagehorizont in Jahren", min_value=2, max_value=40, value=25)
    return GoalInput(
        goal_type=zieltyp,
        target_amount=Decimal(str(zielbetrag)),
        initial_capital=Decimal(str(startkapital)),
        monthly_contribution=Decimal(str(sparrate)),
        years=int(jahre),
    )


def _anlageziel(angaben: GoalInput, policies: PolicySet) -> tuple[Goal | None, DomainError | None]:
    """Baut das Anlageziel bereits vor der Bereitschaftserhebung.

    Die Verlustszenarien werden dem Kunden als Eurobeträge vorgelegt; ihre
    Bezugsgröße ist das über die Laufzeit eingezahlte Kapital. Das Ziel muss
    deshalb vorliegen, bevor Schritt 3 gezeichnet wird.

    :func:`candor.goals.model.build_goal` bricht bei unvollständigen Angaben ab
    (Prinzip P5) — erreichbar ist im Formular der Fall, dass weder Startkapital
    noch Sparrate angegeben sind. Dieser Abbruch darf mitten im Eingabeformular
    nicht als roher Traceback erscheinen, aber ebenso wenig verschluckt werden.
    Die Ausnahme wird deshalb mitgeführt und erst dort ausgelöst, wo die
    Oberfläche Domänenfehler ohnehin anzeigt.

    Die übrigen Feldgrenzen des Formulars (Zielbetrag ab 1 000 €, Horizont ab
    zwei Jahren, keine negativen Beträge) schließen die verbleibenden
    Prüfungen des Zielmodells aus.

    Args:
        angaben: Die im ersten Schritt erhobenen Zielangaben.
        policies: Das geprüfte Regelwerk.

    Returns:
        Das Ziel und ``None``, oder ``None`` und die aufgetretene Ausnahme.
    """
    try:
        ziel = build_goal(
            "Z-DEMO-0001-A",
            angaben.goal_type,
            angaben.target_amount,
            DEMO_DATE,
            policies.goals,
            target_date=date(DEMO_DATE.year + angaben.years, DEMO_DATE.month, 1),
            initial_capital=angaben.initial_capital,
            monthly_contribution=angaben.monthly_contribution,
        )
    except DomainError as fehler:
        return None, fehler
    return ziel, None


def _schritt_finanzlage() -> Household:
    """Erfasst die Finanzlage des Haushalts.

    Returns:
        Der erhobene Haushalt.
    """
    st.subheader(STEPS[1])
    links, rechts = st.columns(2)
    with links:
        einkommen = st.number_input(
            "Nettoeinkommen je Monat", min_value=0, max_value=50_000, value=4_200, step=100
        )
        fixkosten = st.number_input(
            "Feste Ausgaben je Monat", min_value=0, max_value=50_000, value=2_400, step=100
        )
        kapitaldienst = st.number_input(
            "Kredittilgung je Monat", min_value=0, max_value=20_000, value=300, step=50
        )
        sparbetrag = st.number_input(
            "Davon frei zum Sparen", min_value=0, max_value=20_000, value=600, step=50
        )
    with rechts:
        reserve = st.slider(
            "Liquiditätsreserve in Monatsausgaben",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5,
            help=(
                "Unterhalb der im Regelwerk hinterlegten Mindestreserve greift ein "
                "K.-o.-Kriterium: Die Risikoklasse wird unabhängig von allen anderen "
                "Angaben auf 1 gesetzt."
            ),
        )
        schulden = st.number_input(
            "Bestehende Verbindlichkeiten",
            min_value=0,
            max_value=2_000_000,
            value=18_000,
            step=1_000,
        )
        vermoegen = st.number_input(
            "Sonstiges Vermögen", min_value=0, max_value=5_000_000, value=25_000, step=1_000
        )
        beschaeftigung = st.selectbox(
            "Beschäftigungsverhältnis",
            options=list(EmploymentType),
            format_func=lambda art: art.value,
            index=list(EmploymentType).index(EmploymentType.UNBEFRISTET),
        )
    return Household(
        net_monthly_income=Decimal(str(einkommen)),
        fixed_monthly_costs=Decimal(str(fixkosten)),
        monthly_debt_service=Decimal(str(kapitaldienst)),
        free_savings_amount=Decimal(str(sparbetrag)),
        liquidity_reserve_months=Decimal(str(reserve)),
        total_debt=Decimal(str(schulden)),
        other_assets=Decimal(str(vermoegen)),
        employment_type=beschaeftigung,
    )


def _schritt_bereitschaft(ziel: Goal | None, policy: TolerancePolicy) -> dict[str, str]:
    """Erfasst die Risikobereitschaft.

    Die Verlustszenario-Frage wird gesperrt, wenn kein Anlageziel vorliegt: Ohne
    Bezugsgröße gibt es keine Beträge, und Prozentangaben ersatzweise zu zeigen
    wäre eine andere Erhebung als die dokumentierte. Die fehlende Antwort führt
    beim Auswerten zum Abbruch mit Begründung (Prinzip P5) — das ist gewollt.

    Args:
        ziel: Das erhobene Anlageziel, oder ``None`` bei unvollständigen Angaben.
        policy: Punktetabelle der Bereitschaft.

    Returns:
        Die Antworten je Item. Ohne Ziel fehlt die Antwort zum Verlustszenario.
    """
    st.subheader(STEPS[2])
    st.caption(
        "Die Verlustfrage legt Ihnen Eurobeträge vor und keine Prozentangaben: Derselbe "
        "Rückgang wird in Prozent regelmäßig anders bewertet als in Euro. Die Beträge sind "
        "aus Ihrem Anlageziel berechnet und stehen so auch im Entscheidungspfad."
    )
    antworten: dict[str, str] = {}

    szenario_item = policy.verlustszenarien.dimension
    if ziel is None:
        st.error(
            "**Die Verlustfrage kann noch nicht gestellt werden.** Die vorgelegten Beträge "
            "werden aus dem Anlageziel berechnet; ohne Startkapital und ohne Sparrate gibt es "
            "keine Bezugsgröße. Bitte ergänzen Sie Schritt 1."
        )
        st.caption(
            "Ersatzweise Prozentangaben zu zeigen wäre kein Komfort, sondern eine andere "
            "Erhebung als die, die das Regelwerk und die Geeignetheitserklärung beschreiben."
        )
    else:
        szenarien = verlustszenario_optionen(ziel, policy)
        stufen = list(szenarien)
        antworten[szenario_item] = str(
            st.radio(
                SCENARIO_QUESTION,
                options=stufen,
                format_func=lambda wert, gewaehlt=szenarien: gewaehlt[wert],
                horizontal=True,
                index=len(stufen) // 2,
                key=f"tol_{szenario_item}",
            )
        )

    for item, (frage, optionen) in TOLERANCE_CHOICES.items():
        schluessel = list(optionen)
        antworten[item] = str(
            st.radio(
                frage,
                options=schluessel,
                format_func=lambda wert, gewaehlt=optionen: gewaehlt[wert],
                horizontal=True,
                index=len(schluessel) // 2,
                key=f"tol_{item}",
            )
        )
    return antworten


def _mindestanteil_eingabe(schluessel: str) -> Decimal:
    """Erhebt den verlangten Mindestanteil einer Nachhaltigkeitsausprägung.

    Der Anteil wird ausdrücklich erhoben und nicht still vorbelegt: Ein
    unsichtbar gesetzter Wert wäre ein stiller Standardwert in einer
    Pflichtangabe, den die Geeignetheitserklärung anschließend als erhobene
    Kundenangabe ausweisen würde – ein Verstoß gegen Prinzip P5. Der Wert ist
    sichtbar, veränderbar, steht im Trace und wird nach der Optimierung gegen
    den erreichten Anteil geprüft.

    Args:
        schluessel: Kennung der Ausprägung, zugleich Schlüssel des Bedienelements.

    Returns:
        Der verlangte Mindestanteil als Anteil zwischen null und eins.
    """
    prozent = st.slider(
        "Verlangter Mindestanteil am Portfolio",
        min_value=5,
        max_value=100,
        value=50,
        step=5,
        format="%d %%",
        key=f"esg_anteil_{schluessel}",
    )
    return Decimal(int(prozent)) / Decimal(100)


def _schritt_kenntnisse() -> tuple[
    KnowledgeAndExperience, SustainabilityPreferences | None, DomainError | None
]:
    """Erfasst Kenntnisse, Erfahrungen und Nachhaltigkeitspräferenzen.

    Returns:
        Kenntnisstand, die Präferenzen und ``None`` — oder ``None`` und die
        Ausnahme, wenn die Angabe zu den Präferenzen unvollständig ist.
    """
    st.subheader(STEPS[3])
    links, rechts = st.columns(2)
    with links:
        st.markdown("**Kenntnisse und Erfahrungen** — Art. 55 DelVO (EU) 2017/565")
        gattungen = st.multiselect(
            "Womit haben Sie bereits gehandelt?",
            options=list(InstrumentCategory),
            format_func=lambda kategorie: kategorie.value,
            default=[InstrumentCategory.ETF],
        )
        stand = st.select_slider(
            "Kenntnisstand in diesen Gattungen",
            options=list(KnowledgeLevel),
            format_func=lambda stufe: stufe.value,
            value=KnowledgeLevel.ERWEITERT,
        )
        jahre = st.number_input("Erfahrung in Jahren", min_value=0, max_value=60, value=5)
        abschluesse = st.number_input("Abschlüsse je Jahr", min_value=0, max_value=500, value=6)
    with rechts:
        st.markdown("**Nachhaltigkeitspräferenzen** — DelVO (EU) 2021/1253")
        gewuenscht = st.checkbox("Ich habe Nachhaltigkeitspräferenzen", value=False)
        taxonomie = sfdr = pai = False
        anteile: dict[str, Decimal | None] = {"taxonomie": None, "sfdr": None, "pai": None}
        if gewuenscht:
            taxonomie = st.checkbox("ökologisch nachhaltig (Taxonomie-Verordnung)")
            if taxonomie:
                anteile["taxonomie"] = _mindestanteil_eingabe("taxonomie")
            sfdr = st.checkbox("nachhaltige Investition (Offenlegungsverordnung)")
            if sfdr:
                anteile["sfdr"] = _mindestanteil_eingabe("sfdr")
            pai = st.checkbox("Berücksichtigung nachteiliger Auswirkungen", value=True)
            if pai:
                anteile["pai"] = _mindestanteil_eingabe("pai")
            st.caption(
                "Die ersten beiden Ausprägungen kollidieren im hinterlegten Universum mit den "
                "Streuungsvorgaben der Risikoklassen. Das System weist den Widerspruch dann "
                "aus, statt eine der beiden Vorgaben aufzuweichen."
            )

    ohne_kenntnisse = stand is KnowledgeLevel.KEINE
    eintraege = tuple(
        InstrumentExperience(
            category=kategorie,
            knowledge_level=stand,
            experience_years=Decimal("0.0") if ohne_kenntnisse else Decimal(str(jahre)),
            transactions_per_year=0 if ohne_kenntnisse else int(abschluesse),
        )
        for kategorie in gattungen
    )
    kenntnisse = KnowledgeAndExperience(
        education_level=EducationLevel.HOCHSCHULABSCHLUSS,
        profession_finance_related=False,
        entries=eintraege,
    )

    def _komponente(aktiv: bool, schluessel: str) -> SustainabilityComponent:
        """Baut eine Ausprägung der Nachhaltigkeitspräferenz.

        Args:
            aktiv: Ob die Ausprägung verlangt wird.
            schluessel: Kennung der Ausprägung im erhobenen Mindestanteil.

        Returns:
            Die Ausprägung.
        """
        return SustainabilityComponent(
            requested=aktiv, minimum_share=anteile[schluessel] if aktiv else None
        )

    # Kein Umbiegen auf "keine Präferenzen": Wer die Frage bejaht, ohne eine
    # Ausprägung zu wählen, hat eine UNVOLLSTÄNDIGE Angabe gemacht, keine
    # Verneinung. Der Domänenkern weist das ab; die Ausnahme wird mitgeführt und
    # dort ausgelöst, wo die Oberfläche Domänenfehler anzeigt.
    try:
        praeferenzen = SustainabilityPreferences(
            has_preferences=gewuenscht,
            taxonomy_aligned=_komponente(taxonomie, "taxonomie"),
            sfdr_sustainable=_komponente(sfdr, "sfdr"),
            pai_considered=_komponente(pai, "pai"),
        )
    except DomainError as fehler:
        return kenntnisse, None, fehler
    return kenntnisse, praeferenzen, None


def _ergebnis_kopf(advice: Advice) -> None:
    """Zeigt Risikoklasse, bindende Achse und Erfolgswahrscheinlichkeit.

    Args:
        advice: Das Ergebnis der Entscheidungskette.
    """
    abgleich = advice.reconciliation
    eins, zwei, drei, vier = st.columns(4)
    eins.metric("Risikoklasse", advice.profile.risk_class)
    zwei.metric("Bindende Achse", advice.profile.binding_axis.value)
    drei.metric(
        "Erfolgswahrscheinlichkeit",
        format_de.percent(abgleich.achieved_probability),
        delta=format_de.percentage_points(abgleich.margin),
    )
    vier.metric("Gefordert", format_de.percent(abgleich.required_probability))


def _divergenzhinweis(advice: Advice) -> None:
    """Zeigt den Divergenzbefund, sofern das Regelwerk einen ausweist.

    Der Hinweis stand bisher in der Geeignetheitserklärung und — als einer von
    mehreren Befundpunkten — auf der Konfliktseite, nicht aber auf der
    Ergebnisseite im Empfehlungsfall. Damit blieb gerade der Fall unsichtbar,
    für den das System gebaut ist: eine Empfehlung, hinter der zwei weit
    auseinanderliegende Achsen stehen.

    Die Bedingung ist ausschließlich :attr:`RiskProfile.has_divergence`. Die
    Schwelle steht allein in ``candor.profiling.classify``; ein eigener
    Vergleich an dieser Stelle wäre eine zweite Wahrheitsquelle (Prinzip P3).
    Der Wortlaut kommt aus :meth:`Divergence.as_note`, derselben Quelle, aus
    der ihn der Konfliktbericht bezieht.

    Args:
        advice: Das Ergebnis der Entscheidungskette.
    """
    if not advice.profile.has_divergence:
        return
    divergenz = advice.profile.divergence
    if divergenz is None:  # pragma: no cover — durch has_divergence ausgeschlossen
        return
    st.warning(divergenz.as_note())


def _download(advice: Advice, modul: ModuleType, dateiname: str, beschriftung: str) -> None:
    """Erzeugt das PDF und bietet es zum Herunterladen an.

    Args:
        advice: Das Ergebnis der Entscheidungskette.
        modul: Das Modul, das das Dokument erzeugt.
        dateiname: Vorgeschlagener Dateiname.
        beschriftung: Beschriftung der Schaltfläche.
    """
    with TemporaryDirectory() as verzeichnis:
        pfad = modul.write_pdf(advice, Path(verzeichnis) / dateiname)
        daten = BytesIO(pfad.read_bytes())
    st.download_button(beschriftung, daten, file_name=dateiname, mime="application/pdf")


def _ansicht_empfehlung(advice: Advice) -> None:
    """Zeigt die Ergebnisseite im empfehlbaren Fall.

    Args:
        advice: Das Ergebnis der Entscheidungskette.
    """
    optimum = advice.optimization
    if optimum is None:  # pragma: no cover — durch is_recommendation ausgeschlossen
        return

    st.success(f"Anlageempfehlung möglich — Ausgang: {advice.reconciliation.outcome.value}")
    _ergebnis_kopf(advice)
    _divergenzhinweis(advice)

    st.markdown("#### Empfohlene Aufteilung")
    zeilen = [
        {
            "Assetklasse": get_instrument(schluessel).label,
            "Gewicht": format_de.percent(float(optimum.weights[i])),
            "Instrument": get_instrument(schluessel).fund_name,
        }
        for i, schluessel in enumerate(optimum.asset_keys)
        if float(optimum.weights[i]) >= suitability.MIN_WEIGHT
    ]
    st.dataframe(zeilen, hide_index=True, width="stretch")

    if advice.universe is not None and advice.universe.is_reduced:
        st.info(
            "Wegen der Nachhaltigkeitspräferenz vor der Optimierung ausgeschlossen: "
            + ", ".join(sorted(advice.universe.excluded))
        )

    _download(advice, suitability, "geeignetheitserklaerung.pdf", "Geeignetheitserklärung laden")


def _ansicht_konflikt(advice: Advice) -> None:
    """Zeigt die Ergebnisseite im Konfliktfall.

    Args:
        advice: Das Ergebnis der Entscheidungskette.
    """
    st.error(
        "**Es ergeht keine Anlageempfehlung.** Die geforderte Erfolgswahrscheinlichkeit wird "
        "unter der zulässigen Risikoklasse nicht erreicht. Rechtsgrundlage: "
        "Art. 54 Abs. 10 DelVO (EU) 2017/565."
    )
    _ergebnis_kopf(advice)

    bericht = advice.reconciliation.conflict_report
    if bericht is None:  # pragma: no cover — durch has_conflict ausgeschlossen
        return

    st.markdown("#### Befund")
    for punkt in bericht.diagnosis:
        st.markdown(f"- {punkt}")

    st.markdown("#### Die drei Stellschrauben")
    st.caption(
        "Keine davon ist eine Empfehlung. Sie beziffern, was sich ändern müsste — und ob es "
        "im erhobenen Haushalt überhaupt darstellbar wäre."
    )
    for hebel in bericht.levers:
        spalten = st.columns([1, 3, 3, 3])
        spalten[0].markdown(f"**{hebel.lever_id.value}**")
        spalten[1].markdown(hebel.name)
        in_monaten = hebel.unit == "Monate"
        stellen = 0 if in_monaten else 2
        einheit = "Monate" if in_monaten else "€"
        spalten[2].markdown(
            f"{format_de.decimal(hebel.current_value, stellen)} → "
            f"**{format_de.decimal(hebel.required_value, stellen)}** {einheit}"
        )
        if hebel.is_offered:
            spalten[3].success(hebel.feasibility.value)
        else:
            spalten[3].warning(hebel.feasibility.value)

    _download(advice, conflict, "konfliktbericht.pdf", "Konfliktbericht laden")


def _nachweis(advice: Advice) -> None:
    """Zeigt Regelstand, Datenstand und Trace-Hash.

    Args:
        advice: Das Ergebnis der Entscheidungskette.
    """
    with st.expander("Nachweis und Reproduzierbarkeit"):
        st.write(
            {
                "Regelwerk": " · ".join(
                    f"{name} {fassung}"
                    for name, fassung in sorted(advice.policies.versions.items())
                ),
                "Datenstand": advice.snapshot_hash,
                "Trace-Hash": advice.trace_hash,
                "Entscheidungsschritte": len(advice.trace),
                "Softwarefassung": advice.package_version,
            }
        )
        st.caption(
            "Gleiches Profil, gleicher Seed und gleicher Regelstand liefern denselben "
            "Trace-Hash — und ein bitgleiches PDF (Invariante I4)."
        )


def main() -> None:
    """Baut die Oberfläche."""
    st.set_page_config(page_title="CANDOR", page_icon="⚖️", layout="wide")
    st.title("CANDOR")
    st.caption(
        "Robo-Advisor mit ehrlicher Konfliktdiagnose · Demonstrator · **kein Produktionssystem**"
    )

    policies = _regelwerk()
    with st.sidebar:
        st.markdown("### Datenstand")
        st.write(snapshot_info())
        st.markdown("### Regelwerk")
        st.write(dict(sorted(policies.versions.items())))

    ziel_angaben = _schritt_ziel()
    anlageziel, zielfehler = _anlageziel(ziel_angaben, policies)
    st.divider()
    haushalt = _schritt_finanzlage()
    st.divider()
    antworten = _schritt_bereitschaft(anlageziel, policies.tolerance)
    st.divider()
    kenntnisse, praeferenzen, esg_fehler = _schritt_kenntnisse()
    st.divider()

    if not st.button("Auswerten", type="primary"):
        st.stop()

    try:
        # Das Ziel wurde bereits vor Schritt 3 gebaut, weil die Verlustszenarien
        # daraus berechnet werden. Ein dabei aufgetretener Abbruch wird hier
        # ausgelöst — an der Stelle, an der die Oberfläche Domänenfehler anzeigt.
        if zielfehler is not None:
            raise zielfehler
        if esg_fehler is not None:
            raise esg_fehler
        if anlageziel is None:  # pragma: no cover — durch zielfehler ausgeschlossen
            return
        if praeferenzen is None:  # pragma: no cover — durch esg_fehler ausgeschlossen
            return
        kunde = Client(
            client_id="K-DEMO-0001",
            household=haushalt,
            knowledge=kenntnisse,
            sustainability=praeferenzen,
            risk_attitude=RiskAttitude(answers=antworten),
        )
        with st.spinner("Simulation über 10 000 Pfade …"):
            ergebnis = advise(
                kunde,
                anlageziel,
                policies,
                _sampler(),
                _historie(),
                snapshot_hash=str(snapshot_info().get("sha256_short", "")),
                asset_keys=ASSET_KEYS,
            )
    except DomainError as fehler:
        st.error(f"**Die Auswertung wurde abgebrochen.**\n\n{fehler}")
        st.caption(
            "Das System bricht bei unvollständigen oder widersprüchlichen Angaben ab, statt "
            "mit stillen Vorgabewerten weiterzurechnen (Prinzip P5). Das ist kein Fehler der "
            "Oberfläche, sondern die vorgesehene Reaktion."
        )
        st.stop()

    st.divider()
    if ergebnis.is_recommendation:
        _ansicht_empfehlung(ergebnis)
    else:
        _ansicht_konflikt(ergebnis)
    _nachweis(ergebnis)


if __name__ == "__main__":
    main()
