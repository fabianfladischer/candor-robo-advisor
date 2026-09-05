"""Ergebnisobjekte der Profilierung.

Die Typen liegen im Domänenkern, die Logik in :mod:`candor.profiling`. Damit
bleibt der Kern frei von Berechnung und die Ergebnisse frei von Abhängigkeiten
zur Auswertung.

Alle Objekte sind unveränderlich. Ein einmal ermitteltes Risikoprofil darf sich
nicht mehr ändern: Es ist die Grundlage der Geeignetheitserklärung, und eine
nachträglich veränderbare Grundlage wäre kein Nachweis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum

from candor.trace.events import JsonValue, TraceEvent


class Axis(Enum):
    """Die Achse, die das Ergebnis bindet.

    Attributes:
        TRAGFAEHIGKEIT: Die objektive Verlusttragfähigkeit begrenzt.
        BEREITSCHAFT: Die subjektive Risikobereitschaft begrenzt.
        KENNTNISSE: Der Komplexitätsdeckel nach Art. 55 DelVO begrenzt.
        GLIDEPATH: Die verbleibende Restlaufzeit begrenzt.
        KO_LIQUIDITAET: Das K.-o.-Kriterium der Liquiditätsreserve begrenzt.
    """

    TRAGFAEHIGKEIT = "Risikotragfähigkeit"
    BEREITSCHAFT = "Risikobereitschaft"
    KENNTNISSE = "Kenntnisse und Erfahrungen"
    GLIDEPATH = "Restlaufzeit (Glide-Path)"
    KO_LIQUIDITAET = "Liquiditätsreserve (K.-o.-Kriterium)"


@dataclass(frozen=True, slots=True)
class CriterionScore:
    """Punktwert eines einzelnen Bewertungskriteriums.

    Attributes:
        criterion: Kennung des Kriteriums, entspricht dem Schlüssel im Regelwerk.
        measured_value: Gemessener Eingabewert.
        unit: Maßeinheit des Eingabewerts.
        points: Vergebene Punkte von 0 bis 100.
        weight: Gewicht des Kriteriums.
        weighted_points: Beitrag zur Gesamtpunktzahl.
        basis: Angewandtes Band oder gewählte Ausprägung im Klartext.
    """

    criterion: str
    measured_value: Decimal
    unit: str
    points: int
    weight: Decimal
    weighted_points: Decimal
    basis: str


@dataclass(frozen=True, slots=True)
class CapacityResult:
    """Ergebnis der Risikotragfähigkeit (objektive Achse).

    Attributes:
        risk_class: Ermittelte Risikoklasse. Sie weicht von
            ``class_from_points`` ab, wenn das K.-o.-Kriterium greift.
        class_from_points: Klasse, die sich allein aus der Punktzahl über die
            Klassenschwellen ergibt. Sie steht neben ``risk_class`` in der
            Geeignetheitserklärung, weil ein Dokument, das nur die Endklasse
            nennt, im K.-o.-Fall eine Kausalität behauptet, die dort nicht
            gilt.
        total_points: Gewichtete Gesamtpunktzahl von 0 bis 100.
        scores: Einzelergebnisse je Kriterium in Auswertungsreihenfolge.
        ko_triggered: Ob das K.-o.-Kriterium gegriffen hat.
        ko_reason: Begründung des K.-o., sonst ``None``.
        policy_version: Fassung der angewandten Policy.
    """

    risk_class: int
    class_from_points: int
    total_points: Decimal
    scores: tuple[CriterionScore, ...]
    ko_triggered: bool
    ko_reason: str | None
    policy_version: str


@dataclass(frozen=True, slots=True)
class Inconsistency:
    """Ein erkannter Widerspruch in den Antworten zur Risikobereitschaft.

    Attributes:
        rule_id: Kennung der Konsistenzregel.
        description: Der Widerspruch im Klartext.
        answers: Die beteiligten Antworten.
    """

    rule_id: str
    description: str
    answers: dict[str, str]


@dataclass(frozen=True, slots=True)
class ToleranceResult:
    """Ergebnis der Risikobereitschaft (subjektive Achse).

    Attributes:
        risk_class: Ermittelte Risikoklasse.
        total_points: Gewichtete Gesamtpunktzahl von 0 bis 100.
        scores: Einzelergebnisse je Item.
        inconsistencies: Erkannte Widersprüche. Sie verändern die Punktzahl
            **nicht** — sie werden ausgewiesen, nicht weggerechnet.
        scenario_amounts: Die dem Kunden vorgelegten Verlustbeträge in Euro.
        reference_amount: Bezugsgröße, aus der die Beträge berechnet wurden —
            das über die Laufzeit eingezahlte Kapital.
        presented_amount: Der Betrag der Stufe, die der Kunde gewählt hat. Er
            steht in der Geeignetheitserklärung, damit das Dokument seine
            eigene Aussage über die Erhebungsform belegt und nicht nur
            behauptet.
        policy_version: Fassung der angewandten Policy.
    """

    risk_class: int
    total_points: Decimal
    scores: tuple[CriterionScore, ...]
    inconsistencies: tuple[Inconsistency, ...]
    scenario_amounts: dict[str, Decimal]
    reference_amount: Decimal
    presented_amount: Decimal
    policy_version: str

    @property
    def is_consistent(self) -> bool:
        """Gibt an, ob die Antworten widerspruchsfrei sind."""
        return not self.inconsistencies


@dataclass(frozen=True, slots=True)
class KnowledgeResult:
    """Ergebnis der Kenntnisse und Erfahrungen (Art. 55 DelVO (EU) 2017/565).

    Das Ergebnis ist **kein Score**, sondern eine Obergrenze. Kenntnisse können
    eine Risikoklasse begrenzen, aber niemals eine anheben, die Tragfähigkeit
    oder Bereitschaft nicht hergeben.

    Attributes:
        max_risk_class: Höchste zulässige Risikoklasse.
        qualifying_categories: Gattungen, deren Anforderungen erfüllt sind.
        failed_categories: Gattungen mit Angaben, deren Anforderungen nicht
            erfüllt sind, samt Begründung.
        policy_version: Fassung der angewandten Policy.
    """

    max_risk_class: int
    qualifying_categories: tuple[str, ...]
    failed_categories: dict[str, str]
    policy_version: str


@dataclass(frozen=True, slots=True)
class Divergence:
    """Auseinanderfallen der beiden Achsen.

    Attributes:
        capacity_class: Klasse der Tragfähigkeit.
        tolerance_class: Klasse der Bereitschaft.
        gap: Abstand in Stufen, stets nicht negativ.
        binding_axis: Die bindende, also niedrigere Achse.
        description: Der Befund im Klartext für die Geeignetheitserklärung.
    """

    capacity_class: int
    tolerance_class: int
    gap: int
    binding_axis: Axis
    description: str

    def as_note(self) -> str:
        """Formuliert den Divergenzhinweis für die Anzeige.

        Einzige Quelle dieses Satzes. Der Konfliktbericht
        (:func:`candor.engine.reconcile.build_diagnosis`) und die Oberfläche
        beziehen ihn beide von hier, damit der Kunde am Bildschirm und im
        Dokument denselben Wortlaut liest. Zwei nebeneinander gepflegte
        Fassungen desselben Satzes laufen sonst auseinander, ohne dass ein Test
        anschlägt — das ist der Befund, der diese Funktion veranlasst hat.

        Returns:
            Der Hinweis in einem Satz.
        """
        return (
            f"Tragfähigkeit (Klasse {self.capacity_class}) und Bereitschaft "
            f"(Klasse {self.tolerance_class}) liegen {self.gap} "
            "Stufen auseinander. Der Abstand ist Gesprächsstoff, nicht Rechengröße."
        )


@dataclass(frozen=True, slots=True)
class RiskProfile:
    """Zusammengeführtes Risikoprofil.

    Attributes:
        risk_class: Bindende Risikoklasse.
        binding_axis: Achse, die das Ergebnis bestimmt hat.
        class_before_glidepath: Klasse aus dem Minimum der drei Profilachsen,
            vor einer Deckelung durch die Restlaufzeit. Sie entspricht dem
            Ausgabewert des Trace-Ereignisses ``CLS-MIN``.
        axis_before_glidepath: Bindende Achse vor dieser Deckelung. Beide Felder
            überleben :func:`candor.engine.glidepath.apply_glidepath`, damit die
            Geeignetheitserklärung die Minimumrechnung mit denselben Werten
            wiedergeben kann, mit denen sie aufgezeichnet wurde.
        capacity: Ergebnis der Tragfähigkeit.
        tolerance: Ergebnis der Bereitschaft.
        knowledge: Ergebnis der Kenntnisse.
        divergence: Befund zum Auseinanderfallen der Achsen, sonst ``None``.
        glidepath_class: Obergrenze aus der Restlaufzeit, sofern angewandt.
        trace: Vollständiger Entscheidungspfad.
        trace_hash: Inhaltshash des Trace (Invariante I4).
        policy_versions: Fassungen aller beteiligten Policies.
    """

    risk_class: int
    binding_axis: Axis
    class_before_glidepath: int
    axis_before_glidepath: Axis
    capacity: CapacityResult
    tolerance: ToleranceResult
    knowledge: KnowledgeResult
    divergence: Divergence | None
    glidepath_class: int | None
    trace: tuple[TraceEvent, ...]
    trace_hash: str
    policy_versions: dict[str, str] = field(default_factory=dict)

    @property
    def has_divergence(self) -> bool:
        """Gibt an, ob die Achsen um mehr als eine Stufe auseinanderliegen."""
        return self.divergence is not None

    def as_summary(self) -> dict[str, JsonValue]:
        """Verdichtet das Profil für Anzeige und Bericht.

        Returns:
            Zuordnung der wesentlichen Kennzahlen.
        """
        return {
            "risikoklasse": self.risk_class,
            "bindende_achse": self.binding_axis.value,
            "tragfaehigkeit_klasse": self.capacity.risk_class,
            "tragfaehigkeit_punkte": str(self.capacity.total_points),
            "bereitschaft_klasse": self.tolerance.risk_class,
            "bereitschaft_punkte": str(self.tolerance.total_points),
            "kenntnisdeckel": self.knowledge.max_risk_class,
            "glidepath_deckel": self.glidepath_class,
            "ko_liquiditaet": self.capacity.ko_triggered,
            "divergenz": None if self.divergence is None else self.divergence.gap,
            "widersprueche": [eintrag.rule_id for eintrag in self.tolerance.inconsistencies],
            "trace_hash": self.trace_hash,
        }


# --- Erforderliche Rendite -------------------------------------------------


class ReturnCase(Enum):
    """Fachliche Einordnung einer Renditeanforderung.

    Jeder Fall ist ausdrücklich benannt. Es gibt keinen Zweig, der ``NaN`` oder
    ``None`` als Ergebnis liefert: Eine nicht auswertbare Zahl in einer
    Anlageplanung ist gefährlicher als eine unbequeme.

    Attributes:
        ERREICHBAR: Der Zielbetrag erfordert eine positive Rendite.
        OHNE_RENDITE_ERREICHBAR: Einzahlungen allein genügen; die erforderliche
            Rendite ist null oder negativ.
        BEREITS_ERREICHT: Das Startkapital erreicht den Zielbetrag bereits.
        UEBERERFUELLT: Der Zielbetrag liegt unter dem, was der Sparplan selbst
            bei extremen Verlusten liefert. Ausgewiesen wird die untere
            Suchgrenze als Schranke.
        NICHT_ERREICHBAR: Der Zielbetrag ist im Suchbereich nicht erreichbar.
            Ausgewiesen wird die obere Suchgrenze als Schranke.
    """

    ERREICHBAR = "erreichbar"
    OHNE_RENDITE_ERREICHBAR = "ohne Rendite erreichbar"
    BEREITS_ERREICHT = "durch Startkapital bereits erreicht"
    UEBERERFUELLT = "vom Sparplan übererfüllt"
    NICHT_ERREICHBAR = "im Suchbereich nicht erreichbar"


class SolutionMethod(Enum):
    """Verfahren, mit dem die Rendite bestimmt wurde.

    Attributes:
        GESCHLOSSEN: Analytisch gelöst.
        EXAKT_NULL: Der Zielbetrag entspricht exakt der Summe der Einzahlungen.
        BRENT: Numerisch über das Brent-Verfahren.
        SCHRANKE: Keine Lösung im Suchbereich; ausgewiesen ist die Grenze.
    """

    GESCHLOSSEN = "geschlossene Lösung"
    EXAKT_NULL = "exakt null"
    BRENT = "Brent-Verfahren"
    SCHRANKE = "Suchbereichsgrenze"


@dataclass(frozen=True, slots=True)
class ReturnSolution:
    """Lösung für einen konkreten Zielbetrag.

    Attributes:
        target_amount: Zugrunde gelegter Zielbetrag.
        monthly_rate: Erforderliche Rendite je Monat.
        annual_rate: Daraus geometrisch annualisierte Jahresrendite.
        case: Fachliche Einordnung.
        method: Verwendetes Lösungsverfahren.
    """

    target_amount: Decimal
    monthly_rate: Decimal
    annual_rate: Decimal
    case: ReturnCase
    method: SolutionMethod


@dataclass(frozen=True, slots=True)
class RequiredReturn:
    """Erforderliche Rendite eines Anlageziels, nominal und real getrennt.

    Zwei Lesarten desselben Ziels werden gesondert ausgewiesen, weil sie
    verschiedene Fragen beantworten:

    * :attr:`nominal` — „Welche Rendite brauche ich, damit am Zieltermin der
      genannte Betrag auf dem Konto steht?"
    * :attr:`purchasing_power` — „Welche Rendite brauche ich, damit dieser
      Betrag am Zieltermin noch dieselbe Kaufkraft hat wie heute?"

    Ein Zusammenrechnen beider wäre irreführend; welche Lesart gemeint ist,
    entscheidet der Kunde.

    Attributes:
        horizon_months: Anlagehorizont in vollen Monaten.
        initial_capital: Startkapital.
        monthly_contribution: Monatliche Sparrate, nachschüssig.
        inflation_annual: Verwendete jährliche Inflationsannahme.
        value_without_return: Endwert ohne jede Rendite, also die Summe aller
            Einzahlungen zuzüglich Startkapital.
        nominal: Lösung für den Zielbetrag wie angegeben.
        purchasing_power: Lösung für den inflationsbereinigten Zielbetrag.
        warnings: Hinweise, die das Ergebnis nicht ungültig machen, aber bei der
            Auslegung zu beachten sind.
        policy_version: Fassung der Policy, aus der die Inflationsannahme stammt.
    """

    horizon_months: int
    initial_capital: Decimal
    monthly_contribution: Decimal
    inflation_annual: Decimal
    value_without_return: Decimal
    nominal: ReturnSolution
    purchasing_power: ReturnSolution
    warnings: tuple[str, ...]
    policy_version: str

    @property
    def real_annual(self) -> Decimal:
        """Reale Jahresrendite zum nominalen Zielbetrag nach Fisher.

        Returns:
            Die um die Inflationsannahme bereinigte Jahresrendite.
        """
        return deflate(self.nominal.annual_rate, self.inflation_annual)

    def as_summary(self) -> dict[str, JsonValue]:
        """Verdichtet das Ergebnis für Anzeige und Bericht.

        Returns:
            Zuordnung der wesentlichen Kennzahlen.
        """
        return {
            "horizont_monate": self.horizon_months,
            "einzahlungen_gesamt": str(self.value_without_return),
            "zielbetrag_nominal": str(self.nominal.target_amount),
            "erforderliche_rendite_nominal_pa": str(self.nominal.annual_rate),
            "erforderliche_rendite_real_pa": str(self.real_annual),
            "zielbetrag_kaufkrafterhaltend": str(self.purchasing_power.target_amount),
            "erforderliche_rendite_kaufkraft_pa": str(self.purchasing_power.annual_rate),
            "inflationsannahme_pa": str(self.inflation_annual),
            "fall": self.nominal.case.value,
            "verfahren": self.nominal.method.value,
            "hinweise": list(self.warnings),
        }


def deflate(nominal_rate: Decimal, inflation_rate: Decimal, places: int = 6) -> Decimal:
    """Rechnet eine nominale Rendite nach Fisher in eine reale um.

    Verwendet wird die exakte Beziehung ``(1 + r_real) = (1 + r_nom) / (1 + π)``
    und nicht die Näherung ``r_real ≈ r_nom − π``. Über zwanzig Jahre summiert
    sich der Unterschied der Näherung auf mehrere Prozentpunkte Endvermögen.

    Args:
        nominal_rate: Nominale Rendite als Dezimalzahl.
        inflation_rate: Inflationsrate als Dezimalzahl.
        places: Zahl der Nachkommastellen des Ergebnisses.

    Returns:
        Die reale Rendite als Dezimalzahl.
    """
    faktor = (Decimal(1) + nominal_rate) / (Decimal(1) + inflation_rate)
    quant = Decimal(1).scaleb(-places)
    return (faktor - Decimal(1)).quantize(quant, rounding=ROUND_HALF_UP)


# --- Mehrzielfähigkeit -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoalAllocation:
    """Zuteilung der Sparrate an ein einzelnes Ziel.

    Attributes:
        goal_id: Kennung des Ziels.
        priority: Verbindlichkeit des Ziels.
        rank: Rang in der Zuteilungsreihenfolge, beginnend bei eins.
        requested: Gewünschte monatliche Sparrate.
        allocated: Tatsächlich zugeteilte monatliche Sparrate.
    """

    goal_id: str
    priority: str
    rank: int
    requested: Decimal
    allocated: Decimal

    @property
    def fully_funded(self) -> bool:
        """Gibt an, ob die gewünschte Sparrate vollständig zugeteilt wurde."""
        return self.allocated >= self.requested

    @property
    def shortfall(self) -> Decimal:
        """Fehlbetrag zwischen gewünschter und zugeteilter Sparrate."""
        return self.requested - self.allocated


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """Ergebnis der Sparratenzuteilung auf mehrere Ziele.

    Attributes:
        budget: Verfügbare monatliche Sparrate insgesamt.
        allocations: Zuteilungen in Zuteilungsreihenfolge.
        policy_version: Fassung der angewandten Policy.
    """

    budget: Decimal
    allocations: tuple[GoalAllocation, ...]
    policy_version: str

    @property
    def allocated_total(self) -> Decimal:
        """Summe aller zugeteilten Sparraten."""
        return sum((eintrag.allocated for eintrag in self.allocations), Decimal(0))

    @property
    def unallocated(self) -> Decimal:
        """Nicht zugeteilter Rest des Budgets."""
        return self.budget - self.allocated_total

    @property
    def underfunded(self) -> tuple[str, ...]:
        """Kennungen der Ziele, die weniger als gewünscht erhalten haben."""
        return tuple(eintrag.goal_id for eintrag in self.allocations if not eintrag.fully_funded)

    @property
    def is_complete(self) -> bool:
        """Gibt an, ob alle Ziele vollständig bedient werden konnten."""
        return not self.underfunded


# --- Zielabgleich und Konfliktdiagnose -------------------------------------


class ReconcileOutcome(Enum):
    """Ausgang des Abgleichs zwischen gefordertem und erreichtem Erfolg.

    Attributes:
        ERREICHBAR: Die erreichte Erfolgswahrscheinlichkeit liegt mit Abstand
            über der geforderten. Eine Empfehlung ist zulässig.
        ERREICHBAR_MIT_PUFFER: Die Schwelle ist erreicht, aber der Abstand liegt
            innerhalb des Pufferbands. Eine Empfehlung ist zulässig, ergeht
            jedoch mit ausdrücklichem Hinweis auf den geringen Abstand.
        KONFLIKT: Die Schwelle ist nicht erreicht. Es ergeht **keine**
            Empfehlung, sondern ein Konfliktbericht.
    """

    ERREICHBAR = "erreichbar"
    ERREICHBAR_MIT_PUFFER = "erreichbar, geringer Abstand zur Schwelle"
    KONFLIKT = "Konflikt — keine Empfehlung"

    @property
    def allows_recommendation(self) -> bool:
        """Gibt an, ob eine Anlageempfehlung ausgesprochen werden darf."""
        return self is not ReconcileOutcome.KONFLIKT


class LeverId(Enum):
    """Kennung einer Stellschraube des Konfliktberichts.

    Attributes:
        SPARRATE: Erforderliche monatliche Sparrate bei unverändertem Ziel und
            Horizont.
        HORIZONT: Erforderliche Verlängerung des Horizonts bei unveränderter
            Sparrate.
        ZIELBETRAG: Erreichbarer Zielbetrag bei unveränderter Sparrate und
            unverändertem Horizont.
    """

    SPARRATE = "S1"
    HORIZONT = "S2"
    ZIELBETRAG = "S3"


class Feasibility(Enum):
    """Bewertung der Umsetzbarkeit einer Stellschraube.

    Attributes:
        UMSETZBAR: Die Stellschraube lässt sich im Rahmen der erhobenen
            Verhältnisse umsetzen.
        NICHT_UMSETZBAR: Die Stellschraube überschreitet eine erhobene Grenze —
            etwa die freie Sparquote des Haushalts oder einen festen Zieltermin.
            Sie wird ausgewiesen, aber **nicht als Vorschlag angeboten**.
        NICHT_ERREICHBAR: Auch am Rand des Suchbereichs wird die geforderte
            Erfolgswahrscheinlichkeit nicht erreicht. Die Stellschraube löst den
            Konflikt nicht.
    """

    UMSETZBAR = "umsetzbar"
    NICHT_UMSETZBAR = "nicht umsetzbar"
    NICHT_ERREICHBAR = "löst den Konflikt nicht"


@dataclass(frozen=True, slots=True)
class Lever:
    """Eine quantifizierte Stellschraube des Konfliktberichts.

    Attributes:
        lever_id: Kennung der Stellschraube.
        name: Bezeichnung im Klartext.
        unit: Maßeinheit des Werts.
        current_value: Heutiger Wert.
        required_value: Wert, bei dem die geforderte Erfolgswahrscheinlichkeit
            erreicht wird.
        achieved_probability: Erfolgswahrscheinlichkeit bei diesem Wert.
        feasibility: Bewertung der Umsetzbarkeit.
        feasibility_note: Begründung der Bewertung im Klartext.
        iterations: Zahl der Iterationen des Suchverfahrens.
        method: Verwendetes Verfahren.
    """

    lever_id: LeverId
    name: str
    unit: str
    current_value: Decimal
    required_value: Decimal
    achieved_probability: float
    feasibility: Feasibility
    feasibility_note: str
    iterations: int
    method: str

    @property
    def difference(self) -> Decimal:
        """Abstand zwischen erforderlichem und heutigem Wert."""
        return self.required_value - self.current_value

    @property
    def is_offered(self) -> bool:
        """Gibt an, ob die Stellschraube als Vorschlag angeboten werden darf."""
        return self.feasibility is Feasibility.UMSETZBAR


@dataclass(frozen=True, slots=True)
class ConflictReport:
    """Konfliktbericht — der Ersatz für eine unzulässige Empfehlung.

    Rechtsgrundlage: Art. 54 Abs. 10 DelVO (EU) 2017/565. Ist kein geeignetes
    Produkt vorhanden, darf keine Empfehlung ergehen. An ihre Stelle tritt
    dieser Bericht: die Diagnose in Stichpunkten und drei numerisch gelöste
    Stellschrauben, jede mit Machbarkeitsbewertung.

    Attributes:
        client_id: Kennung des Kunden.
        goal_id: Kennung des Anlageziels.
        required_probability: Geforderte Mindest-Erfolgswahrscheinlichkeit.
        achieved_probability: Unter der zulässigen Risikoklasse erreichte
            Erfolgswahrscheinlichkeit.
        risk_class: Zulässige Risikoklasse.
        binding_axis: Achse, die die Risikoklasse gebunden hat.
        diagnosis: Befund in Stichpunkten.
        levers: Die drei Stellschrauben.
        legal_basis: Rechtsgrundlage der Empfehlungsverweigerung.
        policy_versions: Fassungen aller beteiligten Policies.
    """

    client_id: str
    goal_id: str
    required_probability: float
    achieved_probability: float
    risk_class: int
    binding_axis: Axis
    diagnosis: tuple[str, ...]
    levers: tuple[Lever, ...]
    legal_basis: str
    policy_versions: dict[str, str]

    @property
    def gap(self) -> float:
        """Fehlender Abstand zur geforderten Erfolgswahrscheinlichkeit."""
        return self.required_probability - self.achieved_probability

    @property
    def offered_levers(self) -> tuple[Lever, ...]:
        """Stellschrauben, die als Vorschlag angeboten werden dürfen."""
        return tuple(hebel for hebel in self.levers if hebel.is_offered)

    @property
    def has_way_out(self) -> bool:
        """Gibt an, ob wenigstens eine Stellschraube umsetzbar ist."""
        return bool(self.offered_levers)

    def lever(self, lever_id: LeverId) -> Lever:
        """Liefert eine Stellschraube anhand ihrer Kennung.

        Args:
            lever_id: Gesuchte Kennung.

        Returns:
            Die Stellschraube.

        Raises:
            KeyError: Wenn die Stellschraube nicht enthalten ist.
        """
        for hebel in self.levers:
            if hebel.lever_id is lever_id:
                return hebel
        msg = f"Stellschraube {lever_id.value} ist im Bericht nicht enthalten."
        raise KeyError(msg)


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """Ergebnis des Zielabgleichs.

    Attributes:
        outcome: Ausgang des Abgleichs.
        client_id: Kennung des Kunden.
        goal_id: Kennung des Anlageziels.
        risk_class: Zulässige Risikoklasse.
        required_probability: Geforderte Mindest-Erfolgswahrscheinlichkeit.
        achieved_probability: Erreichte Erfolgswahrscheinlichkeit.
        buffer_band: Breite des Pufferbands aus dem Regelwerk.
        weights: Verwendete Zielgewichte je Assetklasse.
        percentiles: Perzentile der Endwertverteilung.
        expected_shortfall: Erwarteter Fehlbetrag im Misserfolgsfall.
        median_max_drawdown: Medianer größter Rückgang über die Pfade.
        conflict_report: Konfliktbericht, sofern der Ausgang ein Konflikt ist.
        trace: Vollständiger Entscheidungspfad.
        trace_hash: Inhaltshash des Trace.
        policy_versions: Fassungen aller beteiligten Policies.
    """

    outcome: ReconcileOutcome
    client_id: str
    goal_id: str
    risk_class: int
    required_probability: float
    achieved_probability: float
    buffer_band: float
    weights: dict[str, Decimal]
    percentiles: dict[int, float]
    expected_shortfall: float
    median_max_drawdown: float
    conflict_report: ConflictReport | None
    trace: tuple[TraceEvent, ...]
    trace_hash: str
    policy_versions: dict[str, str]

    @property
    def allows_recommendation(self) -> bool:
        """Gibt an, ob eine Anlageempfehlung ausgesprochen werden darf."""
        return self.outcome.allows_recommendation

    @property
    def margin(self) -> float:
        """Abstand zwischen erreichter und geforderter Erfolgswahrscheinlichkeit."""
        return self.achieved_probability - self.required_probability

    def as_summary(self) -> dict[str, JsonValue]:
        """Verdichtet das Ergebnis für Anzeige, Trace und Bericht.

        Returns:
            Zuordnung der wesentlichen Kennzahlen.
        """
        return {
            "ausgang": self.outcome.value,
            "empfehlung_zulaessig": self.allows_recommendation,
            "risikoklasse": self.risk_class,
            "gefordert": round(self.required_probability, 6),
            "erreicht": round(self.achieved_probability, 6),
            "abstand": round(self.margin, 6),
            "pufferband": round(self.buffer_band, 6),
            "erwarteter_fehlbetrag_im_misserfolg": round(self.expected_shortfall, 2),
            "median_max_drawdown": round(self.median_max_drawdown, 6),
            "trace_hash": self.trace_hash,
        }
