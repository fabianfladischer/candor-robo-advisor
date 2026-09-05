"""Anlageziel: Zielbetrag, Zieltermin, Sparplan, Priorität und Erfolgsanspruch.

Auch dieses Modul enthält keine Bewertung. Insbesondere steht hier **nicht**,
welche Erfolgswahrscheinlichkeit ein MUSS-Ziel mindestens erreichen muss — dieser
Schwellenwert liegt in ``policies/goals_v1.yaml`` und wird über
:meth:`~candor.domain.policy.GoalPolicy.minimum_success_probability` nachgeschlagen
(Prinzip P3). Solche Schwellenwerte sind Festlegungen des Anbieters und keine
Rechenvorschriften: In einer versionierten YAML-Datei sind sie ohne Blick in den Code
prüfbar, und ihre Fassung wandert in jeden Trace, sodass ein Ergebnis auch später dem
Regelstand zuordenbar bleibt, unter dem es entstanden ist.

Zur Unterscheidung MUSS/WUNSCH: Sie ist der Hebel, an dem der Konfliktmodus
hängt. Ein WUNSCH-Ziel darf mit 70 % Erfolgswahrscheinlichkeit verfolgt werden;
bei einem MUSS-Ziel — Altersvorsorge, Tilgung einer Restschuld — wäre dieselbe
Wahrscheinlichkeit eine in Kauf genommene Fehlplanung mit 30 % Eintrittsrisiko.
Genau dieser Unterschied entscheidet im Zielabgleich darüber, ob eine Empfehlung
ausgesprochen oder verweigert wird.

Alle Datumsangaben sind Kalenderdaten ohne Uhrzeit. Der Bezugszeitpunkt wird
ausdrücklich als Feld geführt und nicht der Systemuhr entnommen: Ein Ziel, dessen
Prüfung morgen ein anderes Ergebnis liefert als heute, wäre nicht reproduzierbar
(Prinzip P1).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from candor.domain.errors import IncompleteProfileError, InconsistentProfileError

#: Rechtsgrundlage der Zielerhebung.
GOAL_LEGAL_BASIS: Final = "Art. 54 Abs. 2 Buchst. a DelVO (EU) 2017/565"

#: Geldbetrag in Euro, nicht negativ, auf Cent genau.
Money = Annotated[Decimal, Field(ge=0, decimal_places=2)]

#: Wahrscheinlichkeit zwischen 0 und 1.
Probability = Annotated[Decimal, Field(ge=0, le=1)]

#: Monate je Jahr — reine Kalenderkonstante.
MONTHS_PER_YEAR: Final = 12


class GoalType(Enum):
    """Typ des Anlageziels.

    Der Typ bestimmt die Vorbelegungen — Standardhorizont, Standardpriorität
    und Mindest-Erfolgswahrscheinlichkeit. Sie stehen nicht hier, sondern im
    Zielkatalog in ``policies/goals_v1.yaml`` (Prinzip P3).

    Attributes:
        ALTERSVORSORGE: Aufbau einer zusätzlichen Altersversorgung.
        IMMOBILIEN_EIGENKAPITAL: Ansparen des Eigenkapitalanteils.
        AUSBILDUNG: Rücklage für Ausbildung oder Studium.
        LIQUIDITAETSPUFFER: Aufbau der Notfallreserve.
        VERMOEGENSAUFBAU: Vermögensaufbau ohne festen Verwendungszweck.
    """

    ALTERSVORSORGE = "Altersvorsorge"
    IMMOBILIEN_EIGENKAPITAL = "Eigenkapital für eine Immobilie"
    AUSBILDUNG = "Ausbildungssparen"
    LIQUIDITAETSPUFFER = "Liquiditätspuffer"
    VERMOEGENSAUFBAU = "Freier Vermögensaufbau"


class GoalPriority(Enum):
    """Verbindlichkeit des Anlageziels.

    Attributes:
        MUSS: Das Ziel muss erreicht werden. Eine Verfehlung hat für den Kunden
            wesentliche Folgen, etwa eine Versorgungslücke.
        WUNSCH: Das Ziel ist erstrebenswert, aber eine Verfehlung ist tragbar.
    """

    MUSS = "MUSS-Ziel"
    WUNSCH = "WUNSCH-Ziel"


class Goal(BaseModel):
    """Ein Anlageziel des Kunden.

    Rechtsgrundlage: Art. 54 Abs. 2 Buchst. a DelVO (EU) 2017/565 — die Firma hat
    Informationen über die Anlageziele einschließlich der Risikotoleranz und des
    Anlagehorizonts einzuholen.

    Attributes:
        goal_id: Fachliche Kennung des Ziels. Wandert in jeden Trace.
        name: Bezeichnung des Ziels im Klartext, etwa „Ruhestand“.
        goal_type: Typ des Ziels aus dem Zielkatalog. Bestimmt die
            Vorbelegungen, die beim Anlegen herangezogen wurden.
        priority: Verbindlichkeit des Ziels.
        reference_date: Stichtag der Erhebung. Alle Fristen beziehen sich darauf.
        target_date: Termin, zu dem der Zielbetrag verfügbar sein soll.
        target_amount: Angestrebter Betrag zum Zieltermin.
        initial_capital: Bereits vorhandenes, für dieses Ziel gebundenes Kapital.
        monthly_contribution: Geplante monatliche Sparrate.
        required_success_probability: Vom Kunden oder Berater abweichend
            festgelegte Mindest-Erfolgswahrscheinlichkeit. Bleibt das Feld leer,
            gilt der Schwellenwert aus dem Regelwerk. Ein hier gesetzter Wert
            überschreibt ihn nur nach oben; die Absicherung dagegen erfolgt beim
            Nachschlagen in der Policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    goal_id: Annotated[str, Field(min_length=1, max_length=64)]
    name: Annotated[str, Field(min_length=1, max_length=200)]
    goal_type: GoalType = GoalType.VERMOEGENSAUFBAU
    priority: GoalPriority
    reference_date: date
    target_date: date
    target_amount: Annotated[Decimal, Field(gt=0, decimal_places=2)]
    initial_capital: Money
    monthly_contribution: Money
    required_success_probability: Probability | None = None

    @model_validator(mode="after")
    def _pruefe_termine(self) -> Goal:
        """Prüft, dass der Zieltermin in der Zukunft liegt und der Horizont trägt.

        Returns:
            Das unveränderte Modell.

        Raises:
            InconsistentProfileError: Wenn der Zieltermin nicht nach dem Stichtag
                liegt oder der Horizont kürzer als ein voller Monat ist.
        """
        if self.target_date <= self.reference_date:
            raise InconsistentProfileError(
                ("goal.target_date", "goal.reference_date"),
                f"Der Zieltermin {self.target_date.isoformat()} liegt nicht nach dem "
                f"Stichtag {self.reference_date.isoformat()}.",
            )
        if self.horizon_months < 1:
            raise InconsistentProfileError(
                ("goal.target_date", "goal.reference_date"),
                "Der Anlagehorizont beträgt weniger als einen vollen Monat und ist "
                "für eine Monatsrenditesimulation nicht auswertbar.",
            )
        return self

    @model_validator(mode="after")
    def _pruefe_sparplan(self) -> Goal:
        """Prüft, dass überhaupt ein Weg zum Ziel besteht.

        Ohne Startkapital und ohne Sparrate ist jeder Zielbetrag größer null
        unerreichbar. Das ist kein Konflikt im Sinne der Konfliktdiagnose, den man
        quantifizieren könnte, sondern eine unvollständige Eingabe.

        Returns:
            Das unveränderte Modell.

        Raises:
            IncompleteProfileError: Wenn weder Startkapital noch Sparrate
                angegeben sind.
        """
        if self.initial_capital == 0 and self.monthly_contribution == 0:
            raise IncompleteProfileError(
                "goal.monthly_contribution",
                "Weder Startkapital noch monatliche Sparrate sind angegeben. Damit ist "
                "kein Zielbetrag erreichbar und keine Anlageentscheidung möglich.",
                GOAL_LEGAL_BASIS,
            )
        return self

    @property
    def horizon_months(self) -> int:
        """Anlagehorizont in vollen Monaten zwischen Stichtag und Zieltermin.

        Reine Kalenderableitung aus zwei Feldern des Modells, keine
        Bewertungslogik. Angebrochene Monate werden abgeschnitten: Ein Ziel gilt
        erst als erreichbar, wenn der Monat vollständig zur Verfügung steht.

        Returns:
            Zahl der vollen Monate, mindestens null.
        """
        monate = (self.target_date.year - self.reference_date.year) * MONTHS_PER_YEAR + (
            self.target_date.month - self.reference_date.month
        )
        if self.target_date.day < self.reference_date.day:
            monate -= 1
        return max(monate, 0)

    @property
    def horizon_years(self) -> Decimal:
        """Anlagehorizont in Jahren, auf zwei Stellen gerundet.

        Returns:
            Der Horizont in Jahren als ``Decimal``.
        """
        return (Decimal(self.horizon_months) / Decimal(MONTHS_PER_YEAR)).quantize(Decimal("0.01"))


def parse_goal(data: dict[str, object]) -> Goal:
    """Erzeugt ein Anlageziel aus einer Zuordnung und meldet Mängel fachlich.

    Args:
        data: Rohangaben des Ziels.

    Returns:
        Das geprüfte, unveränderliche Ziel.

    Raises:
        IncompleteProfileError: Wenn eine Pflichtangabe fehlt oder unzulässig ist.
        InconsistentProfileError: Wenn Angaben sich widersprechen.
    """
    try:
        return Goal.model_validate(data)
    except ValidationError as fehler:
        erster = fehler.errors()[0]
        pfad = ".".join(str(teil) for teil in erster["loc"])
        raise IncompleteProfileError(
            f"goal.{pfad}" if pfad else "goal",
            erster.get("msg", "unzulässiger Wert"),
            GOAL_LEGAL_BASIS,
        ) from fehler
