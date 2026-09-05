"""Modelle des versionierten Regelwerks aus ``policies/*.yaml``.

Hier stehen die **Typen** des Regelwerks, nicht seine Werte. Die Werte liegen in
YAML; eingelesen wird in :mod:`candor.data.policy_loader`, damit der Domänenkern
frei von Dateizugriffen bleibt (Prinzip P4).

Jedes Modell verbietet unbekannte Schlüssel. Ein Tippfehler in einer Policy —
``mindestkenntniss`` statt ``mindestkenntnis`` — führt damit zum Abbruch und
nicht dazu, dass eine Regel stillschweigend ihren Vorgabewert behält. In einem
System, dessen Ausgabe eine Anlageempfehlung ist, wäre Letzteres der gefährlichere
Fehler: Er bliebe unbemerkt und veränderte trotzdem jedes Ergebnis.

Die Modelle prüfen zusätzlich die **innere Stimmigkeit** des Regelwerks, etwa ob
sich Dimensionsgewichte zu eins addieren oder ob die Aktienquote über die
Risikoklassen monoton steigt (Invariante I6). Diese Prüfungen gehören hierher
und nicht in die Tests: Ein widersprüchliches Regelwerk soll gar nicht erst
ladbar sein.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Final

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from candor.domain.client import EmploymentType, InstrumentCategory, KnowledgeLevel
from candor.domain.errors import PolicyError
from candor.domain.goal import GoalPriority, GoalType

#: Zulässiger Bereich der Risikoklassen.
MIN_RISK_CLASS: Final = 1
MAX_RISK_CLASS: Final = 5

#: Kennungen der Assetklassen. Muss zu ``candor.data.universe_meta.ASSET_KEYS``
#: passen; der Abgleich erfolgt in den Tests, damit der Domänenkern nicht von
#: der Datenschicht abhängt (Prinzip P4).
ASSET_KEYS: Final[tuple[str, ...]] = (
    "EQ_DEV",
    "EQ_EM",
    "EQ_EUR",
    "BD_GOV",
    "BD_CORP",
    "BD_SHORT",
    "RE",
    "GOLD",
)

#: Toleranz beim Vergleich von Dezimalsummen gegen eins.
SUM_TOLERANCE: Final = Decimal("0.0001")

#: Mindestzahl an Verlustszenarien. Mit weniger als zwei Stufen liesse sich
#: keine Abstufung der Bereitschaft erfassen.
MIN_LOSS_SCENARIOS: Final = 2

#: Mindestzahl an Bedingungen je Konsistenzregel. Ein Widerspruch entsteht erst
#: aus dem Zusammentreffen zweier Angaben; eine einzelne Antwort kann nicht
#: widerspruechlich sein.
MIN_CONSISTENCY_CONDITIONS: Final = 2


def _priority_by_name(value: object) -> object:
    """Lässt im Regelwerk den Namen einer Priorität statt ihres Werts zu.

    In den YAML-Dateien steht ``MUSS``, nicht ``MUSS-Ziel``. Der Name ist dort
    die lesbarere Schreibweise und zugleich diejenige, die auch in
    ``zuteilungsreihenfolge`` verwendet wird. Beide Schreibweisen werden
    angenommen; alles andere führt weiterhin zu einem Schemafehler.

    Args:
        value: Rohwert aus der Policy.

    Returns:
        Die aufgelöste Priorität oder der unveränderte Wert zur weiteren Prüfung.
    """
    if isinstance(value, str) and value in GoalPriority.__members__:
        return GoalPriority[value]
    return value


#: Priorität, im Regelwerk über ihren Namen angegeben.
PriorityByName = Annotated[GoalPriority, BeforeValidator(_priority_by_name)]

RiskClass = Annotated[int, Field(ge=MIN_RISK_CLASS, le=MAX_RISK_CLASS)]
Share = Annotated[Decimal, Field(ge=0, le=1)]
Points = Annotated[int, Field(ge=0, le=100)]


class _PolicyModel(BaseModel):
    """Basismodell aller Regelwerksteile: unbekannte Schlüssel sind verboten."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PolicyHeader(_PolicyModel):
    """Pflichtkopf jeder Policy-Datei.

    Attributes:
        version: Fassung des Regelwerks, etwa ``"1.0.0"``. Wandert in jeden Trace
            und in jede Geeignetheitserklärung.
        gueltig_ab: Datum, ab dem die Fassung anzuwenden ist.
        beschreibung: Zweck der Datei in einem Satz.
        quelle: Herkunft der Werte — Rechtsnorm, Fachliteratur oder eigene
            Festlegung. Bei eigener Festlegung ist das ausdrücklich zu benennen.
    """

    version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    gueltig_ab: date
    beschreibung: Annotated[str, Field(min_length=10)]
    quelle: Annotated[str, Field(min_length=5)]


# --- Risikotragfähigkeit ---------------------------------------------------


class ScoreBand(_PolicyModel):
    """Punktband einer stetigen Dimension.

    Attributes:
        bis: Obergrenze des Bandes, einschließlich. ``None`` bedeutet „offen nach
            oben" und ist genau einmal je Dimension zulässig, als letztes Band.
        punkte: Punktwert, wenn der Messwert in dieses Band fällt.
    """

    bis: Decimal | None = None
    punkte: Points


class CapacityDimension(_PolicyModel):
    """Eine Dimension der Risikotragfähigkeit.

    Attributes:
        beschreibung: Was die Dimension misst.
        gewicht: Anteil an der Gesamtpunktzahl.
        einheit: Maßeinheit bei stetigen Dimensionen.
        baender: Punktbänder bei stetigen Dimensionen.
        optionen: Punktwerte je Ausprägung bei kategorialen Dimensionen.
    """

    beschreibung: Annotated[str, Field(min_length=5)]
    gewicht: Share
    einheit: str | None = None
    baender: tuple[ScoreBand, ...] | None = None
    optionen: dict[str, Points] | None = None

    @model_validator(mode="after")
    def _pruefe_dimension(self) -> CapacityDimension:
        """Prüft, dass genau eine Bewertungsform vorliegt und Bänder aufsteigen.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Bei fehlender, doppelter oder unsortierter Bewertungsform.
        """
        if (self.baender is None) == (self.optionen is None):
            raise PolicyError(
                f"Dimension '{self.beschreibung}': genau eines von 'baender' oder "
                "'optionen' ist anzugeben."
            )
        if self.baender is not None:
            self._pruefe_baender(self.baender)
        return self

    @staticmethod
    def _pruefe_baender(baender: tuple[ScoreBand, ...]) -> None:
        """Prüft Reihenfolge und offenes Ende der Punktbänder.

        Args:
            baender: Zu prüfende Bänder.

        Raises:
            PolicyError: Wenn die Bänder leer, unsortiert oder nicht nach oben
                offen sind.
        """
        if not baender:
            raise PolicyError("Eine stetige Dimension benötigt mindestens ein Band.")
        if baender[-1].bis is not None:
            raise PolicyError(
                "Das letzte Band einer stetigen Dimension muss nach oben offen sein "
                "('bis: null'), sonst bliebe ein Messwert unbewertet."
            )
        grenzen = [band.bis for band in baender[:-1]]
        if any(grenze is None for grenze in grenzen):
            raise PolicyError("Nur das letzte Band darf nach oben offen sein.")
        werte = [grenze for grenze in grenzen if grenze is not None]
        if werte != sorted(werte) or len(set(werte)) != len(werte):
            raise PolicyError("Die Bandgrenzen müssen streng aufsteigend sortiert sein.")


class ClassThreshold(_PolicyModel):
    """Schwelle, ab der eine Risikoklasse erreicht wird.

    Attributes:
        ab_punkte: Untere Punktschwelle, einschließlich.
        klasse: Erreichte Risikoklasse.
    """

    ab_punkte: Annotated[Decimal, Field(ge=0, le=100)]
    klasse: RiskClass


class ScorePolicy(PolicyHeader):
    """Gemeinsame Struktur der punktebasierten Policies.

    Attributes:
        dimensionen: Bewertungsdimensionen, Schlüssel ist die Dimensionskennung.
        klassenschwellen: Abbildung der Gesamtpunktzahl auf eine Risikoklasse.
    """

    dimensionen: dict[str, CapacityDimension]
    klassenschwellen: tuple[ClassThreshold, ...]

    @model_validator(mode="after")
    def _pruefe_gewichte_und_schwellen(self) -> ScorePolicy:
        """Prüft Gewichtssumme, Schwellenordnung und Klassenabdeckung.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn Gewichte nicht eins ergeben, Schwellen nicht
                aufsteigen oder nicht alle Klassen erreichbar sind.
        """
        if not self.dimensionen:
            raise PolicyError("Es ist mindestens eine Dimension anzugeben.")
        summe = sum((dimension.gewicht for dimension in self.dimensionen.values()), Decimal(0))
        if abs(summe - Decimal(1)) > SUM_TOLERANCE:
            raise PolicyError(
                f"Die Dimensionsgewichte ergeben {summe} statt 1. Ohne diese Bedingung "
                "wäre die Gesamtpunktzahl nicht auf 0 bis 100 normiert."
            )
        if not self.klassenschwellen:
            raise PolicyError("Es ist mindestens eine Klassenschwelle anzugeben.")
        punkte = [schwelle.ab_punkte for schwelle in self.klassenschwellen]
        klassen = [schwelle.klasse for schwelle in self.klassenschwellen]
        if punkte != sorted(punkte) or len(set(punkte)) != len(punkte):
            raise PolicyError("Die Klassenschwellen müssen streng aufsteigend sortiert sein.")
        if klassen != sorted(klassen):
            raise PolicyError("Höhere Punktzahlen müssen zu höheren Klassen führen (Monotonie).")
        if punkte[0] != 0:
            raise PolicyError("Die unterste Klassenschwelle muss bei 0 Punkten beginnen.")
        fehlend = set(range(MIN_RISK_CLASS, MAX_RISK_CLASS + 1)) - set(klassen)
        if fehlend:
            raise PolicyError(f"Nicht erreichbare Risikoklassen: {sorted(fehlend)}.")
        return self


class CapacityPolicy(ScorePolicy):
    """Punktetabelle der Risikotragfähigkeit (objektive Achse).

    Attributes:
        mindestreserve_monate: K.-o.-Schwelle der Liquiditätsreserve. Liegt die
            Reserve darunter, gilt Risikoklasse 1 — ohne Verrechnung mit den
            übrigen Dimensionen.
    """

    mindestreserve_monate: Annotated[Decimal, Field(ge=0, le=120)]

    @model_validator(mode="after")
    def _pruefe_beschaeftigungsarten(self) -> CapacityPolicy:
        """Prüft, dass jede Beschäftigungsart bewertet ist.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn eine Ausprägung fehlt oder unbekannt ist.
        """
        dimension = self.dimensionen.get("einkommensstabilitaet")
        if dimension is None or dimension.optionen is None:
            raise PolicyError(
                "Die Dimension 'einkommensstabilitaet' fehlt oder ist nicht kategorial."
            )
        erwartet = {art.name for art in EmploymentType}
        _pruefe_schluesselmenge(set(dimension.optionen), erwartet, "einkommensstabilitaet")
        return self


class LossScenarioStep(_PolicyModel):
    """Eine Stufe der Verlustszenarien.

    Attributes:
        id: Kennung der Stufe, zugleich Antwortoption der zugehörigen Dimension.
        verlust_anteil: Anteil des Bezugsbetrags, der als Verlust dargestellt
            wird. Dient ausschließlich der Berechnung des Eurobetrags für den
            Fragebogen. Der berechnete Betrag geht in keine Bewertung ein — die
            Punkte hängen allein an der gewählten Stufe. Für die Bezugsgröße
            gilt das nicht; sie ist zugleich Zähler des Klumpenrisikos.
    """

    id: Annotated[str, Field(min_length=1, max_length=16)]
    verlust_anteil: Share


class LossScenarioSet(_PolicyModel):
    """Definition der in Euro dargestellten Verlustszenarien.

    Attributes:
        bezugsgroesse: Kennung der Größe, auf die sich die Anteile beziehen.
        dimension: Dimension, deren Antwortoptionen die Stufen bilden.
        stufen: Die Stufen in aufsteigender Verlusthöhe.
    """

    bezugsgroesse: Annotated[str, Field(min_length=3)]
    dimension: Annotated[str, Field(min_length=3)]
    stufen: tuple[LossScenarioStep, ...]

    @model_validator(mode="after")
    def _pruefe_stufen(self) -> LossScenarioSet:
        """Prüft Anzahl, Eindeutigkeit und Reihenfolge der Stufen.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn Stufen fehlen, doppelt vorkommen oder nicht
                aufsteigend sortiert sind.
        """
        if len(self.stufen) < MIN_LOSS_SCENARIOS:
            raise PolicyError("Es sind mindestens zwei Verlustszenarien anzugeben.")
        kennungen = [stufe.id for stufe in self.stufen]
        if len(set(kennungen)) != len(kennungen):
            raise PolicyError("Die Kennungen der Verlustszenarien müssen eindeutig sein.")
        anteile = [stufe.verlust_anteil for stufe in self.stufen]
        if anteile != sorted(anteile) or len(set(anteile)) != len(anteile):
            raise PolicyError("Die Verlustanteile müssen streng aufsteigend sortiert sein.")
        return self


class ConsistencyRule(_PolicyModel):
    """Regel zur Erkennung widersprüchlicher Antworten.

    Attributes:
        id: Kennung der Regel, wandert unverändert in den Trace.
        beschreibung: Der Widerspruch im Klartext.
        bedingungen: Zuordnung Dimension auf Antwortoptionen. Der Widerspruch
            gilt als erkannt, wenn **alle** Bedingungen zutreffen.
    """

    id: Annotated[str, Field(min_length=3, max_length=32)]
    beschreibung: Annotated[str, Field(min_length=20)]
    bedingungen: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def _pruefe_bedingungen(self) -> ConsistencyRule:
        """Prüft, dass die Regel überhaupt greifen kann.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn weniger als zwei Bedingungen vorliegen oder eine
                Bedingung ohne Antwortoption bleibt.
        """
        if len(self.bedingungen) < MIN_CONSISTENCY_CONDITIONS:
            raise PolicyError(
                f"Konsistenzregel {self.id!r}: Ein Widerspruch benötigt mindestens zwei "
                "Bedingungen."
            )
        for dimension, optionen in self.bedingungen.items():
            if not optionen:
                raise PolicyError(
                    f"Konsistenzregel {self.id!r}: Bedingung {dimension!r} ohne Antwortoption."
                )
        return self


class TolerancePolicy(ScorePolicy):
    """Punktetabelle der Risikobereitschaft (subjektive Achse).

    Attributes:
        verlustszenarien: In Euro darzustellende Verlustszenarien.
        konsistenzregeln: Regeln zur Erkennung widersprüchlicher Antworten.
    """

    verlustszenarien: LossScenarioSet
    konsistenzregeln: tuple[ConsistencyRule, ...] = ()

    @model_validator(mode="after")
    def _pruefe_verweise(self) -> TolerancePolicy:
        """Prüft, dass Szenarien und Konsistenzregeln auf vorhandene Items zeigen.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn eine Dimension oder Antwortoption nicht existiert.
        """
        dimension = self.dimensionen.get(self.verlustszenarien.dimension)
        if dimension is None or dimension.optionen is None:
            raise PolicyError(
                f"Die Verlustszenarien verweisen auf die Dimension "
                f"{self.verlustszenarien.dimension!r}, die nicht existiert oder nicht "
                "kategorial ist."
            )
        _pruefe_schluesselmenge(
            {stufe.id for stufe in self.verlustszenarien.stufen},
            set(dimension.optionen),
            "verlustszenarien.stufen",
        )
        kennungen = [regel.id for regel in self.konsistenzregeln]
        if len(set(kennungen)) != len(kennungen):
            raise PolicyError("Die Kennungen der Konsistenzregeln müssen eindeutig sein.")
        for regel in self.konsistenzregeln:
            for name, optionen in regel.bedingungen.items():
                item = self.dimensionen.get(name)
                if item is None or item.optionen is None:
                    raise PolicyError(
                        f"Konsistenzregel {regel.id!r} verweist auf die unbekannte oder "
                        f"nicht kategoriale Dimension {name!r}."
                    )
                unbekannt = set(optionen) - set(item.optionen)
                if unbekannt:
                    raise PolicyError(
                        f"Konsistenzregel {regel.id!r}: unbekannte Antwortoption(en) "
                        f"{sorted(unbekannt)} für {name!r}."
                    )
        return self


# --- Kenntnisse und Erfahrungen -------------------------------------------


class CategoryRequirement(_PolicyModel):
    """Mindestanforderung an eine Instrumentengattung.

    Attributes:
        schaltet_risikoklasse_frei: Höchste Risikoklasse, die Kenntnisse in
            dieser Gattung zulassen.
        mindestkenntnis: Erforderlicher Kenntnisstand.
        mindesterfahrung_jahre: Erforderliche Erfahrungsdauer in Jahren.
        mindesttransaktionen_pro_jahr: Erforderliche Zahl an Geschäften je Jahr.
    """

    schaltet_risikoklasse_frei: RiskClass
    mindestkenntnis: KnowledgeLevel
    mindesterfahrung_jahre: Annotated[Decimal, Field(ge=0, le=80)]
    mindesttransaktionen_pro_jahr: Annotated[int, Field(ge=0, le=1000)]


class KnowledgePolicy(PolicyHeader):
    """Abbildung von Kenntnissen und Erfahrungen auf einen Komplexitätsdeckel.

    Rechtsgrundlage: Art. 55 DelVO (EU) 2017/565.

    Attributes:
        ohne_nachweis_max_risikoklasse: Höchste Risikoklasse, wenn zu keiner
            Gattung die Anforderungen erfüllt sind.
        gattungen: Anforderung je Instrumentengattung.
    """

    ohne_nachweis_max_risikoklasse: RiskClass
    gattungen: dict[str, CategoryRequirement]

    @model_validator(mode="after")
    def _pruefe_gattungen(self) -> KnowledgePolicy:
        """Prüft, dass jede Instrumentengattung geregelt ist.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn eine Gattung fehlt oder unbekannt ist.
        """
        _pruefe_schluesselmenge(
            set(self.gattungen), {kategorie.name for kategorie in InstrumentCategory}, "gattungen"
        )
        return self


# --- Risikoklassen und Bandbreiten ----------------------------------------


class Band(_PolicyModel):
    """Zulässige Bandbreite eines Anteils.

    Attributes:
        min: Untergrenze des Anteils.
        max: Obergrenze des Anteils.
    """

    min: Share
    max: Share

    @model_validator(mode="after")
    def _pruefe_reihenfolge(self) -> Band:
        """Prüft, dass die Untergrenze nicht über der Obergrenze liegt.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn ``min`` größer als ``max`` ist.
        """
        if self.min > self.max:
            raise PolicyError(f"Bandbreite unzulässig: min {self.min} > max {self.max}.")
        return self


class RiskClassDefinition(_PolicyModel):
    """Definition einer Risikoklasse.

    Attributes:
        name: Bezeichnung der Klasse.
        beschreibung: Charakterisierung im Klartext für die Geeignetheitserklärung.
        positionsobergrenze: Höchstanteil einer einzelnen Position.
        erwarteter_maximalverlust: Zwischenzeitlicher Rückgang, mit dem der
            Kunde dieser Klasse rechnen muss. Quantitative Fassung der
            ``beschreibung``, gebunden an die Verlustszenario-Leiter aus
            ``tolerance_v1.yaml``. Eine Erwartung, keine Obergrenze.
        gruppen: Bandbreiten je Assetgruppe.
        einzelwerte: Bandbreiten je Assetklasse.
    """

    name: Annotated[str, Field(min_length=3)]
    beschreibung: Annotated[str, Field(min_length=10)]
    positionsobergrenze: Share
    erwarteter_maximalverlust: Share
    gruppen: dict[str, Band]
    einzelwerte: dict[str, Band]

    @model_validator(mode="after")
    def _pruefe_baender(self) -> RiskClassDefinition:
        """Prüft, dass die Bänder überhaupt ein zulässiges Portfolio zulassen.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn die Summen der Unter- oder Obergrenzen kein
                Portfolio mit Gewichtssumme eins zulassen (Invariante I5) oder
                eine Assetklasse fehlt.
        """
        _pruefe_schluesselmenge(set(self.einzelwerte), set(ASSET_KEYS), "einzelwerte")
        min_summe = sum((band.min for band in self.einzelwerte.values()), Decimal(0))
        max_summe = sum((band.max for band in self.einzelwerte.values()), Decimal(0))
        if min_summe > Decimal(1) + SUM_TOLERANCE:
            raise PolicyError(
                f"Klasse '{self.name}': Die Untergrenzen summieren sich auf {min_summe} "
                "und lassen keine Gewichtssumme von 1 zu."
            )
        if max_summe < Decimal(1) - SUM_TOLERANCE:
            raise PolicyError(
                f"Klasse '{self.name}': Die Obergrenzen summieren sich auf {max_summe} "
                "und lassen keine Gewichtssumme von 1 zu."
            )
        for schluessel, band in self.einzelwerte.items():
            if band.max > self.positionsobergrenze:
                raise PolicyError(
                    f"Klasse '{self.name}': Obergrenze für {schluessel} ({band.max}) "
                    f"überschreitet die Positionsobergrenze ({self.positionsobergrenze})."
                )
        return self


class ClassesPolicy(PolicyHeader):
    """Risikoklassen 1 bis 5 mit ihren Assetklassen-Bandbreiten.

    Attributes:
        gruppenzuordnung: Zuordnung der Assetklassen zu Gruppen.
        risikoklassen: Definition je Risikoklasse, Schlüssel ist die Klassenzahl.
    """

    gruppenzuordnung: dict[str, tuple[str, ...]]
    risikoklassen: dict[int, RiskClassDefinition]

    @model_validator(mode="after")
    def _pruefe_klassen(self) -> ClassesPolicy:
        """Prüft Vollständigkeit, Gruppenzuordnung und Monotonie der Aktienquote.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn Klassen fehlen, die Gruppenzuordnung unvollständig
                ist oder die Aktienquote nicht monoton steigt (Invariante I6).
        """
        erwartete_klassen = set(range(MIN_RISK_CLASS, MAX_RISK_CLASS + 1))
        if set(self.risikoklassen) != erwartete_klassen:
            raise PolicyError(
                f"Risikoklassen unvollständig: erwartet {sorted(erwartete_klassen)}, "
                f"vorhanden {sorted(self.risikoklassen)}."
            )
        zugeordnet = [
            schluessel for werte in self.gruppenzuordnung.values() for schluessel in werte
        ]
        _pruefe_schluesselmenge(set(zugeordnet), set(ASSET_KEYS), "gruppenzuordnung")
        if len(zugeordnet) != len(set(zugeordnet)):
            raise PolicyError("Eine Assetklasse ist mehreren Gruppen zugeordnet.")

        for klasse in sorted(self.risikoklassen):
            definition = self.risikoklassen[klasse]
            _pruefe_schluesselmenge(
                set(definition.gruppen), set(self.gruppenzuordnung), f"gruppen der Klasse {klasse}"
            )

        aktiengruppe = self.gruppenzuordnung.get("aktien")
        if aktiengruppe is None:
            raise PolicyError("Die Gruppe 'aktien' fehlt in der Gruppenzuordnung.")
        untergrenzen = [
            self.risikoklassen[k].gruppen["aktien"].min for k in sorted(self.risikoklassen)
        ]
        obergrenzen = [
            self.risikoklassen[k].gruppen["aktien"].max for k in sorted(self.risikoklassen)
        ]
        if untergrenzen != sorted(untergrenzen) or obergrenzen != sorted(obergrenzen):
            raise PolicyError(
                "Die Aktienquote muss über die Risikoklassen monoton steigen (Invariante I6)."
            )

        verluste = [
            self.risikoklassen[k].erwarteter_maximalverlust for k in sorted(self.risikoklassen)
        ]
        if verluste != sorted(verluste):
            raise PolicyError(
                "Der erwartete Maximalverlust muss über die Risikoklassen monoton steigen. "
                f"Vorhanden: {verluste}. Eine höhere Klasse, die weniger Verlust zusagt, "
                "wäre in sich widersprüchlich."
            )
        return self


# --- Glide-Path ------------------------------------------------------------


class GlidepathStep(_PolicyModel):
    """Eine Stufe des Risikoabbaus.

    Attributes:
        restlaufzeit_monate_ab: Untere Grenze der Restlaufzeit, einschließlich.
        max_risikoklasse: Höchste zulässige Risikoklasse ab dieser Restlaufzeit.
    """

    restlaufzeit_monate_ab: Annotated[int, Field(ge=0, le=1200)]
    max_risikoklasse: RiskClass


class GlidepathPolicy(PolicyHeader):
    """Regelbasierter Risikoabbau mit abnehmender Restlaufzeit.

    Attributes:
        stufen: Stufen, absteigend nach Restlaufzeit sortiert.
    """

    stufen: tuple[GlidepathStep, ...]

    @model_validator(mode="after")
    def _pruefe_stufen(self) -> GlidepathPolicy:
        """Prüft Sortierung, Monotonie und Abdeckung bis Restlaufzeit null.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn die Stufen unsortiert sind, die zulässige Klasse
                mit kürzerer Restlaufzeit steigt oder der Bereich nicht bis null
                reicht.
        """
        if not self.stufen:
            raise PolicyError("Der Glide-Path benötigt mindestens eine Stufe.")
        monate = [stufe.restlaufzeit_monate_ab for stufe in self.stufen]
        if monate != sorted(monate, reverse=True) or len(set(monate)) != len(monate):
            raise PolicyError(
                "Die Stufen müssen streng absteigend nach Restlaufzeit sortiert sein."
            )
        klassen = [stufe.max_risikoklasse for stufe in self.stufen]
        if klassen != sorted(klassen, reverse=True):
            raise PolicyError(
                "Mit kürzerer Restlaufzeit darf die zulässige Risikoklasse nicht steigen."
            )
        if monate[-1] != 0:
            raise PolicyError(
                "Die unterste Stufe muss bei Restlaufzeit 0 beginnen, sonst bliebe der "
                "Zeitraum kurz vor dem Zieltermin ungeregelt."
            )
        return self


# --- Zielvorgaben ----------------------------------------------------------


class GoalTemplate(_PolicyModel):
    """Vorbelegungen eines Zieltyps.

    Attributes:
        bezeichnung: Bezeichnung des Zieltyps für die Anzeige.
        beschreibung: Zweck des Ziels im Klartext.
        standardhorizont_monate: Vorbelegter Anlagehorizont in Monaten.
        standardprioritaet: Vorbelegte Verbindlichkeit.
        mindest_erfolgswahrscheinlichkeit: Vorbelegte Mindestwahrscheinlichkeit.
            ``None`` bedeutet, dass der Schwellenwert der Priorität gilt.
        termin_verschiebbar: Ob sich der Zieltermin verschieben lässt. Steuert
            die Machbarkeitsbewertung der Stellschraube S2 im Konfliktbericht.
        betrag_reduzierbar: Ob sich der Zielbetrag senken lässt. Steuert die
            Machbarkeitsbewertung der Stellschraube S3.
    """

    bezeichnung: Annotated[str, Field(min_length=3)]
    beschreibung: Annotated[str, Field(min_length=20)]
    standardhorizont_monate: Annotated[int, Field(ge=1, le=1200)]
    standardprioritaet: PriorityByName
    mindest_erfolgswahrscheinlichkeit: Share | None = None
    termin_verschiebbar: bool = True
    betrag_reduzierbar: bool = True


class GoalPolicy(PolicyHeader):
    """Schwellenwerte der Zielerreichung, Zielkatalog und Inflationsannahme.

    Attributes:
        mindest_erfolgswahrscheinlichkeit: Geforderte Mindestwahrscheinlichkeit
            je Priorität, Schlüssel ist der Name der Priorität.
        inflationsannahme_jaehrlich: Jährliche Inflationsrate für die Trennung
            von nominaler und realer Betrachtung.
        pufferband: Abstand zur Schwelle, unterhalb dessen eine Empfehlung mit
            ausdrücklichem Hinweis auf den geringen Abstand ergeht.
        zuteilungsreihenfolge: Reihenfolge der Prioritäten bei der Zuteilung
            der Sparrate auf mehrere Ziele.
        zieltypen: Vorbelegungen je Zieltyp.
    """

    mindest_erfolgswahrscheinlichkeit: dict[str, Share]
    inflationsannahme_jaehrlich: Annotated[Decimal, Field(ge=0, le=1)]
    pufferband: Annotated[Decimal, Field(ge=0, le=Decimal("0.5"))]
    zuteilungsreihenfolge: tuple[str, ...]
    zieltypen: dict[str, GoalTemplate]

    @model_validator(mode="after")
    def _pruefe_prioritaeten(self) -> GoalPolicy:
        """Prüft Vollständigkeit und Rangfolge der Schwellenwerte.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn eine Priorität fehlt oder ein MUSS-Ziel keine
                höhere Anforderung trägt als ein WUNSCH-Ziel.
        """
        _pruefe_schluesselmenge(
            set(self.mindest_erfolgswahrscheinlichkeit),
            {prioritaet.name for prioritaet in GoalPriority},
            "mindest_erfolgswahrscheinlichkeit",
        )
        muss = self.mindest_erfolgswahrscheinlichkeit[GoalPriority.MUSS.name]
        wunsch = self.mindest_erfolgswahrscheinlichkeit[GoalPriority.WUNSCH.name]
        if muss <= wunsch:
            raise PolicyError(
                f"Ein MUSS-Ziel muss eine höhere Mindestwahrscheinlichkeit tragen als ein "
                f"WUNSCH-Ziel; angegeben sind MUSS {muss} und WUNSCH {wunsch}."
            )
        _pruefe_schluesselmenge(
            set(self.zuteilungsreihenfolge),
            {prioritaet.name for prioritaet in GoalPriority},
            "zuteilungsreihenfolge",
        )
        if len(self.zuteilungsreihenfolge) != len(set(self.zuteilungsreihenfolge)):
            raise PolicyError("Die Zuteilungsreihenfolge enthält eine Priorität doppelt.")
        _pruefe_schluesselmenge(
            set(self.zieltypen), {zieltyp.name for zieltyp in GoalType}, "zieltypen"
        )
        for name, vorlage in self.zieltypen.items():
            if vorlage.mindest_erfolgswahrscheinlichkeit is None:
                continue
            schwelle = self.mindest_erfolgswahrscheinlichkeit[vorlage.standardprioritaet.name]
            if vorlage.mindest_erfolgswahrscheinlichkeit < schwelle:
                raise PolicyError(
                    f"Zieltyp {name!r}: Die Mindestwahrscheinlichkeit "
                    f"{vorlage.mindest_erfolgswahrscheinlichkeit} liegt unter dem "
                    f"Schwellenwert {schwelle} der Priorität "
                    f"{vorlage.standardprioritaet.name}. Ein Zieltyp darf die Anforderung "
                    "nur anheben, nicht senken."
                )
        return self

    def minimum_success_probability(
        self,
        priority: GoalPriority,
        requested: Decimal | None = None,
        goal_type: GoalType | None = None,
    ) -> Decimal:
        """Ermittelt die anzuwendende Mindest-Erfolgswahrscheinlichkeit.

        Nachschlagen in drei Stufen, jede darf die Anforderung nur **anheben**:

        1. Schwellenwert der Priorität,
        2. Vorbelegung des Zieltyps, sofern gesetzt,
        3. individuelle Vorgabe, sofern gesetzt.

        Die Einbahnrichtung ist wesentlich: Ließe sich der Schwellenwert senken,
        wäre die Konfliktdiagnose durch eine niedrigere Eigenvorgabe zu umgehen —
        genau die Beschönigung, die dieses Projekt vermeidet.

        Args:
            priority: Priorität des Ziels.
            requested: Abweichend gewünschte Mindestwahrscheinlichkeit.
            goal_type: Zieltyp, dessen Vorbelegung zu berücksichtigen ist.

        Returns:
            Der anzuwendende Schwellenwert.
        """
        schwelle = self.mindest_erfolgswahrscheinlichkeit[priority.name]
        if goal_type is not None:
            vorlage = self.zieltypen.get(goal_type.name)
            if vorlage is not None and vorlage.mindest_erfolgswahrscheinlichkeit is not None:
                schwelle = max(schwelle, vorlage.mindest_erfolgswahrscheinlichkeit)
        if requested is None:
            return schwelle
        return max(schwelle, requested)


# --- Simulation ------------------------------------------------------------


class AssetCost(_PolicyModel):
    """Kostenannahme für eine Assetklasse.

    Attributes:
        ter_jaehrlich: Laufende Kosten je Jahr, monatlich geometrisch abgezogen.
        spread_einmalig: Transaktionskosten je gehandeltem Euro, einseitig.
    """

    ter_jaehrlich: Annotated[Decimal, Field(ge=0, le=Decimal("0.1"))]
    spread_einmalig: Annotated[Decimal, Field(ge=0, le=Decimal("0.1"))]


class SimulationPolicy(PolicyHeader):
    """Parameter der Simulation und Kostenannahmen.

    Attributes:
        blocklaenge_monate: Länge eines Bootstrap-Blocks in Monaten.
        standard_pfade: Vorgegebene Zahl der Simulationspfade.
        basis_seed: Vorgegebener Seed (Prinzip P1).
        rebalancing_monate: Abstand zweier Rebalancings in Monaten.
        kosten: Kostenannahme je Assetklasse.
    """

    blocklaenge_monate: Annotated[int, Field(ge=1, le=120)]
    standard_pfade: Annotated[int, Field(ge=1, le=1_000_000)]
    basis_seed: Annotated[int, Field(ge=0)]
    rebalancing_monate: Annotated[int, Field(ge=1, le=600)]
    kosten: dict[str, AssetCost]

    @model_validator(mode="after")
    def _pruefe_kosten(self) -> SimulationPolicy:
        """Prüft, dass jede Assetklasse eine Kostenannahme trägt.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn eine Assetklasse fehlt oder unbekannt ist.
        """
        _pruefe_schluesselmenge(set(self.kosten), set(ASSET_KEYS), "kosten")
        return self


# --- Portfoliokonstruktion -------------------------------------------------


class MeanMethod(Enum):
    """Verfahren zur Schätzung der erwarteten Renditen.

    Attributes:
        HISTORISCH: Arithmetisches Mittel der beobachteten Monatsrenditen.
        SHRINKAGE: Zum Gesamtmittel geschrumpfter Schätzer nach Jorion (1986).
    """

    HISTORISCH = "historisch"
    SHRINKAGE = "shrinkage"


class CovarianceMethod(Enum):
    """Verfahren zur Schätzung der Kovarianzmatrix.

    Attributes:
        LEDOIT_WOLF: Schrumpfung nach Ledoit & Wolf (2004).
        STICHPROBE: Unbearbeitete Stichprobenkovarianz.
    """

    LEDOIT_WOLF = "ledoit_wolf"
    STICHPROBE = "stichprobe"


class ShrinkageTarget(Enum):
    """Ziel der Schrumpfung der Erwartungswerte.

    Attributes:
        GESAMTMITTEL: Gleichgewichtetes Mittel über alle Assetklassen.
    """

    GESAMTMITTEL = "gesamtmittel"


class TargetReturnSource(Enum):
    """Herkunft der Zielrendite in der Nebenbedingung ``w'µ >= r_ziel``.

    Attributes:
        REFERENZALLOKATION: Erwartete Rendite der regelbasierten
            Referenzallokation derselben Risikoklasse.
    """

    REFERENZALLOKATION = "referenzallokation"


class EstimationSettings(_PolicyModel):
    """Parameter der Momentenschätzung.

    Attributes:
        erwartungswert_verfahren: Gewähltes Verfahren für die erwarteten Renditen.
        erwartungswert_ziel: Ziel der Schrumpfung.
        erwartungswert_intensitaet: Feste Schrumpfungsintensität; ``None``
            bestimmt sie analytisch aus den Daten.
        kovarianz_verfahren: Gewählter Kovarianzschätzer.
        risikofreier_proxy: Assetklasse als Näherung des risikofreien Zinses;
            ``None`` rechnet mit null Prozent.
    """

    erwartungswert_verfahren: MeanMethod
    erwartungswert_ziel: ShrinkageTarget
    erwartungswert_intensitaet: Annotated[Decimal, Field(ge=0, le=1)] | None
    kovarianz_verfahren: CovarianceMethod
    risikofreier_proxy: str | None

    @model_validator(mode="after")
    def _pruefe_proxy(self) -> EstimationSettings:
        """Prüft, dass der Proxy eine bekannte Assetklasse ist.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn der Proxy nicht zum Anlageuniversum gehört.
        """
        if self.risikofreier_proxy is not None and self.risikofreier_proxy not in ASSET_KEYS:
            raise PolicyError(
                f"Der risikofreie Proxy {self.risikofreier_proxy!r} gehört nicht zum "
                f"Anlageuniversum. Bekannt sind: {sorted(ASSET_KEYS)}."
            )
        return self


class OptimizationSettings(_PolicyModel):
    """Parameter der restringierten Mean-Variance-Optimierung.

    Attributes:
        solver_reihenfolge: Solver in der Reihenfolge ihrer Verwendung.
        fallback: Rückfallverfahren, wenn kein Solver ein Optimum meldet.
        restriktionstoleranz: Toleranz beim Prüfen des Ergebnisses.
        zielrendite_quelle: Herkunft der Zielrendite.
        zielrendite_absenkung_schritt: Schrittweite der Absenkung bei
            unerfüllbarer Zielrendite.
        zielrendite_absenkung_max_schritte: Höchstzahl der Absenkungen.
    """

    solver_reihenfolge: Annotated[tuple[str, ...], Field(min_length=1)]
    fallback: str
    restriktionstoleranz: Annotated[float, Field(gt=0, le=0.01)]
    zielrendite_quelle: TargetReturnSource
    zielrendite_absenkung_schritt: Annotated[float, Field(gt=0, le=0.01)]
    zielrendite_absenkung_max_schritte: Annotated[int, Field(ge=1, le=500)]


class BacktestSettings(_PolicyModel):
    """Vorgaben des rollierenden Out-of-Sample-Backtests.

    Attributes:
        schaetzfenster_monate: Länge des rollierenden Schätzfensters.
        haltedauer_monate: Haltedauer bis zum nächsten Rebalancing.
        monate_pro_jahr: Zahl der Monate je Jahr für die Annualisierung.
        risikoklasse: Risikoklasse, deren Restriktionen im Backtest gelten.
    """

    schaetzfenster_monate: Annotated[int, Field(ge=12, le=600)]
    haltedauer_monate: Annotated[int, Field(ge=1, le=120)]
    monate_pro_jahr: Annotated[int, Field(ge=1, le=12)]
    risikoklasse: RiskClass

    @model_validator(mode="after")
    def _pruefe_fenster(self) -> BacktestSettings:
        """Prüft, dass das Schätzfenster eine eindeutige Lösung zulässt.

        Bei einem Fenster, das nicht länger ist als die Zahl der Assetklassen,
        ist die Stichprobenkovarianz singulär. Die Optimierung hätte dann keine
        eindeutige Lösung — ein Fehler, der sich sonst erst als merkwürdiges
        Ergebnis zeigen würde.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn das Fenster zu kurz ist.
        """
        if self.schaetzfenster_monate <= len(ASSET_KEYS):
            raise PolicyError(
                f"Das Schätzfenster von {self.schaetzfenster_monate} Monaten ist nicht "
                f"länger als die Zahl der Assetklassen ({len(ASSET_KEYS)}). Die "
                "Stichprobenkovarianz wäre singulär."
            )
        return self


class SustainabilityRating(_PolicyModel):
    """Einstufung einer Assetklasse nach den drei Ausprägungen der Verordnung.

    **Annahme, keine erhobene Größe.** Siehe den Kopfkommentar von
    ``policies/portfolio_v1.yaml`` und Recherchelücke L-06.

    Attributes:
        taxonomie: Ökologisch nachhaltig im Sinne der VO (EU) 2020/852.
        sfdr: Nachhaltige Investition nach Art. 2 Nr. 17 VO (EU) 2019/2088.
        pai: Berücksichtigt die wichtigsten nachteiligen Auswirkungen.
        hinweis: Begründung der Einstufung im Klartext.
    """

    taxonomie: bool
    sfdr: bool
    pai: bool
    hinweis: Annotated[str, Field(min_length=10)]


class SustainabilitySettings(_PolicyModel):
    """Parameter des Nachhaltigkeitsfilters.

    Attributes:
        mindestuniversum: Untergrenze des gefilterten Universums (Invariante I8).
        einstufung: Einstufung je Assetklasse.
    """

    mindestuniversum: Annotated[int, Field(ge=2, le=len(ASSET_KEYS))]
    einstufung: dict[str, SustainabilityRating]

    @model_validator(mode="after")
    def _pruefe_einstufung(self) -> SustainabilitySettings:
        """Prüft, dass jede Assetklasse eingestuft ist.

        Returns:
            Das unveränderte Modell.

        Raises:
            PolicyError: Wenn eine Assetklasse fehlt oder unbekannt ist.
        """
        _pruefe_schluesselmenge(set(self.einstufung), set(ASSET_KEYS), "einstufung")
        return self


class PortfolioPolicy(PolicyHeader):
    """Verfahrensparameter der Portfoliokonstruktion.

    Attributes:
        schaetzung: Parameter der Momentenschätzung.
        optimierung: Parameter der Optimierung.
        backtest: Vorgaben des Out-of-Sample-Backtests.
        nachhaltigkeit: Parameter des Nachhaltigkeitsfilters.
    """

    schaetzung: EstimationSettings
    optimierung: OptimizationSettings
    backtest: BacktestSettings
    nachhaltigkeit: SustainabilitySettings


# --- Zusammenfassung -------------------------------------------------------


class PolicySet(_PolicyModel):
    """Das vollständige, geladene Regelwerk.

    Attributes:
        capacity: Punktetabelle der Risikotragfähigkeit.
        tolerance: Punktetabelle der Risikobereitschaft.
        knowledge: Komplexitätsdeckel aus Kenntnissen und Erfahrungen.
        classes: Risikoklassen und Bandbreiten.
        glidepath: Risikoabbau über die Restlaufzeit.
        goals: Schwellenwerte der Zielerreichung.
        simulation: Parameter der Simulation und Kostenannahmen.
        portfolio: Verfahrensparameter der Portfoliokonstruktion.
    """

    capacity: CapacityPolicy
    tolerance: TolerancePolicy
    knowledge: KnowledgePolicy
    classes: ClassesPolicy
    glidepath: GlidepathPolicy
    goals: GoalPolicy
    simulation: SimulationPolicy
    portfolio: PortfolioPolicy

    @property
    def versions(self) -> dict[str, str]:
        """Fassungen aller Teilpolicies.

        Diese Zuordnung wandert in jeden Trace und in jede Geeignetheitserklärung.
        Ohne sie ließe sich ein Ergebnis nicht dem Regelstand zuordnen, unter dem
        es entstanden ist (Prinzip P3).

        Returns:
            Zuordnung Policy-Name auf Versionskennung.
        """
        return {
            "capacity": self.capacity.version,
            "tolerance": self.tolerance.version,
            "knowledge": self.knowledge.version,
            "classes": self.classes.version,
            "glidepath": self.glidepath.version,
            "goals": self.goals.version,
            "simulation": self.simulation.version,
            "portfolio": self.portfolio.version,
        }

    @property
    def combined_version(self) -> str:
        """Verdichtete Versionskennung des gesamten Regelwerks.

        Returns:
            Zeichenkette der Form ``capacity=1.0.0;classes=1.0.0;…``.
        """
        return ";".join(f"{name}={version}" for name, version in sorted(self.versions.items()))


def _pruefe_schluesselmenge(vorhanden: set[str], erwartet: set[str], kontext: str) -> None:
    """Prüft, dass eine Schlüsselmenge genau der erwarteten entspricht.

    Args:
        vorhanden: Tatsächlich vorhandene Schlüssel.
        erwartet: Erwartete Schlüssel.
        kontext: Bezeichnung der geprüften Stelle für die Fehlermeldung.

    Raises:
        PolicyError: Bei fehlenden oder unbekannten Schlüsseln.
    """
    fehlend = erwartet - vorhanden
    unbekannt = vorhanden - erwartet
    if fehlend or unbekannt:
        teile = []
        if fehlend:
            teile.append(f"fehlend: {sorted(fehlend)}")
        if unbekannt:
            teile.append(f"unbekannt: {sorted(unbekannt)}")
        raise PolicyError(f"Schlüssel in '{kontext}' stimmen nicht — {'; '.join(teile)}.")
