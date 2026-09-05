"""Kundenprofil: Haushalt, Kenntnisse und Erfahrungen, Nachhaltigkeitspräferenzen.

Das Modul enthält **ausschließlich Modelle und Validierung**, keine Bewertung.
Ob eine Liquiditätsreserve von drei Monatsausgaben viele oder wenige Punkte
bringt, steht in ``policies/capacity_v1.yaml`` und wird in
:mod:`candor.profiling.capacity` ausgewertet (Prinzip P3). Solche Schwellenwerte sind
Festlegungen des Anbieters und keine Rechenvorschriften: In einer versionierten
YAML-Datei sind sie ohne Blick in den Code prüfbar, und ihre Fassung wandert in jeden
Trace, sodass ein Ergebnis auch später dem Regelstand zuordenbar bleibt, unter dem es
entstanden ist.

Drei Erhebungsblöcke, jeder mit eigener Rechtsgrundlage:

* :class:`Household` — wirtschaftliche Verhältnisse und Verlusttragfähigkeit,
  Art. 54 Abs. 2 Buchst. b und Abs. 4 DelVO (EU) 2017/565. Die Gliederung ist
  an die Struktur der DIN 77230 angelehnt; die Norm wurde nicht eingesehen und
  wird deshalb nicht im Wortlaut wiedergegeben.
* :class:`KnowledgeAndExperience` — Art. 55 DelVO (EU) 2017/565. Erhoben werden
  je Instrumentengattung Kenntnisstand, Erfahrungsdauer und
  Transaktionshäufigkeit sowie Bildungsstand und Beruf.
* :class:`SustainabilityPreferences` — Art. 2 Nr. 7 DelVO (EU) 2017/565 in der
  Fassung der DelVO (EU) 2021/1253, anwendbar seit 02.08.2022.

Geldbeträge sind durchgängig ``Decimal``. Für Beträge, die in eine
Geeignetheitserklärung wandern, ist binäre Gleitkommaarithmetik ungeeignet:
``0.1 + 0.2`` ist dort nicht ``0.3``, und ein Beratungsprotokoll, dessen Zahlen
sich nicht exakt reproduzieren lassen, verfehlt seinen Zweck. Die Umwandlung in
``float`` geschieht erst an der Grenze zur Simulation.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from candor.domain.errors import IncompleteProfileError, InconsistentProfileError

#: Rechtsgrundlage je Erhebungsblock. Wandert in Fehlermeldungen und Traces.
LEGAL_BASIS: Final[dict[str, str]] = {
    "household": "Art. 54 Abs. 2 Buchst. b und Abs. 4 DelVO (EU) 2017/565",
    "knowledge": "Art. 55 DelVO (EU) 2017/565",
    "sustainability": "Art. 2 Nr. 7 DelVO (EU) 2017/565 i. d. F. der DelVO (EU) 2021/1253",
    "identity": "Art. 54 Abs. 2 DelVO (EU) 2017/565",
    "risk_attitude": "Art. 54 Abs. 2 Buchst. a und Abs. 5 DelVO (EU) 2017/565",
}

#: Geldbetrag in Euro, nicht negativ, auf Cent genau.
Money = Annotated[Decimal, Field(ge=0, decimal_places=2)]

#: Anteil zwischen 0 und 1.
Share = Annotated[Decimal, Field(ge=0, le=1)]


class EmploymentType(Enum):
    """Beschäftigungsart als Indikator der Einkommensstabilität.

    Die Zuordnung zu Punktwerten erfolgt nicht hier, sondern in
    ``policies/capacity_v1.yaml``.
    """

    UNBEFRISTET = "unbefristet angestellt"
    VERBEAMTET = "verbeamtet"
    BEFRISTET = "befristet angestellt"
    SELBSTSTAENDIG = "selbstständig"
    IN_AUSBILDUNG = "in Ausbildung oder Studium"
    ARBEITSSUCHEND = "arbeitssuchend"
    RUHESTAND = "im Ruhestand"


class EducationLevel(Enum):
    """Bildungsstand nach Art. 55 Abs. 1 Buchst. c DelVO (EU) 2017/565."""

    OHNE_ABSCHLUSS = "ohne Abschluss"
    SCHULABSCHLUSS = "Schulabschluss"
    BERUFSAUSBILDUNG = "abgeschlossene Berufsausbildung"
    HOCHSCHULABSCHLUSS = "Hochschulabschluss"
    HOCHSCHULABSCHLUSS_WIRTSCHAFT = "Hochschulabschluss mit wirtschaftlichem Schwerpunkt"


class InstrumentCategory(Enum):
    """Instrumentengattungen, für die Kenntnisse gesondert zu erheben sind.

    Die Gattungen sind nach steigender Komplexität geordnet. Welche Gattung
    welche Risikoklasse noch zulässt, regelt ``policies/knowledge_v1.yaml``.
    """

    GELDMARKT = "Geldmarkt und Termineinlagen"
    ANLEIHEN_IG = "Anleihen guter Bonität"
    ANLEIHEN_HY = "Anleihen schwacher Bonität"
    FONDS_UCITS = "Investmentfonds (OGAW)"
    ETF = "börsengehandelte Indexfonds (ETF)"
    AKTIEN = "Aktien"
    ZERTIFIKATE = "Zertifikate und strukturierte Produkte"
    DERIVATE = "Derivate"


class KnowledgeLevel(Enum):
    """Selbsteingeschätzter Kenntnisstand je Instrumentengattung.

    Die Stufen sind bewusst nur benannt und nicht mit Zahlen hinterlegt. Ihre
    Gewichtung steht im Regelwerk.
    """

    KEINE = "keine Kenntnisse"
    GRUNDKENNTNISSE = "Grundkenntnisse"
    ERWEITERT = "erweiterte Kenntnisse"
    UMFASSEND = "umfassende Kenntnisse"


class _StrictModel(BaseModel):
    """Basismodell mit strenger Konfiguration.

    ``extra="forbid"`` weist unbekannte Felder ab, statt sie stillschweigend zu
    verwerfen: Ein Tippfehler im Feldnamen eines Kundenprofils darf nicht dazu
    führen, dass die betreffende Angabe unbemerkt fehlt (Prinzip P5).
    ``frozen=True`` macht ein einmal geprüftes Profil unveränderlich, sodass
    zwischen Prüfung und Verwendung nichts mehr verändert werden kann.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


class Household(_StrictModel):
    """Wirtschaftliche Verhältnisse des Haushalts.

    Rechtsgrundlage: Art. 54 Abs. 2 Buchst. b und Abs. 4 DelVO (EU) 2017/565 —
    die Firma hat Informationen über die finanziellen Verhältnisse einschließlich
    der Fähigkeit einzuholen, Verluste zu tragen.

    Attributes:
        net_monthly_income: Monatliches Nettoeinkommen des Haushalts in Euro.
        fixed_monthly_costs: Feste monatliche Ausgaben ohne Schuldendienst.
        monthly_debt_service: Monatlicher Kapitaldienst für bestehende Kredite.
        free_savings_amount: Monatlich frei verfügbarer Betrag (freie Sparquote).
        liquidity_reserve_months: Vorhandene Liquiditätsreserve, gemessen in
            Monatsausgaben. Die Einheit ist bewusst nicht Euro: Eine Reserve von
            10 000 Euro bedeutet für zwei Haushalte mit unterschiedlichen
            Ausgaben Verschiedenes.
        total_debt: Summe der bestehenden Verbindlichkeiten.
        other_assets: Sonstiges Vermögen außerhalb der geplanten Anlage.
        employment_type: Beschäftigungsart als Indikator der Einkommensstabilität.
    """

    net_monthly_income: Money
    fixed_monthly_costs: Money
    monthly_debt_service: Money
    free_savings_amount: Money
    liquidity_reserve_months: Annotated[Decimal, Field(ge=0, le=120, decimal_places=1)]
    total_debt: Money
    other_assets: Money
    employment_type: EmploymentType

    @model_validator(mode="after")
    def _pruefe_haushaltsrechnung(self) -> Household:
        """Prüft die Haushaltsrechnung auf innere Widersprüche.

        Das ist Validierung, keine Bewertung: Geprüft wird nur, ob die Angaben
        nebeneinander bestehen können, nicht wie gut sie sind.

        Returns:
            Das unveränderte Modell.

        Raises:
            InconsistentProfileError: Wenn die freie Sparquote den tatsächlich
                verfügbaren Überschuss übersteigt oder der Schuldendienst allein
                das Einkommen überschreitet.
        """
        ueberschuss = self.net_monthly_income - self.fixed_monthly_costs - self.monthly_debt_service
        if self.free_savings_amount > ueberschuss:
            raise InconsistentProfileError(
                (
                    "household.free_savings_amount",
                    "household.net_monthly_income",
                    "household.fixed_monthly_costs",
                    "household.monthly_debt_service",
                ),
                f"Die freie Sparquote von {self.free_savings_amount} € übersteigt den "
                f"verfügbaren Überschuss von {ueberschuss} € "
                "(Nettoeinkommen abzüglich Fixkosten und Schuldendienst).",
            )
        if self.monthly_debt_service > self.net_monthly_income:
            raise InconsistentProfileError(
                ("household.monthly_debt_service", "household.net_monthly_income"),
                f"Der Schuldendienst von {self.monthly_debt_service} € übersteigt das "
                f"Nettoeinkommen von {self.net_monthly_income} €.",
            )
        return self


class InstrumentExperience(_StrictModel):
    """Kenntnisse und Erfahrungen zu einer Instrumentengattung.

    Rechtsgrundlage: Art. 55 Abs. 1 Buchst. a und b DelVO (EU) 2017/565 — Art
    der vertrauten Instrumente sowie Art, Umfang, Häufigkeit und Zeitraum der
    Geschäfte.

    Attributes:
        category: Betroffene Instrumentengattung.
        knowledge_level: Selbsteingeschätzter Kenntnisstand.
        experience_years: Dauer der Erfahrung in Jahren.
        transactions_per_year: Zahl der Geschäfte je Jahr in dieser Gattung.
    """

    category: InstrumentCategory
    knowledge_level: KnowledgeLevel
    experience_years: Annotated[Decimal, Field(ge=0, le=80, decimal_places=1)]
    transactions_per_year: Annotated[int, Field(ge=0, le=10_000)]

    @model_validator(mode="after")
    def _pruefe_erfahrungsangaben(self) -> InstrumentExperience:
        """Weist Erfahrungsangaben ohne jede Kenntnis zurück.

        Returns:
            Das unveränderte Modell.

        Raises:
            InconsistentProfileError: Wenn ohne Kenntnisse Geschäfte oder
                Erfahrungsjahre angegeben sind.
        """
        ohne_kenntnisse = self.knowledge_level is KnowledgeLevel.KEINE
        mit_erfahrung = self.experience_years > 0 or self.transactions_per_year > 0
        if ohne_kenntnisse and mit_erfahrung:
            raise InconsistentProfileError(
                (
                    f"knowledge.{self.category.name}.knowledge_level",
                    f"knowledge.{self.category.name}.experience_years",
                ),
                "Es sind Erfahrungen angegeben, zugleich aber 'keine Kenntnisse'. "
                "Beides zusammen ist nicht auswertbar.",
            )
        return self


class KnowledgeAndExperience(_StrictModel):
    """Kenntnisse und Erfahrungen des Kunden.

    Rechtsgrundlage: Art. 55 DelVO (EU) 2017/565.

    Attributes:
        education_level: Bildungsstand (Art. 55 Abs. 1 Buchst. c).
        profession_finance_related: Ob der aktuelle oder frühere Beruf einen
            Bezug zu Finanzdienstleistungen hat (Art. 55 Abs. 1 Buchst. c).
        entries: Angaben je Instrumentengattung. Gattungen ohne Eintrag gelten
            als „keine Kenntnisse“ — das ist keine Annahme, sondern die
            konservative Auslegung: Was nicht erhoben wurde, darf nicht zur
            Freigabe komplexerer Produkte führen.
    """

    education_level: EducationLevel
    profession_finance_related: bool
    entries: tuple[InstrumentExperience, ...]

    @model_validator(mode="after")
    def _pruefe_gattungen(self) -> KnowledgeAndExperience:
        """Stellt sicher, dass Angaben vorliegen und keine Gattung doppelt vorkommt.

        Returns:
            Das unveränderte Modell.

        Raises:
            IncompleteProfileError: Wenn überhaupt keine Angaben vorliegen.
            InconsistentProfileError: Wenn eine Gattung mehrfach auftritt.
        """
        if not self.entries:
            raise IncompleteProfileError(
                "knowledge.entries",
                "Zu keiner Instrumentengattung liegen Angaben vor. Ohne Kenntnisse und "
                "Erfahrungen darf keine Geeignetheitsbeurteilung erfolgen.",
                LEGAL_BASIS["knowledge"],
            )
        gattungen = [eintrag.category for eintrag in self.entries]
        if len(set(gattungen)) != len(gattungen):
            doppelt = sorted({g.name for g in gattungen if gattungen.count(g) > 1})
            raise InconsistentProfileError(
                ("knowledge.entries",),
                f"Mehrfache Angaben zu derselben Gattung: {', '.join(doppelt)}.",
            )
        return self

    def for_category(self, category: InstrumentCategory) -> InstrumentExperience | None:
        """Liefert die Angaben zu einer Gattung.

        Args:
            category: Gesuchte Instrumentengattung.

        Returns:
            Der Eintrag oder ``None``, wenn zu dieser Gattung nichts erhoben
            wurde. ``None`` ist hier zulässig und bedeutet ausdrücklich
            „nicht erhoben“; die konservative Behandlung erfolgt in
            :mod:`candor.profiling.knowledge`.
        """
        for eintrag in self.entries:
            if eintrag.category is category:
                return eintrag
        return None


class SustainabilityComponent(_StrictModel):
    """Eine der drei Ausprägungen einer Nachhaltigkeitspräferenz.

    Attributes:
        requested: Ob der Kunde diese Ausprägung verlangt.
        minimum_share: Verlangter Mindestanteil am Portfolio. Nur zulässig,
            wenn ``requested`` gesetzt ist.
    """

    requested: bool = False
    minimum_share: Share | None = None

    @model_validator(mode="after")
    def _pruefe_mindestanteil(self) -> SustainabilityComponent:
        """Prüft, dass Mindestanteil und Verlangen zusammenpassen.

        Returns:
            Das unveränderte Modell.

        Raises:
            InconsistentProfileError: Wenn ein Mindestanteil ohne Verlangen
                gesetzt ist oder ein Verlangen ohne Mindestanteil bleibt.
        """
        if self.minimum_share is not None and not self.requested:
            raise InconsistentProfileError(
                ("sustainability.minimum_share", "sustainability.requested"),
                "Es ist ein Mindestanteil angegeben, die Ausprägung wird aber nicht verlangt.",
            )
        if self.requested and self.minimum_share is None:
            raise InconsistentProfileError(
                ("sustainability.requested", "sustainability.minimum_share"),
                "Die Ausprägung wird verlangt, ein Mindestanteil fehlt jedoch. Ohne "
                "bezifferten Mindestanteil lässt sich die Präferenz nicht gegen das "
                "empfohlene Portfolio prüfen.",
            )
        return self


class SustainabilityPreferences(_StrictModel):
    """Nachhaltigkeitspräferenzen des Kunden.

    Rechtsgrundlage: Art. 2 Nr. 7 DelVO (EU) 2017/565 in der Fassung der
    DelVO (EU) 2021/1253, anwendbar seit 02.08.2022. Die Vorschrift kennt drei
    Ausprägungen, die einzeln oder nebeneinander verlangt werden können.

    Ausdrücklich vorgesehen ist auch, **keine** Nachhaltigkeitspräferenzen zu
    haben. Dieser Fall wird eigens abgebildet und nicht als leere Angabe
    behandelt, weil er zu dokumentieren ist: Der Unterschied zwischen „nicht
    gefragt" und „gefragt und verneint“ ist aufsichtsrechtlich erheblich.

    Attributes:
        has_preferences: Ob der Kunde überhaupt Nachhaltigkeitspräferenzen hat.
        taxonomy_aligned: Mindestanteil ökologisch nachhaltiger Investitionen
            im Sinne der Taxonomie-Verordnung (EU) 2020/852.
        sfdr_sustainable: Mindestanteil nachhaltiger Investitionen im Sinne des
            Art. 2 Nr. 17 der Offenlegungsverordnung (EU) 2019/2088.
        pai_considered: Mindestanteil an Instrumenten, die die wichtigsten
            nachteiligen Auswirkungen auf Nachhaltigkeitsfaktoren
            berücksichtigen. Die Verordnung schreibt für diese Ausprägung keinen
            quantitativen Mindestanteil vor, sondern qualitative Elemente; der
            Mindestanteil ist hier die operationalisierte Fassung und im
            Bericht als solche zu kennzeichnen.
    """

    has_preferences: bool
    taxonomy_aligned: SustainabilityComponent = SustainabilityComponent()
    sfdr_sustainable: SustainabilityComponent = SustainabilityComponent()
    pai_considered: SustainabilityComponent = SustainabilityComponent()

    @model_validator(mode="after")
    def _pruefe_stimmigkeit(self) -> SustainabilityPreferences:
        """Prüft, dass die Gesamtangabe zu den Einzelausprägungen passt.

        Returns:
            Das unveränderte Modell.

        Raises:
            InconsistentProfileError: Wenn Gesamtangabe und Ausprägungen sich
                widersprechen.
        """
        aktive = [
            name
            for name, komponente in (
                ("taxonomy_aligned", self.taxonomy_aligned),
                ("sfdr_sustainable", self.sfdr_sustainable),
                ("pai_considered", self.pai_considered),
            )
            if komponente.requested
        ]
        if self.has_preferences and not aktive:
            raise InconsistentProfileError(
                ("sustainability.has_preferences",),
                "Es sind Nachhaltigkeitspräferenzen angegeben, aber keine der drei "
                "Ausprägungen ist verlangt.",
            )
        if not self.has_preferences and aktive:
            raise InconsistentProfileError(
                ("sustainability.has_preferences",),
                f"Es sind keine Nachhaltigkeitspräferenzen angegeben, zugleich sind "
                f"{', '.join(aktive)} verlangt.",
            )
        return self


class RiskAttitude(_StrictModel):
    """Antworten des Kunden auf die Items zur Risikobereitschaft.

    Rechtsgrundlage: Art. 54 Abs. 2 Buchst. a und Abs. 5 DelVO (EU) 2017/565 —
    die Risikotoleranz ist Teil der Anlageziele und gesondert zu erheben.

    Die Antworten sind bewusst als offene Zuordnung geführt und nicht als feste
    Felder: Welche Items es gibt und welche Antwortoptionen zulässig sind, steht
    im Regelwerk (``policies/tolerance_v1.yaml``). Ein neues Item darf keine
    Codeänderung erzwingen (Prinzip P3). Die Prüfung gegen das Regelwerk erfolgt
    in :mod:`candor.profiling.tolerance`; der Domänenkern kennt das Regelwerk
    nicht (Prinzip P4).

    Attributes:
        answers: Zuordnung Item-Kennung auf gewählte Antwortoption.
    """

    answers: dict[str, str]

    @model_validator(mode="after")
    def _pruefe_antworten(self) -> RiskAttitude:
        """Prüft, dass überhaupt Antworten vorliegen und keine leer ist.

        Returns:
            Das unveränderte Modell.

        Raises:
            IncompleteProfileError: Wenn keine Antworten vorliegen.
            InconsistentProfileError: Wenn eine Kennung oder Antwort leer ist.
        """
        if not self.answers:
            raise IncompleteProfileError(
                "risk_attitude.answers",
                "Es liegen keine Antworten zur Risikobereitschaft vor. Ohne sie darf "
                "keine Geeignetheitsbeurteilung erfolgen.",
                LEGAL_BASIS["risk_attitude"],
            )
        for item, antwort in self.answers.items():
            if not item.strip() or not antwort.strip():
                raise InconsistentProfileError(
                    ("risk_attitude.answers",),
                    f"Leere Item-Kennung oder Antwort: {item!r} -> {antwort!r}.",
                )
        return self


class Client(_StrictModel):
    """Vollständiges Kundenprofil als Eingang der Geeignetheitsprüfung.

    Attributes:
        client_id: Fachliche Kennung des Kunden. Wandert in jeden Trace.
        household: Wirtschaftliche Verhältnisse.
        knowledge: Kenntnisse und Erfahrungen.
        sustainability: Nachhaltigkeitspräferenzen.
        risk_attitude: Antworten zur Risikobereitschaft.
    """

    client_id: Annotated[str, Field(min_length=1, max_length=64)]
    household: Household
    knowledge: KnowledgeAndExperience
    sustainability: SustainabilityPreferences
    risk_attitude: RiskAttitude


def parse_client(data: dict[str, object]) -> Client:
    """Erzeugt ein Kundenprofil aus einer Zuordnung und meldet Mängel fachlich.

    pydantic meldet fehlende Felder technisch korrekt, aber ohne Rechtsbezug.
    Diese Funktion übersetzt den ersten gefundenen Mangel in einen
    :class:`~candor.domain.errors.IncompleteProfileError`, der Feldpfad und
    Rechtsgrundlage nennt.

    Args:
        data: Rohangaben des Kundenprofils.

    Returns:
        Das geprüfte, unveränderliche Kundenprofil.

    Raises:
        IncompleteProfileError: Wenn eine Pflichtangabe fehlt oder unzulässig ist.
        InconsistentProfileError: Wenn Angaben sich widersprechen. Dieser Fehler
            entsteht bereits in den Modellvalidierungen und wird durchgereicht.
    """
    try:
        return Client.model_validate(data)
    except ValidationError as fehler:
        erster = fehler.errors()[0]
        pfad = ".".join(str(teil) for teil in erster["loc"])
        block = pfad.split(".")[0] if pfad else "identity"
        raise IncompleteProfileError(
            pfad or "client",
            erster.get("msg", "unzulässiger Wert"),
            LEGAL_BASIS.get(block, LEGAL_BASIS["identity"]),
        ) from fehler
