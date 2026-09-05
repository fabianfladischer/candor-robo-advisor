"""Geeignetheitserklärung nach Art. 54 Abs. 12 DelVO 2017/565, § 64 Abs. 4 WpHG.

Was das Dokument leisten muss
-----------------------------
Die Vorschrift verlangt, dass dem Kunden dargelegt wird, **weshalb** die
Empfehlung zu ihm passt — nicht bloß, dass sie das tut. Das Dokument muss die
erhobenen Angaben, die daraus abgeleitete Einstufung und die Verbindung zwischen
beiden erkennen lassen.

Wie es hier erzeugt wird
------------------------
Vollständig aus dem Entscheidungspfad. Jeder inhaltliche Absatz stammt aus einem
Textbaustein in :mod:`candor.report.textblocks`, der an die Kennung genau der
Regel gebunden ist, die den Schritt aufgezeichnet hat. Die Werte in den
Bausteinen kommen aus den Trace-Ereignissen und den Ergebnisobjekten, nie aus
freiem Text.

Die Folge ist eine harte Zusicherung: **Was im Dokument steht, steht im Trace.**
Ein Absatz ohne zugehörige Regel lässt sich nicht erzeugen, und ein Baustein mit
fehlendem Wert bricht ab, statt eine Lücke zu überdecken (Prinzip P5).

Aufbau
------
1. Zweck und Rechtsgrundlage
2. Erfasstes Kundenprofil — alle drei Achsen mit Punkteständen
3. Ermittelte Risikoklasse mit Angabe der bindenden Achse
4. Anlageziel und erforderliche Rendite
5. Simulierte Zielerreichung
6. Empfohlenes Portfolio mit Begründung je Position
7. Nachhaltigkeitspräferenzen
8. Risikohinweise und Kostenausweis
9. Nachweis: Regelstand, Datenstand, Zeitstempel, Hash

Aufruf:
    python -m candor.report.suitability --help
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path
from typing import Final

import numpy as np

from candor import format_de
from candor.data.universe_meta import get_instrument
from candor.domain.result import Axis
from candor.engine.advise import DOCUMENTATION_LEGAL_BASIS, Advice
from candor.report.document import Block, BlockKind, Document, render_pdf
from candor.report.textblocks import render, static

#: Titel des Dokuments.
TITLE: Final = "Geeignetheitserklärung"

#: Monate je Jahr.
MONTHS_PER_YEAR: Final = 12

#: Untergrenze, ab der eine Position im Portfolio ausgewiesen wird. Positionen
#: darunter sind wirtschaftlich bedeutungslos und würden die Tabelle aufblähen.
MIN_WEIGHT: Final = 0.0005


class SuitabilityError(ValueError):
    """Für den vorliegenden Fall ist keine Geeignetheitserklärung zu erstellen."""


def _mindestanteil(komponente: object) -> str:
    """Formatiert den verlangten Mindestanteil einer Nachhaltigkeitsausprägung.

    Für eine **nicht** verlangte Ausprägung gibt es keinen Mindestanteil; das
    Feld trägt dann einen Gedankenstrich und nicht null Prozent — die beiden
    Aussagen sind verschieden. Für eine verlangte Ausprägung erzwingt der
    Domänenkern eine Zahl (``SustainabilityComponent``), sodass der Fall
    „verlangt, aber ohne Angabe“ nicht auftreten kann.

    Args:
        komponente: Die Ausprägung.

    Returns:
        Der formatierte Anteil oder ein Gedankenstrich.
    """
    if not komponente.requested:  # type: ignore[attr-defined]
        return "—"
    anteil = komponente.minimum_share  # type: ignore[attr-defined]
    return "ohne Angabe" if anteil is None else format_de.percent(anteil)


def _profilabschnitt(advice: Advice) -> list[Block]:
    """Baut Abschnitt 2 — das erfasste Kundenprofil.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Die Blöcke des Abschnitts.
    """
    profil = advice.profile
    haushalt = advice.client.household
    bloecke = [
        Block(BlockKind.HEADING, text="2. Erfasstes Kundenprofil"),
        Block(
            BlockKind.TABLE,
            rows=(
                ("Angabe", "Wert"),
                ("Kundenkennung", advice.client.client_id),
                ("Monatliches Nettoeinkommen", format_de.euro(haushalt.net_monthly_income)),
                ("Monatliche Fixkosten", format_de.euro(haushalt.fixed_monthly_costs)),
                ("Monatlicher Kapitaldienst", format_de.euro(haushalt.monthly_debt_service)),
                ("Frei verfügbarer Sparbetrag", format_de.euro(haushalt.free_savings_amount)),
                (
                    "Liquiditätsreserve",
                    f"{format_de.decimal(haushalt.liquidity_reserve_months, 1)} Monatsausgaben",
                ),
                ("Bestehende Verbindlichkeiten", format_de.euro(haushalt.total_debt)),
                ("Sonstiges Vermögen", format_de.euro(haushalt.other_assets)),
                ("Beschäftigungsverhältnis", haushalt.employment_type.value),
            ),
        ),
        Block(BlockKind.SUBHEADING, text="2.1 Risikotragfähigkeit"),
        Block(
            BlockKind.PARAGRAPH,
            rule_id="CAP-CLASS",
            text=render(
                "CAP-CLASS",
                punkte=format_de.decimal(profil.capacity.total_points, 2),
                klasse_aus_punkten=profil.capacity.class_from_points,
                klasse=profil.capacity.risk_class,
            ),
        ),
    ]

    if profil.capacity.ko_triggered:
        bloecke.append(
            Block(
                BlockKind.NOTICE,
                rule_id="CAP-KO-LIQ",
                text=render(
                    "CAP-KO-LIQ",
                    reserve=format_de.decimal(haushalt.liquidity_reserve_months, 1),
                    mindest=format_de.decimal(advice.policies.capacity.mindestreserve_monate, 1),
                ),
            )
        )

    bloecke += [
        Block(BlockKind.SUBHEADING, text="2.2 Risikobereitschaft"),
        Block(
            BlockKind.PARAGRAPH,
            rule_id="TOL-CLASS",
            text=render(
                "TOL-CLASS",
                punkte=format_de.decimal(profil.tolerance.total_points, 2),
                klasse=profil.tolerance.risk_class,
            ),
        ),
        Block(
            BlockKind.PARAGRAPH,
            rule_id="TOL-SCEN",
            text=render(
                "TOL-SCEN",
                bezugsbetrag=format_de.euro(profil.tolerance.reference_amount),
                betrag=format_de.euro(profil.tolerance.presented_amount),
            ),
        ),
    ]

    if profil.tolerance.inconsistencies:
        befund = " ".join(str(eintrag) for eintrag in profil.tolerance.inconsistencies)
        bloecke.append(
            Block(
                BlockKind.PARAGRAPH,
                rule_id="TOL-CONFLICT",
                text=render("TOL-CONFLICT", befund=befund),
            )
        )

    bloecke += [
        Block(BlockKind.SUBHEADING, text="2.3 Kenntnisse und Erfahrungen"),
        Block(
            BlockKind.PARAGRAPH,
            rule_id="KNW-CAP",
            text=render("KNW-CAP", deckel=profil.knowledge.max_risk_class),
        ),
    ]
    return bloecke


def _klassenabschnitt(advice: Advice) -> list[Block]:
    """Baut Abschnitt 3 — die ermittelte Risikoklasse.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Die Blöcke des Abschnitts.
    """
    profil = advice.profile
    bloecke = [
        Block(BlockKind.HEADING, text="3. Ermittelte Risikoklasse"),
        Block(
            BlockKind.TABLE,
            rows=(
                ("Achse", "Punkte", "Klasse"),
                (
                    "Risikotragfähigkeit",
                    format_de.decimal(profil.capacity.total_points, 2),
                    str(profil.capacity.risk_class),
                ),
                (
                    "Risikobereitschaft",
                    format_de.decimal(profil.tolerance.total_points, 2),
                    str(profil.tolerance.risk_class),
                ),
                ("Kenntnisse und Erfahrungen", "—", str(profil.knowledge.max_risk_class)),
                (
                    "Maßgeblich",
                    "—",
                    f"{profil.risk_class} — bindend: {profil.binding_axis.value}",
                ),
            ),
        ),
        Block(
            BlockKind.PARAGRAPH,
            rule_id="CLS-MIN",
            text=render(
                "CLS-MIN",
                tragfaehigkeit=profil.capacity.risk_class,
                bereitschaft=profil.tolerance.risk_class,
                deckel=profil.knowledge.max_risk_class,
                klasse=profil.class_before_glidepath,
                achse=profil.axis_before_glidepath.value,
                endklasse=profil.risk_class,
            ),
        ),
    ]

    if profil.divergence is not None:
        bloecke.append(
            Block(
                BlockKind.PARAGRAPH,
                rule_id="CLS-DIV",
                text=render("CLS-DIV", abstand=profil.divergence.gap),
            )
        )

    if profil.glidepath_class is not None and profil.binding_axis is Axis.GLIDEPATH:
        bloecke.append(
            Block(
                BlockKind.PARAGRAPH,
                rule_id="GLD-CAP",
                text=render(
                    "GLD-CAP",
                    restlaufzeit=advice.goal.horizon_months,
                    deckel=profil.glidepath_class,
                ),
            )
        )
    return bloecke


def _zielabschnitt(advice: Advice) -> list[Block]:
    """Baut Abschnitt 4 — Anlageziel und erforderliche Rendite.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Die Blöcke des Abschnitts.
    """
    ziel = advice.goal
    rendite = advice.required
    return [
        Block(BlockKind.HEADING, text="4. Anlageziel und erforderliche Rendite"),
        Block(
            BlockKind.TABLE,
            rows=(
                ("Angabe", "Wert"),
                ("Zielkennung", ziel.goal_id),
                ("Bezeichnung", ziel.name),
                ("Zieltyp", ziel.goal_type.value),
                ("Priorität", ziel.priority.value),
                ("Zielbetrag", format_de.euro(ziel.target_amount)),
                ("Zieltermin", ziel.target_date.strftime("%d.%m.%Y")),
                (
                    "Anlagehorizont",
                    f"{ziel.horizon_months} Monate "
                    f"({format_de.decimal(ziel.horizon_years, 2)} Jahre)",
                ),
                ("Startkapital", format_de.euro(ziel.initial_capital)),
                ("Monatliche Sparrate", format_de.euro(ziel.monthly_contribution)),
                ("Summe der Einzahlungen", format_de.euro(rendite.value_without_return)),
            ),
        ),
        Block(
            BlockKind.PARAGRAPH,
            rule_id="GOAL-RET",
            text=render(
                "GOAL-RET",
                zielbetrag=format_de.euro(ziel.target_amount),
                monate=ziel.horizon_months,
                startkapital=format_de.euro(ziel.initial_capital),
                sparrate=format_de.euro(ziel.monthly_contribution),
                nominal=format_de.percent(rendite.nominal.annual_rate),
                inflation=format_de.percent(rendite.inflation_annual),
                real=format_de.percent(rendite.real_annual),
            ),
        ),
    ]


def _simulationsabschnitt(advice: Advice) -> list[Block]:
    """Baut Abschnitt 5 — die simulierte Zielerreichung.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Die Blöcke des Abschnitts.
    """
    abgleich = advice.reconciliation
    pfade = advice.policies.simulation.standard_pfade
    perzentile = abgleich.percentiles
    return [
        Block(BlockKind.HEADING, text="5. Simulierte Zielerreichung"),
        Block(
            BlockKind.PARAGRAPH,
            rule_id="REC-SIM",
            text=render("REC-SIM", pfade=format_de.integer(pfade)),
        ),
        Block(
            BlockKind.PARAGRAPH,
            rule_id="REC-THR",
            text=render("REC-THR", gefordert=format_de.percent(abgleich.required_probability)),
        ),
        Block(
            BlockKind.PARAGRAPH,
            rule_id="REC-OUT",
            text=render(
                "REC-OUT",
                erreicht=format_de.percent(abgleich.achieved_probability),
                gefordert=format_de.percent(abgleich.required_probability),
                ausgang=abgleich.outcome.value,
            ),
        ),
        Block(
            BlockKind.TABLE,
            rows=(
                ("Perzentil des Endwerts", "Betrag"),
                *(
                    (f"{stufe} %", format_de.euro(wert, decimals=0))
                    for stufe, wert in sorted(perzentile.items())
                ),
                ("Zielbetrag zum Vergleich", format_de.euro(advice.goal.target_amount, decimals=0)),
            ),
        ),
        Block(
            BlockKind.PARAGRAPH,
            text=(
                "Im Misserfolgsfall — also gemittelt allein über die Verläufe, die den "
                "Zielbetrag verfehlen — fehlen im Mittel "
                f"{format_de.euro(abgleich.expected_shortfall, decimals=0)}. Der mediane "
                "größte zwischenzeitliche Rückgang beträgt "
                f"{format_de.percent(abgleich.median_max_drawdown)}."
            ),
        ),
    ]


def _portfolioabschnitt(advice: Advice) -> list[Block]:
    """Baut Abschnitt 6 — das empfohlene Portfolio mit Begründung je Position.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Die Blöcke des Abschnitts.

    Raises:
        SuitabilityError: Wenn kein Portfolio vorliegt.
    """
    optimum = advice.optimization
    momente = advice.moments
    if optimum is None or momente is None:
        msg = "Ohne optimiertes Portfolio ist Abschnitt 6 nicht erzeugbar."
        raise SuitabilityError(msg)

    definition = advice.policies.classes.risikoklassen[optimum.risk_class]
    zeilen: list[tuple[str, ...]] = [
        ("Assetklasse", "Instrument (ISIN)", "Gewicht", "Bandbreite der Klasse")
    ]
    begruendungen: list[str] = []
    for i, schluessel in enumerate(optimum.asset_keys):
        gewicht = float(optimum.weights[i])
        if gewicht < MIN_WEIGHT:
            continue
        instrument = get_instrument(schluessel)
        band = definition.einzelwerte[schluessel]
        zeilen.append(
            (
                instrument.label,
                f"{instrument.fund_name} ({instrument.isin})",
                format_de.percent(gewicht),
                f"{format_de.percent(float(band.min), 0)} bis "
                f"{format_de.percent(float(band.max), 0)}",
            )
        )
        begruendungen.append(
            f"{instrument.label}: {format_de.percent(gewicht)} — zulässige Bandbreite der "
            f"Risikoklasse {optimum.risk_class} ist "
            f"{format_de.percent(float(band.min), 0)} bis {format_de.percent(float(band.max), 0)}; "
            f"erwartete Rendite {format_de.percent(float(momente.annualised_mean[i]))} p. a. bei "
            f"{format_de.percent(float(momente.annualised_volatility[i]))} Volatilität p. a."
        )

    jahresvolatilitaet = optimum.volatility * float(np.sqrt(MONTHS_PER_YEAR))
    jahresrendite = (1.0 + optimum.expected_return) ** MONTHS_PER_YEAR - 1.0

    return [
        Block(BlockKind.HEADING, text="6. Empfohlenes Portfolio"),
        Block(
            BlockKind.PARAGRAPH,
            rule_id="ADV-PF",
            text=render("ADV-PF", klasse=optimum.risk_class),
        ),
        Block(BlockKind.TABLE, rows=tuple(zeilen)),
        Block(BlockKind.SUBHEADING, text="6.1 Begründung je Position"),
        Block(BlockKind.BULLETS, items=tuple(begruendungen)),
        Block(BlockKind.SUBHEADING, text="6.2 Kennzahlen des Portfolios"),
        Block(
            BlockKind.TABLE,
            rows=(
                ("Kennzahl", "Wert"),
                ("Erwartete Rendite p. a.", format_de.percent(jahresrendite)),
                ("Erwartete Volatilität p. a.", format_de.percent(jahresvolatilitaet)),
                (
                    "Schätzverfahren erwartete Rendite",
                    f"{momente.mean_method.value} (Intensität "
                    f"{format_de.decimal(momente.mean_shrinkage, 4)})",
                ),
                (
                    "Schätzverfahren Kovarianz",
                    f"{momente.covariance_method.value} (Intensität "
                    f"{format_de.decimal(momente.covariance_shrinkage, 4)})",
                ),
                ("Verwendeter Solver", optimum.solver),
                ("Beobachtungen der Schätzung", f"{momente.observations} Monate"),
            ),
        ),
    ]


def _erreichte_anteile(advice: Advice) -> dict[str, str]:
    """Liest die erreichten Nachhaltigkeitsanteile aus dem Trace.

    Bewusst über :meth:`Advice.events_for` und nicht durch erneutes Rechnen auf
    den Gewichten: Was im Dokument steht, soll aus dem Entscheidungspfad
    stammen und nicht daneben ermittelt werden. So können Dokument und Trace
    nicht auseinanderlaufen.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Zuordnung der Ausprägung auf den formatierten Anteil. Leer, wenn keine
        Präferenz erhoben wurde oder keine Empfehlung ergeht.
    """
    ereignisse = advice.events_for("ADV-ESG-QUOTE")
    if not ereignisse:
        return {}
    ausgabe = ereignisse[-1].output
    anteile = ausgabe.get("erreichte_anteile") if isinstance(ausgabe, dict) else None
    if not isinstance(anteile, dict):  # pragma: no cover — durch die Regel ausgeschlossen
        return {}
    return {name: format_de.percent(Decimal(str(wert))) for name, wert in anteile.items()}


def _nachhaltigkeitsabschnitt(advice: Advice) -> list[Block]:
    """Baut Abschnitt 7 — die Behandlung der Nachhaltigkeitspräferenzen.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Die Blöcke des Abschnitts.
    """
    praeferenzen = advice.client.sustainability
    universum = advice.universe
    erreicht = _erreichte_anteile(advice)

    if not praeferenzen.has_preferences:
        befund = (
            "Der Kunde hat angegeben, keine Nachhaltigkeitspräferenzen zu haben. Die Angabe "
            "wurde erhoben und ausdrücklich verneint; das Anlageuniversum wurde deshalb nicht "
            "eingeschränkt."
        )
    elif universum is None or not universum.is_reduced:
        befund = (
            "Die angegebenen Präferenzen werden von allen Assetklassen des Anlageuniversums "
            "erfüllt; eine Einschränkung war nicht erforderlich."
        )
    else:
        ausgeschlossen = ", ".join(sorted(universum.excluded))
        befund = (
            f"Das Anlageuniversum wurde vor der Optimierung um {len(universum.excluded)} von "
            f"{len(advice.optimization.asset_keys) if advice.optimization else 0} Assetklassen "
            f"bereinigt. Ausgeschlossen: {ausgeschlossen}. Die ausgeschlossenen Positionen "
            "tragen im empfohlenen Portfolio ein Gewicht von exakt null."
        )

    if praeferenzen.has_preferences:
        befund += (
            " Der Filter schließt nicht konforme Assetklassen vollständig aus; der im "
            "empfohlenen Portfolio erreichte Anteil wird deshalb je verlangter Ausprägung "
            "ausgewiesen und war gegen den verlangten Mindestanteil zu prüfen."
        )

    bloecke = [
        Block(BlockKind.HEADING, text="7. Nachhaltigkeitspräferenzen"),
        Block(BlockKind.PARAGRAPH, rule_id="ADV-ESG", text=render("ADV-ESG", befund=befund)),
        Block(
            BlockKind.TABLE,
            rows=(
                ("Ausprägung", "Verlangt", "Mindestanteil", "Erreicht"),
                (
                    "Ökologisch nachhaltig (VO 2020/852)",
                    "ja" if praeferenzen.taxonomy_aligned.requested else "nein",
                    _mindestanteil(praeferenzen.taxonomy_aligned),
                    erreicht.get("taxonomie", "—"),
                ),
                (
                    "Nachhaltige Investition (VO 2019/2088)",
                    "ja" if praeferenzen.sfdr_sustainable.requested else "nein",
                    _mindestanteil(praeferenzen.sfdr_sustainable),
                    erreicht.get("sfdr", "—"),
                ),
                (
                    "Nachteilige Auswirkungen (PAI)",
                    "ja" if praeferenzen.pai_considered.requested else "nein",
                    _mindestanteil(praeferenzen.pai_considered),
                    erreicht.get("pai", "—"),
                ),
            ),
        ),
    ]

    if universum is not None and universum.is_reduced:
        bloecke.append(
            Block(
                BlockKind.BULLETS,
                items=tuple(
                    f"{name}: {grund}" for name, grund in sorted(universum.excluded.items())
                ),
            )
        )
    return bloecke


def _risikoabschnitt(advice: Advice) -> list[Block]:
    """Baut Abschnitt 8 — Risikohinweise und Kostenausweis.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Die Blöcke des Abschnitts.
    """
    kosten = advice.policies.simulation.kosten
    optimum = advice.optimization

    zeilen: list[tuple[str, ...]] = [("Assetklasse", "Laufende Kosten p. a.", "Spread je Handel")]
    gewichtete_ter = Decimal(0)
    for i, schluessel in enumerate(optimum.asset_keys if optimum else ()):
        gewicht = float(optimum.weights[i]) if optimum else 0.0
        if gewicht < MIN_WEIGHT:
            continue
        eintrag = kosten[schluessel]
        gewichtete_ter += eintrag.ter_jaehrlich * Decimal(str(gewicht))
        zeilen.append(
            (
                get_instrument(schluessel).label,
                format_de.percent(eintrag.ter_jaehrlich),
                format_de.percent(eintrag.spread_einmalig),
            )
        )
    zeilen.append(
        ("Gewichteter Durchschnitt des Portfolios", format_de.percent(gewichtete_ter), "—")
    )

    return [
        Block(BlockKind.HEADING, text="8. Risikohinweise und Kosten"),
        Block(BlockKind.PARAGRAPH, text=static("risiko_allgemein")),
        Block(BlockKind.PARAGRAPH, text=static("risiko_simulation")),
        Block(BlockKind.SUBHEADING, text="8.1 Kostenausweis"),
        Block(BlockKind.TABLE, rows=tuple(zeilen)),
        Block(BlockKind.PARAGRAPH, text=static("risiko_annahmen")),
    ]


def _nachweisabschnitt(advice: Advice, section: int) -> list[Block]:
    """Baut den Nachweisabschnitt — Regelstand, Datenstand, Hash.

    Args:
        advice: Das Ergebnis der Entscheidungskette.
        section: Nummer des Abschnitts im jeweiligen Dokument.

    Returns:
        Die Blöcke des Abschnitts.
    """
    versionen = " · ".join(
        f"{name} {fassung}" for name, fassung in sorted(advice.policies.versions.items())
    )
    return [
        Block(BlockKind.HEADING, text=f"{section}. Nachweis und Reproduzierbarkeit"),
        Block(BlockKind.PARAGRAPH, text=static("reproduktion")),
        Block(
            BlockKind.TABLE,
            rows=(
                ("Angabe", "Wert"),
                ("Fassung des Regelwerks", versionen),
                ("Datenstand (Kurzhash)", advice.snapshot_hash or "nicht angegeben"),
                ("Inhaltshash des Entscheidungspfads", advice.trace_hash),
                ("Aufgezeichnete Entscheidungsschritte", str(len(advice.trace))),
                ("Fassung der Software", advice.package_version),
                ("Rechtsgrundlage der Dokumentation", DOCUMENTATION_LEGAL_BASIS),
            ),
        ),
    ]


def build_document(advice: Advice) -> Document:
    """Baut die Geeignetheitserklärung als Dokumentmodell.

    Args:
        advice: Das Ergebnis der Entscheidungskette.

    Returns:
        Das Dokument.

    Raises:
        SuitabilityError: Wenn keine Empfehlung ergeht. Für diesen Fall ist der
            Konfliktbericht zu erzeugen (:mod:`candor.report.conflict`).
    """
    if not advice.is_recommendation:
        msg = (
            f"Für Kunde {advice.client.client_id} und Ziel {advice.goal.goal_id} ergeht keine "
            f"Empfehlung (Ausgang: {advice.reconciliation.outcome.value}). Eine "
            "Geeignetheitserklärung würde eine Empfehlung dokumentieren, die es nicht gibt. "
            "Für diesen Fall ist der Konfliktbericht zu erzeugen."
        )
        raise SuitabilityError(msg)

    bloecke: list[Block] = [
        Block(BlockKind.TITLE, text=TITLE),
        Block(
            BlockKind.PARAGRAPH,
            text=(
                f"Kunde {advice.client.client_id} · Ziel {advice.goal.goal_id} · "
                f"Stichtag {advice.goal.reference_date.strftime('%d.%m.%Y')}"
            ),
        ),
        Block(BlockKind.HEADING, text="1. Zweck und Rechtsgrundlage"),
        Block(BlockKind.PARAGRAPH, text=static("zweck_geeignetheit")),
        *_profilabschnitt(advice),
        *_klassenabschnitt(advice),
        Block(BlockKind.PAGEBREAK),
        *_zielabschnitt(advice),
        *_simulationsabschnitt(advice),
        Block(BlockKind.PAGEBREAK),
        *_portfolioabschnitt(advice),
        *_nachhaltigkeitsabschnitt(advice),
        Block(BlockKind.PAGEBREAK),
        *_risikoabschnitt(advice),
        *_nachweisabschnitt(advice, 9),
    ]
    return Document(
        title=TITLE,
        subject=f"Geeignetheitserklärung {advice.client.client_id}/{advice.goal.goal_id}",
        blocks=tuple(bloecke),
    )


def write_pdf(advice: Advice, output: Path) -> Path:
    """Erzeugt die Geeignetheitserklärung als PDF.

    Args:
        advice: Das Ergebnis der Entscheidungskette.
        output: Zielpfad der PDF-Datei.

    Returns:
        Der Pfad der geschriebenen Datei.
    """
    dokument = build_document(advice)
    fusszeile = (
        f"CANDOR · {advice.client.client_id}/{advice.goal.goal_id} · Trace {advice.trace_hash[:12]}"
    )
    return render_pdf(dokument, output, fusszeile)


def main(argv: list[str] | None = None) -> int:
    """Einstiegspunkt der Kommandozeile.

    Args:
        argv: Argumente ohne Programmnamen.

    Returns:
        ``0`` bei Erfolg, ``1`` wenn für den Fall keine Empfehlung ergeht.
    """
    # Verzoegert: candor.report.examples importiert dieses Modul; ein Import
    # auf Modulebene waere ein Zirkelbezug.
    from candor.report.examples import EXAMPLES, build_example  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Geeignetheitserklärung erzeugen.")
    parser.add_argument("--fall", choices=sorted(EXAMPLES), default="ausgewogen")
    parser.add_argument("--out", type=Path, default=Path("beispiele"))
    args = parser.parse_args(argv)

    advice = build_example(args.fall)
    try:
        ziel = write_pdf(advice, args.out / f"geeignetheit_{args.fall}.pdf")
    except SuitabilityError as fehler:
        sys.stdout.write(f"Keine Geeignetheitserklärung: {fehler}\n")
        return 1
    sys.stdout.write(f"Geschrieben: {ziel}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
