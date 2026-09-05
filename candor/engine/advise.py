"""Die vollständige Anlageentscheidung von der Erhebung bis zum Portfolio.

Dieses Modul führt Profilierung, Glide-Path, Zielabgleich und
Portfoliokonstruktion zu einer Kette zusammen. Es leistet damit die Ermittlung
und Dokumentation der Anlageentscheidung.

Die Reihenfolge ist nicht beliebig
----------------------------------
1. **Profilierung** — drei Achsen, bindendes Minimum. Ohne sie gibt es keine
   zulässige Risikoklasse, und ohne die keine zulässige Allokation.
2. **Glide-Path** — die Restlaufzeit kann die Klasse zusätzlich deckeln.
   Muss vor der Optimierung wirken, sonst würde ein Portfolio gebaut, das der
   Kunde zum Zieltermin nicht mehr halten dürfte.
3. **Zielabgleich** — Simulation gegen die geforderte Erfolgswahrscheinlichkeit.
   Endet er im Konflikt, **entfällt die Optimierung**: Ein Portfolio zu
   berechnen, das nicht empfohlen werden darf, wäre bestenfalls
   Rechenzeit und schlimmstenfalls ein Missverständnis.
4. **Nachhaltigkeitsfilter und Optimierung** — nur im empfehlbaren Fall.

Der letzte Punkt ist die eigentliche Aussage des Moduls: Die Entscheidung, *ob*
empfohlen wird, steht vor der Entscheidung, *was* empfohlen wird. Das ist die
Reihenfolge, die Art. 54 DelVO (EU) 2017/565 vorgibt, und sie ist hier im
Kontrollfluss abgebildet und nicht bloß im Text behauptet.

Determinismus
-------------
Jeder Schritt schreibt in denselben Recorder. Der Trace der gesamten Kette ist
damit lückenlos und bei gleichem Seed, Profil und Regelstand bitidentisch
(Invariante I4).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from candor import __version__
from candor.data.universe_meta import ASSET_KEYS
from candor.domain.client import Client, SustainabilityPreferences
from candor.domain.goal import Goal
from candor.domain.policy import PolicySet
from candor.domain.result import Reconciliation, RequiredReturn, RiskProfile
from candor.engine.glidepath import apply_glidepath_for_goal
from candor.engine.reconcile import reconcile
from candor.engine.simulate import FloatArray, ReturnSampler
from candor.goals.required_return import required_return
from candor.portfolio.estimate import MomentEstimate, estimate_moments
from candor.portfolio.optimize import (
    OptimizationResult,
    SustainabilityConflictError,
    SustainabilityQuotaError,
    SustainabilityUniverseError,
    UniverseFilter,
    achieved_shares,
    filter_universe,
    optimize,
)
from candor.portfolio.universe import reference_weights
from candor.profiling.classify import classify
from candor.trace.events import Rule, TraceEvent
from candor.trace.recorder import TraceMetadata, TraceRecorder

#: Rechtsgrundlage der Geeignetheitsprüfung insgesamt.
LEGAL_BASIS: Final = "Art. 54 DelVO (EU) 2017/565"

#: Rechtsgrundlage der Dokumentationspflicht.
DOCUMENTATION_LEGAL_BASIS: Final = (
    "Art. 54 Abs. 12 DelVO (EU) 2017/565 in Verbindung mit § 64 Abs. 4 WpHG"
)

RULE_PORTFOLIO = Rule(
    rule_id="ADV-PF",
    description=(
        "Das empfohlene Portfolio wird durch restringierte Mean-Variance-Optimierung "
        "unter den Bandbreiten der zulässigen Risikoklasse bestimmt."
    ),
    legal_basis=LEGAL_BASIS,
)

RULE_SUSTAINABILITY = Rule(
    rule_id="ADV-ESG",
    description=(
        "Das Anlageuniversum wird vor der Optimierung um Assetklassen bereinigt, die die "
        "Nachhaltigkeitspräferenzen des Kunden nicht erfüllen."
    ),
    legal_basis=("Art. 54 Abs. 9 und 10 DelVO (EU) 2017/565 i. d. F. der DelVO (EU) 2021/1253"),
)

RULE_SUSTAINABILITY_QUOTA = Rule(
    rule_id="ADV-ESG-QUOTE",
    description=(
        "Nach der Optimierung wird je verlangter Ausprägung der im empfohlenen Portfolio "
        "erreichte Anteil ermittelt und gegen den vom Kunden verlangten Mindestanteil "
        "geprüft. Wird er unterschritten, ergeht keine Empfehlung."
    ),
    legal_basis=("Art. 54 Abs. 9 und 10 DelVO (EU) 2017/565 i. d. F. der DelVO (EU) 2021/1253"),
)

RULE_NO_PORTFOLIO = Rule(
    rule_id="ADV-STOP",
    description=(
        "Im Konfliktfall wird kein Portfolio berechnet. Die Entscheidung über das Ob der "
        "Empfehlung geht der Entscheidung über das Was voraus."
    ),
    legal_basis="Art. 54 Abs. 10 DelVO (EU) 2017/565",
)


@dataclass(frozen=True, slots=True)
class Advice:
    """Das vollständige Ergebnis einer Anlageentscheidung.

    Attributes:
        client: Erhobenes Kundenprofil.
        goal: Erhobenes Anlageziel.
        profile: Ermitteltes Risikoprofil, gegebenenfalls durch den Glide-Path
            gedeckelt.
        reconciliation: Ergebnis des Zielabgleichs.
        required: Erforderliche Rendite des Ziels, nominal und real.
        moments: Geschätzte Momente, sofern optimiert wurde.
        optimization: Optimiertes Portfolio, sofern eine Empfehlung zulässig ist.
        universe: Ergebnis des Nachhaltigkeitsfilters.
        policies: Regelwerk, unter dem entschieden wurde.
        trace: Vollständiger Entscheidungspfad über alle Schritte — Grundlage
            jedes erzeugten Dokuments.
        trace_hash: Inhaltshash des Trace (Invariante I4).
        snapshot_hash: Kurzhash des verwendeten Datenstands.
        package_version: Fassung der Software.
    """

    client: Client
    goal: Goal
    profile: RiskProfile
    reconciliation: Reconciliation
    required: RequiredReturn
    moments: MomentEstimate | None
    optimization: OptimizationResult | None
    universe: UniverseFilter | None
    policies: PolicySet
    trace: tuple[TraceEvent, ...]
    trace_hash: str
    snapshot_hash: str
    package_version: str

    @property
    def is_recommendation(self) -> bool:
        """Gibt an, ob eine Anlageempfehlung ergeht."""
        return self.optimization is not None

    @property
    def has_conflict(self) -> bool:
        """Gibt an, ob der Zielabgleich im Konflikt endete."""
        return self.reconciliation.conflict_report is not None

    def events_for(self, rule_id: str) -> tuple[TraceEvent, ...]:
        """Liefert alle Trace-Ereignisse einer Regel in ihrer Reihenfolge.

        Die erzeugten Dokumente beziehen ihren Inhalt über diese Methode aus
        dem Trace, statt die Ergebnisobjekte erneut auszuwerten. So kann kein
        Dokument etwas behaupten, was nicht im Trace steht.

        Args:
            rule_id: Kennung der Regel, etwa ``CAP-KO-LIQ``.

        Returns:
            Die passenden Ereignisse; leer, wenn die Regel nicht angewandt wurde.
        """
        return tuple(ereignis for ereignis in self.trace if ereignis.rule_id == rule_id)


def check_sustainability_quotas(
    preferences: SustainabilityPreferences,
    optimum: OptimizationResult,
    policies: PolicySet,
    recorder: TraceRecorder,
    *,
    asset_keys: tuple[str, ...],
) -> None:
    """Prüft das empfohlene Portfolio gegen die verlangten Mindestanteile.

    Die Prüfung schließt die Lücke zwischen erhobener Kundenvorgabe und
    tatsächlichem Ergebnis: Ein Mindestanteil, der in der Geeignetheitserklärung
    ausgewiesen, aber nie gegen das Portfolio gehalten wird, ist eine unbelegte
    Angabe.

    Beim gegenwärtigen Stand kann die Prüfung nicht fehlschlagen, weil
    :func:`~candor.portfolio.optimize.filter_universe` nicht konforme
    Assetklassen vollständig ausschließt und der erreichte Anteil damit stets
    eins beträgt. Sie wird dennoch gerechnet und protokolliert: Sie ist der
    **Nachweis** dieser Übereinstimmung, und sie wird zur wirksamen Schranke,
    sobald der Ausschluss durch eine Quote ersetzt würde.

    Args:
        preferences: Nachhaltigkeitspräferenzen des Kunden.
        optimum: Ergebnis der Optimierung.
        policies: Geprüftes Regelwerk.
        recorder: Recorder, in den die Prüfung geschrieben wird.
        asset_keys: Assetklassen in Spaltenreihenfolge.

    Raises:
        SustainabilityQuotaError: Wenn ein verlangter Mindestanteil
            unterschritten wird. Eine Empfehlung, die eine erhobene Vorgabe
            verfehlt, ergeht nicht (Prinzip P5).
    """
    if not preferences.has_preferences:
        return

    verlangt = {
        "taxonomie": preferences.taxonomy_aligned.minimum_share,
        "sfdr": preferences.sfdr_sustainable.minimum_share,
        "pai": preferences.pai_considered.minimum_share,
    }
    erreicht = achieved_shares(optimum.weights, preferences, policies.portfolio, asset_keys)
    toleranz = Decimal(str(policies.portfolio.optimierung.restriktionstoleranz))

    unterschritten: dict[str, tuple[Decimal, Decimal]] = {}
    for merkmal, anteil in erreicht.items():
        vorgabe = verlangt[merkmal]
        if vorgabe is not None and anteil < vorgabe - toleranz:
            unterschritten[merkmal] = (vorgabe, anteil)

    recorder.record(
        RULE_SUSTAINABILITY_QUOTA,
        {
            "verlangte_mindestanteile": {
                merkmal: wert for merkmal, wert in sorted(verlangt.items()) if wert is not None
            }
        },
        {
            "erreichte_anteile": dict(sorted(erreicht.items())),
            "alle_vorgaben_eingehalten": not unterschritten,
        },
        policies.portfolio.version,
    )

    if unterschritten:
        raise SustainabilityQuotaError(unterschritten)


def advise(  # noqa: PLR0913 — jede Angabe ist eine eigene fachliche Groesse
    client: Client,
    goal: Goal,
    policies: PolicySet,
    sampler: ReturnSampler,
    returns: FloatArray,
    *,
    seed: int | None = None,
    paths: int | None = None,
    snapshot_hash: str = "",
    asset_keys: tuple[str, ...] = ASSET_KEYS,
) -> Advice:
    """Führt die vollständige Entscheidungskette aus.

    Args:
        client: Erhobenes Kundenprofil.
        goal: Erhobenes Anlageziel.
        policies: Vollständiges Regelwerk.
        sampler: Ziehungsverfahren der Monatsrenditen.
        returns: Historische Renditematrix für die Momentenschätzung.
        seed: Abweichender Seed. Ohne Angabe der Wert aus der Policy.
        paths: Abweichende Pfadzahl. Ohne Angabe der Wert aus der Policy.
        snapshot_hash: Kurzhash des Datenstands für die Dokumentation.
        asset_keys: Assetklassen in Spaltenreihenfolge.

    Returns:
        Das vollständige Ergebnis samt Trace.

    Raises:
        SustainabilityConflictError: Wenn Nachhaltigkeitspräferenz und
            Risikoklasse unvereinbar sind.
        SustainabilityUniverseError: Wenn der Filter zu wenige Assetklassen
            übrig lässt.
    """
    recorder = TraceRecorder(
        TraceMetadata(
            client_id=client.client_id,
            goal_id=goal.goal_id,
            policy_versions=dict(policies.versions),
            package_version=__version__,
            seed=seed if seed is not None else policies.simulation.basis_seed,
        )
    )

    # Die erforderliche Rendite gehört in den Entscheidungspfad, nicht in den
    # Bericht: Sie ist eine Rechengröße der Entscheidung. Würde sie erst beim
    # Erzeugen des Dokuments bestimmt, stünde im Dokument eine Zahl, die der
    # Trace nicht kennt — und die Zusicherung „was im Dokument steht, steht im
    # Trace" wäre gebrochen.
    erforderlich = required_return(goal, policies.goals, recorder=recorder)

    profil = classify(client, goal, policies, recorder)
    profil = apply_glidepath_for_goal(profil, goal, policies.glidepath, recorder)
    abgleich = reconcile(
        client,
        goal,
        profil,
        policies,
        sampler,
        recorder,
        asset_keys=asset_keys,
        paths=paths,
        seed=seed,
    )

    momente: MomentEstimate | None = None
    optimum: OptimizationResult | None = None
    universum: UniverseFilter | None = None

    if abgleich.allows_recommendation:
        universum = filter_universe(client.sustainability, policies.portfolio, asset_keys)
        recorder.record(
            RULE_SUSTAINABILITY,
            {
                "praeferenzen_vorhanden": client.sustainability.has_preferences,
                "universum_vollstaendig": list(asset_keys),
            },
            {
                "zugelassen": list(universum.included),
                "ausgeschlossen": sorted(universum.excluded),
                "filter_angewandt": universum.applied,
            },
            policies.portfolio.version,
        )

        momente = estimate_moments(returns, policies.portfolio.schaetzung, asset_keys)
        referenz = reference_weights(profil.risk_class, policies.classes, asset_keys)
        zielrendite = float(referenz @ momente.mean)
        optimum = optimize(
            momente,
            profil.risk_class,
            policies.classes,
            policies.portfolio,
            asset_keys,
            target_return=zielrendite,
            universe=universum,
        )
        recorder.record(
            RULE_PORTFOLIO,
            {
                "risikoklasse": profil.risk_class,
                "zielrendite_monatlich": round(zielrendite, 8),
                "schaetzfenster_monate": momente.observations,
            },
            optimum.as_summary(),
            policies.portfolio.version,
        )
        check_sustainability_quotas(
            client.sustainability, optimum, policies, recorder, asset_keys=asset_keys
        )
    else:
        recorder.record(
            RULE_NO_PORTFOLIO,
            {"ausgang": abgleich.outcome.value},
            {"portfolio_berechnet": False},
            policies.combined_version,
        )

    return Advice(
        client=client,
        goal=goal,
        profile=profil,
        reconciliation=abgleich,
        required=erforderlich,
        moments=momente,
        optimization=optimum,
        universe=universum,
        policies=policies,
        trace=recorder.events,
        trace_hash=recorder.content_hash(),
        snapshot_hash=snapshot_hash,
        package_version=__version__,
    )


__all__ = [
    "DOCUMENTATION_LEGAL_BASIS",
    "LEGAL_BASIS",
    "Advice",
    "SustainabilityConflictError",
    "SustainabilityQuotaError",
    "SustainabilityUniverseError",
    "advise",
    "check_sustainability_quotas",
]
