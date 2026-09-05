"""Zielabgleich — die Stelle, an der eine Empfehlung verweigert wird.

Hier trifft alles aufeinander: das Risikoprofil, das Anlageziel und die
Simulation. Verglichen wird eine einzige Zahl mit einer einzigen Schwelle:

    erreichte Erfolgswahrscheinlichkeit  gegen  geforderte Mindestwahrscheinlichkeit

Die erreichte Wahrscheinlichkeit wird **unter der zulässigen Risikoklasse**
ermittelt — also unter der Klasse, die nach dem bindenden Minimum aus
Tragfähigkeit, Bereitschaft, Kenntnissen und Glide-Path übrig bleibt. Genau
darin liegt der Unterschied zu einem gewöhnlichen Renditerechner: Gerechnet wird
nicht mit der Rendite, die das Ziel bräuchte, sondern mit der, die der Kunde
tragen darf.

Drei Ausgänge
-------------
* :attr:`~candor.domain.result.ReconcileOutcome.ERREICHBAR` — die Schwelle ist
  mit Abstand überschritten. Eine Empfehlung ist zulässig.
* :attr:`~candor.domain.result.ReconcileOutcome.ERREICHBAR_MIT_PUFFER` — die
  Schwelle ist erreicht, aber der Abstand liegt innerhalb des Pufferbands aus
  dem Regelwerk. Die Empfehlung ergeht mit ausdrücklichem Hinweis auf den
  geringen Abstand. Der Fall ist eigens ausgewiesen, weil eine Zielerreichung
  mit 85,4 % bei einer Schwelle von 85 % etwas anderes ist als eine mit 97 % —
  auch wenn beide „erreicht“ heißen.
* :attr:`~candor.domain.result.ReconcileOutcome.KONFLIKT` — die Schwelle ist
  nicht erreicht. **Es ergeht keine Empfehlung.** An ihre Stelle tritt der
  Konfliktbericht mit den drei quantifizierten Stellschrauben.

Rechtsgrundlage der Verweigerung
--------------------------------
Art. 54 Abs. 10 DelVO (EU) 2017/565: Ist kein geeignetes Produkt vorhanden, darf
keine Empfehlung ausgesprochen werden. Die Vorschrift kennt keine Verrechnung
eines Eignungsmangels mit einer Stärke an anderer Stelle. Der Verweis wandert
mit jedem Konfliktfall in den Trace — nicht als Zierde, sondern damit im
Streitfall belegt ist, auf welcher Grundlage die Empfehlung unterblieb.

Warum das Pufferband breiter ist als das Rauschen
--------------------------------------------------
Die Erfolgswahrscheinlichkeit ist eine Schätzung mit einem Standardfehler von
rund einem halben Prozentpunkt bei 10 000 Pfaden. Das Pufferband von fünf
Prozentpunkten entspricht etwa dem Zehnfachen. Damit hängt die Einordnung eines
Falls in eines der drei Fächer nicht vom Seed ab — ein Kunde bekäme sonst beim
zweiten Lauf eine andere Auskunft.
"""

from __future__ import annotations

from typing import Final

from candor import format_de
from candor.domain.client import Client
from candor.domain.goal import Goal
from candor.domain.policy import PolicySet
from candor.domain.result import (
    ConflictReport,
    Feasibility,
    Lever,
    ReconcileOutcome,
    Reconciliation,
    RiskProfile,
)
from candor.engine.simulate import FloatArray, ReturnSampler, simulate
from candor.engine.solve import SolverContext, build_levers
from candor.portfolio.universe import reference_weights, weights_as_mapping
from candor.trace.events import Rule
from candor.trace.recorder import TraceRecorder

#: Rechtsgrundlage der Empfehlungsverweigerung.
CONFLICT_LEGAL_BASIS: Final = "Art. 54 Abs. 10 DelVO (EU) 2017/565"

#: Rechtsgrundlage der Geeignetheitsbeurteilung insgesamt.
LEGAL_BASIS: Final = "Art. 54 DelVO (EU) 2017/565"

RULE_SIMULATION = Rule(
    rule_id="REC-SIM",
    description=(
        "Unter der zulässigen Risikoklasse wird die Erfolgswahrscheinlichkeit des "
        "Anlageziels simuliert; maßgeblich sind die Zielgewichte dieser Klasse, nicht "
        "die für das Ziel rechnerisch nötige Rendite."
    ),
    legal_basis=LEGAL_BASIS,
)
RULE_THRESHOLD = Rule(
    rule_id="REC-THR",
    description=(
        "Die geforderte Mindest-Erfolgswahrscheinlichkeit wird aus Zielpriorität, "
        "Zieltyp und individueller Vorgabe nachgeschlagen; sie kann durch keine der "
        "drei Stufen gesenkt werden."
    ),
    legal_basis=LEGAL_BASIS,
)
RULE_OUTCOME = Rule(
    rule_id="REC-OUT",
    description=(
        "Erreichte und geforderte Erfolgswahrscheinlichkeit werden verglichen; das "
        "Ergebnis ist einer von drei Ausgängen. Unterschreitet die erreichte die "
        "geforderte, ergeht keine Empfehlung."
    ),
    legal_basis=LEGAL_BASIS,
)
RULE_CONFLICT = Rule(
    rule_id="REC-CONFLICT",
    description=(
        "Die geforderte Erfolgswahrscheinlichkeit wird unter der zulässigen "
        "Risikoklasse nicht erreicht. Eine Anlageempfehlung ist unzulässig; an ihre "
        "Stelle tritt ein Konfliktbericht mit drei quantifizierten Stellschrauben."
    ),
    legal_basis=CONFLICT_LEGAL_BASIS,
)
RULE_LEVER = Rule(
    rule_id="REC-LEVER",
    description=(
        "Eine Stellschraube wurde numerisch auf die geforderte "
        "Erfolgswahrscheinlichkeit gelöst und gegen die erhobenen Verhältnisse auf "
        "Umsetzbarkeit geprüft."
    ),
    legal_basis=CONFLICT_LEGAL_BASIS,
)


def classify_outcome(achieved: float, required: float, buffer_band: float) -> ReconcileOutcome:
    """Ordnet das Verhältnis von erreichter und geforderter Wahrscheinlichkeit ein.

    Args:
        achieved: Erreichte Erfolgswahrscheinlichkeit.
        required: Geforderte Mindest-Erfolgswahrscheinlichkeit.
        buffer_band: Breite des Pufferbands.

    Returns:
        Der Ausgang des Abgleichs.
    """
    if achieved < required:
        return ReconcileOutcome.KONFLIKT
    if achieved < required + buffer_band:
        return ReconcileOutcome.ERREICHBAR_MIT_PUFFER
    return ReconcileOutcome.ERREICHBAR


def build_diagnosis(
    goal: Goal,
    profile: RiskProfile,
    required: float,
    achieved: float,
    expected_shortfall: float,
) -> tuple[str, ...]:
    """Formuliert den Befund in Stichpunkten.

    Bewusst Stichpunkte und kein Fließtext: Der Konfliktbericht soll den Grund
    benennen, nicht ihn erklären. Die Erklärung ist Aufgabe des Beraters.

    Args:
        goal: Anlageziel.
        profile: Risikoprofil.
        required: Geforderte Mindest-Erfolgswahrscheinlichkeit.
        achieved: Erreichte Erfolgswahrscheinlichkeit.
        expected_shortfall: Erwarteter Fehlbetrag im Misserfolgsfall.

    Returns:
        Die Stichpunkte des Befunds.
    """
    luecke = required - achieved
    achse = "\u201e" + profile.binding_axis.value + "\u201c"
    punkte = [
        f"Ziel {goal.goal_id} ({goal.goal_type.value}, {goal.priority.value}): "
        f"{format_de.euro(goal.target_amount)} in {goal.horizon_months} Monaten.",
        f"Gefordert sind {format_de.percent(required)} Erfolgswahrscheinlichkeit, erreicht "
        f"werden {format_de.percent(achieved)} — es fehlen "
        f"{format_de.percentage_points(luecke)}.",
        f"Maßgeblich ist Risikoklasse {profile.risk_class}; sie wird durch die Achse "
        f"{achse} begrenzt.",
        f"Im Misserfolgsfall fehlen im Mittel {format_de.euro(expected_shortfall, decimals=0)}.",
    ]

    if profile.capacity.ko_triggered:
        punkte.append(
            "Die Risikoklasse ist durch das K.-o.-Kriterium der Liquiditätsreserve auf 1 "
            "gesetzt. Vor jeder Anlageentscheidung ist die Rücklage aufzubauen."
        )
    if profile.divergence is not None:
        punkte.append(profile.divergence.as_note())
    if (
        profile.glidepath_class is not None
        and profile.glidepath_class < profile.capacity.risk_class
    ):
        punkte.append(
            f"Die Restlaufzeit begrenzt die zulässige Klasse zusätzlich auf "
            f"{profile.glidepath_class} (Glide-Path)."
        )
    punkte.append(
        "Eine Anlageempfehlung ergeht nicht. Rechtsgrundlage: " + CONFLICT_LEGAL_BASIS + "."
    )
    return tuple(punkte)


def reconcile(  # noqa: PLR0913, PLR0917 — jede Angabe ist eine eigene fachliche Groesse
    client: Client,
    goal: Goal,
    profile: RiskProfile,
    policies: PolicySet,
    sampler: ReturnSampler,
    recorder: TraceRecorder,
    *,
    asset_keys: tuple[str, ...],
    paths: int | None = None,
    seed: int | None = None,
    solve_levers: bool = True,
) -> Reconciliation:
    """Gleicht das Anlageziel mit der zulässigen Risikoklasse ab.

    Args:
        client: Geprüftes Kundenprofil.
        goal: Geprüftes Anlageziel.
        profile: Ermitteltes Risikoprofil, gegebenenfalls bereits gedeckelt.
        policies: Vollständiges Regelwerk.
        sampler: Ziehungsverfahren der Monatsrenditen.
        recorder: Recorder des laufenden Trace.
        asset_keys: Assetklassen in Spaltenreihenfolge.
        paths: Zahl der Pfade. Ohne Angabe der Wert aus der Policy.
        seed: Seed. Ohne Angabe der Wert aus der Policy.
        solve_levers: Ob im Konfliktfall die drei Stellschrauben gelöst werden.
            Ausschaltbar, weil die Suche mehrere Simulationen kostet.

    Returns:
        Das Ergebnis des Abgleichs, im Konfliktfall samt Konfliktbericht.
    """
    anzahl_pfade = paths if paths is not None else policies.simulation.standard_pfade
    verwendeter_seed = seed if seed is not None else policies.simulation.basis_seed
    gewichte: FloatArray = reference_weights(profile.risk_class, policies.classes, asset_keys)

    gefordert = float(
        policies.goals.minimum_success_probability(
            goal.priority, goal.required_success_probability, goal.goal_type
        )
    )
    recorder.record(
        RULE_THRESHOLD,
        {
            "prioritaet": goal.priority,
            "zieltyp": goal.goal_type,
            "individuelle_vorgabe": goal.required_success_probability,
        },
        {"geforderte_erfolgswahrscheinlichkeit": gefordert},
        policies.goals.version,
    )

    ergebnis = simulate(
        sampler,
        gewichte,
        initial_capital=float(goal.initial_capital),
        monthly_contribution=float(goal.monthly_contribution),
        horizon_months=goal.horizon_months,
        target_amount=float(goal.target_amount),
        policy=policies.simulation,
        paths=anzahl_pfade,
        seed=verwendeter_seed,
        asset_keys=asset_keys,
        track_bands=False,
    )
    erreicht = ergebnis.success_probability
    perzentile = ergebnis.percentiles()

    recorder.record(
        RULE_SIMULATION,
        {
            "risikoklasse": profile.risk_class,
            "zielgewichte": weights_as_mapping(gewichte, asset_keys),
            "startkapital": goal.initial_capital,
            "sparrate_monatlich": goal.monthly_contribution,
            "horizont_monate": goal.horizon_months,
            "zielbetrag": goal.target_amount,
            "pfade": anzahl_pfade,
            "seed": verwendeter_seed,
            "verfahren": ergebnis.method,
        },
        {
            "erfolgswahrscheinlichkeit": erreicht,
            "perzentile": {str(stufe): round(wert, 2) for stufe, wert in perzentile.items()},
            "erwarteter_fehlbetrag_im_misserfolg": round(ergebnis.expected_shortfall, 2),
            "median_max_drawdown": round(ergebnis.median_max_drawdown, 6),
        },
        policies.simulation.version,
    )

    puffer = float(policies.goals.pufferband)
    ausgang = classify_outcome(erreicht, gefordert, puffer)
    recorder.record(
        RULE_OUTCOME,
        {"erreicht": erreicht, "gefordert": gefordert, "pufferband": puffer},
        {"ausgang": ausgang.value, "empfehlung_zulaessig": ausgang.allows_recommendation},
        policies.combined_version,
    )

    bericht: ConflictReport | None = None
    if ausgang is ReconcileOutcome.KONFLIKT:
        bericht = _build_conflict_report(
            client=client,
            goal=goal,
            profile=profile,
            policies=policies,
            sampler=sampler,
            recorder=recorder,
            weights=gewichte,
            asset_keys=asset_keys,
            paths=anzahl_pfade,
            seed=verwendeter_seed,
            required=gefordert,
            achieved=erreicht,
            expected_shortfall=ergebnis.expected_shortfall,
            solve_levers=solve_levers,
        )

    return Reconciliation(
        outcome=ausgang,
        client_id=client.client_id,
        goal_id=goal.goal_id,
        risk_class=profile.risk_class,
        required_probability=gefordert,
        achieved_probability=erreicht,
        buffer_band=puffer,
        weights=weights_as_mapping(gewichte, asset_keys),
        percentiles=perzentile,
        expected_shortfall=ergebnis.expected_shortfall,
        median_max_drawdown=ergebnis.median_max_drawdown,
        conflict_report=bericht,
        trace=recorder.events,
        trace_hash=recorder.content_hash(),
        policy_versions=dict(policies.versions),
    )


def _build_conflict_report(  # noqa: PLR0913 — der Bericht bündelt den gesamten Fall
    *,
    client: Client,
    goal: Goal,
    profile: RiskProfile,
    policies: PolicySet,
    sampler: ReturnSampler,
    recorder: TraceRecorder,
    weights: FloatArray,
    asset_keys: tuple[str, ...],
    paths: int,
    seed: int,
    required: float,
    achieved: float,
    expected_shortfall: float,
    solve_levers: bool,
) -> ConflictReport:
    """Erstellt den Konfliktbericht samt Stellschrauben.

    Args:
        client: Kundenprofil.
        goal: Anlageziel.
        profile: Risikoprofil.
        policies: Regelwerk.
        sampler: Ziehungsverfahren.
        recorder: Recorder des laufenden Trace.
        weights: Zielgewichte der zulässigen Risikoklasse.
        asset_keys: Assetklassen in Spaltenreihenfolge.
        paths: Zahl der Pfade.
        seed: Fixierter Seed.
        required: Geforderte Erfolgswahrscheinlichkeit.
        achieved: Erreichte Erfolgswahrscheinlichkeit.
        expected_shortfall: Erwarteter Fehlbetrag im Misserfolgsfall.
        solve_levers: Ob die Stellschrauben gelöst werden.

    Returns:
        Der Konfliktbericht.
    """
    diagnose = build_diagnosis(goal, profile, required, achieved, expected_shortfall)
    recorder.record(
        RULE_CONFLICT,
        {
            "erreicht": achieved,
            "gefordert": required,
            "fehlende_prozentpunkte": round((required - achieved) * 100, 4),
            "risikoklasse": profile.risk_class,
            "bindende_achse": profile.binding_axis.value,
        },
        {"empfehlung": None, "befund": list(diagnose)},
        policies.combined_version,
    )

    hebel: tuple[Lever, ...] = ()
    if solve_levers:
        kontext = SolverContext(
            sampler=sampler,
            weights=weights,
            policies=policies,
            paths=paths,
            seed=seed,
            asset_keys=asset_keys,
        )
        hebel = build_levers(kontext, client, goal, required)
        for eintrag in hebel:
            recorder.record(
                RULE_LEVER,
                {
                    "stellschraube": eintrag.lever_id.value,
                    "bezeichnung": eintrag.name,
                    "istwert": eintrag.current_value,
                    "verfahren": eintrag.method,
                    "auswertungen": eintrag.iterations,
                },
                {
                    "sollwert": eintrag.required_value,
                    "differenz": eintrag.difference,
                    "erreichte_wahrscheinlichkeit": round(eintrag.achieved_probability, 6),
                    "machbarkeit": eintrag.feasibility.value,
                    "begruendung": eintrag.feasibility_note,
                },
                policies.combined_version,
            )

    return ConflictReport(
        client_id=client.client_id,
        goal_id=goal.goal_id,
        required_probability=required,
        achieved_probability=achieved,
        risk_class=profile.risk_class,
        binding_axis=profile.binding_axis,
        diagnosis=diagnose,
        levers=hebel,
        legal_basis=CONFLICT_LEGAL_BASIS,
        policy_versions=dict(policies.versions),
    )


def format_probability(value: float) -> str:
    """Formatiert eine Wahrscheinlichkeit mit deutschem Dezimalkomma.

    Args:
        value: Wahrscheinlichkeit zwischen 0 und 1.

    Returns:
        Die formatierte Angabe, etwa ``"85,00 %"``.
    """
    return format_de.percent(value)


def all_levers_infeasible(report: ConflictReport) -> bool:
    """Gibt an, ob keine der Stellschrauben umsetzbar ist.

    Der Fall ist der härteste des Systems: Das Ziel ist unter den erhobenen
    Verhältnissen nicht erreichbar, und keine der drei Änderungen ist gangbar.
    Er wird eigens benannt, statt in der Liste der Stellschrauben unterzugehen.

    Args:
        report: Der Konfliktbericht.

    Returns:
        ``True``, wenn keine Stellschraube umsetzbar ist.
    """
    return bool(report.levers) and all(
        eintrag.feasibility is not Feasibility.UMSETZBAR for eintrag in report.levers
    )
