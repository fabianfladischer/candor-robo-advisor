"""Zusammenführung der drei Achsen zu einer Risikoklasse.

Die Regel lautet::

    risikoklasse = min(tragfaehigkeit, bereitschaft, kenntnisdeckel)

**Es wird niemals gemittelt.** Diese Regel ist der inhaltliche Kern des
gesamten Projekts: Die Achsen messen verschiedene Dinge, und keine kann die
andere ersetzen. Ein Mittelwert setzt Ersetzbarkeit voraus und erzeugt genau
dort eine Empfehlung, wo eigentlich keine zulässig wäre.

Divergenz
---------
Liegen Tragfähigkeit und Bereitschaft um mehr als eine Stufe auseinander, wird
das als eigener Befund geführt. Der Abstand ist die eigentliche Information: Ein
Kunde, der viel riskieren könnte, aber nicht will, braucht ein anderes Gespräch
als einer, der will, aber nicht kann. Ein Mittelwert liefert für beide Fälle
dieselbe Zahl und verschweigt den Unterschied.

Bindende Achse bei Gleichstand
------------------------------
Erreichen mehrere Achsen dasselbe Minimum, wird diejenige ausgewiesen, die am
schwersten zu ändern ist. Die Reihenfolge lautet: K.-o.-Kriterium der
Liquidität, dann Kenntnisdeckel, dann Tragfähigkeit, dann Bereitschaft. So nennt
das Ergebnis stets den Hebel, an dem der Kunde zuerst ansetzen müsste — und
nicht den, der zufällig zuerst geprüft wurde.
"""

from __future__ import annotations

from typing import Final

from candor.domain.client import Client
from candor.domain.goal import Goal
from candor.domain.policy import PolicySet
from candor.domain.result import (
    Axis,
    CapacityResult,
    Divergence,
    KnowledgeResult,
    RiskProfile,
    ToleranceResult,
)
from candor.profiling.capacity import assess_capacity
from candor.profiling.knowledge import assess_knowledge
from candor.profiling.tolerance import assess_tolerance
from candor.trace.events import Rule
from candor.trace.recorder import TraceRecorder

#: Rechtsgrundlage der Gesamtbeurteilung.
LEGAL_BASIS: Final = "Art. 54 Abs. 2 und Abs. 10 DelVO (EU) 2017/565"

#: Abstand in Stufen, ab dem eine Divergenz als solche ausgewiesen wird.
DIVERGENCE_THRESHOLD: Final = 2

RULE_MIN = Rule(
    rule_id="CLS-MIN",
    description=(
        "Die Risikoklasse ist das Minimum aus Risikotragfähigkeit, Risikobereitschaft "
        "und Kenntnisdeckel. Es wird nicht gemittelt: Die Achsen messen verschiedene "
        "Dinge und können einander nicht ersetzen."
    ),
    legal_basis=LEGAL_BASIS,
)
RULE_DIVERGENCE = Rule(
    rule_id="CLS-DIV",
    description=(
        "Liegen Tragfähigkeit und Bereitschaft um mindestens zwei Stufen auseinander, "
        "wird die Divergenz als eigener Befund ausgewiesen."
    ),
    legal_basis=LEGAL_BASIS,
)


def _binding_axis(
    capacity: CapacityResult,
    knowledge: KnowledgeResult,
    final_class: int,
) -> Axis:
    """Bestimmt die Achse, die das Ergebnis bindet.

    Bei Gleichstand entscheidet die Änderbarkeit: Zuerst wird das
    K.-o.-Kriterium genannt, dann der Kenntnisdeckel, dann die Tragfähigkeit,
    zuletzt die Bereitschaft.

    Args:
        capacity: Ergebnis der Tragfähigkeit.
        knowledge: Ergebnis der Kenntnisse.
        final_class: Ermittelte Endklasse.

    Returns:
        Die bindende Achse. Trifft keine der drei geprüften Bedingungen zu,
        bindet die Bereitschaft — sie ist dann die einzige verbleibende Achse
        mit dem Minimalwert.
    """
    if capacity.ko_triggered and capacity.risk_class == final_class:
        return Axis.KO_LIQUIDITAET
    if knowledge.max_risk_class == final_class:
        return Axis.KENNTNISSE
    if capacity.risk_class == final_class:
        return Axis.TRAGFAEHIGKEIT
    return Axis.BEREITSCHAFT


def _divergence(capacity: CapacityResult, tolerance: ToleranceResult) -> Divergence | None:
    """Ermittelt, ob die beiden Achsen erheblich auseinanderliegen.

    Args:
        capacity: Ergebnis der Tragfähigkeit.
        tolerance: Ergebnis der Bereitschaft.

    Returns:
        Der Befund oder ``None``, wenn der Abstand unter der Schwelle bleibt.
    """
    abstand = abs(capacity.risk_class - tolerance.risk_class)
    if abstand < DIVERGENCE_THRESHOLD:
        return None

    if capacity.risk_class < tolerance.risk_class:
        bindend = Axis.TRAGFAEHIGKEIT
        beschreibung = (
            f"Der Kunde ist zu Klasse {tolerance.risk_class} bereit, seine wirtschaftlichen "
            f"Verhältnisse tragen aber nur Klasse {capacity.risk_class}. Bindend ist die "
            "Tragfähigkeit: Der Wunsch übersteigt die Belastbarkeit."
        )
    else:
        bindend = Axis.BEREITSCHAFT
        beschreibung = (
            f"Die wirtschaftlichen Verhältnisse trügen Klasse {capacity.risk_class}, der "
            f"Kunde ist aber nur zu Klasse {tolerance.risk_class} bereit. Bindend ist die "
            "Bereitschaft: Die Belastbarkeit übersteigt den Wunsch."
        )

    return Divergence(
        capacity_class=capacity.risk_class,
        tolerance_class=tolerance.risk_class,
        gap=abstand,
        binding_axis=bindend,
        description=beschreibung,
    )


def classify(
    client: Client,
    goal: Goal,
    policies: PolicySet,
    recorder: TraceRecorder,
) -> RiskProfile:
    """Ermittelt das Risikoprofil aus den drei Achsen.

    Der Glide-Path wird hier **nicht** angewandt. Er hängt an der Restlaufzeit
    und nicht am Profil; die Anwendung erfolgt über
    :func:`candor.engine.glidepath.apply_glidepath`.

    Args:
        client: Geprüftes Kundenprofil.
        goal: Geprüftes Anlageziel.
        policies: Vollständiges Regelwerk.
        recorder: Recorder, in den alle Schritte geschrieben werden.

    Returns:
        Das Risikoprofil mit Klasse, bindender Achse, Punkteständen, Divergenz
        und vollständigem Trace.
    """
    capacity = assess_capacity(client, goal, policies.capacity, recorder)
    tolerance = assess_tolerance(client, goal, policies.tolerance, recorder)
    knowledge = assess_knowledge(client, policies.knowledge, recorder)

    endklasse = min(capacity.risk_class, tolerance.risk_class, knowledge.max_risk_class)
    bindend = _binding_axis(capacity, knowledge, endklasse)

    recorder.record(
        RULE_MIN,
        {
            "tragfaehigkeit_klasse": capacity.risk_class,
            "tragfaehigkeit_punkte": capacity.total_points,
            "bereitschaft_klasse": tolerance.risk_class,
            "bereitschaft_punkte": tolerance.total_points,
            "kenntnisdeckel": knowledge.max_risk_class,
            "verfahren": "Minimum, keine Mittelung",
        },
        {"risikoklasse": endklasse, "bindende_achse": bindend.value},
        policies.combined_version,
    )

    divergenz = _divergence(capacity, tolerance)
    if divergenz is not None:
        recorder.record(
            RULE_DIVERGENCE,
            {
                "tragfaehigkeit_klasse": divergenz.capacity_class,
                "bereitschaft_klasse": divergenz.tolerance_class,
                "schwelle_stufen": DIVERGENCE_THRESHOLD,
            },
            {
                "abstand_stufen": divergenz.gap,
                "bindende_achse": divergenz.binding_axis.value,
                "befund": divergenz.description,
            },
            policies.combined_version,
        )

    return RiskProfile(
        risk_class=endklasse,
        binding_axis=bindend,
        class_before_glidepath=endklasse,
        axis_before_glidepath=bindend,
        capacity=capacity,
        tolerance=tolerance,
        knowledge=knowledge,
        divergence=divergenz,
        glidepath_class=None,
        trace=recorder.events,
        trace_hash=recorder.content_hash(),
        policy_versions=dict(policies.versions),
    )
