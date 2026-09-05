"""Glide-Path — Risikoabbau mit abnehmender Restlaufzeit.

Der Glide-Path ist eine **weitere Obergrenze**, keine eigene Achse. Er kann eine
Risikoklasse senken, niemals anheben. Das ist dieselbe Rechenrichtung wie beim
Kenntnisdeckel und aus demselben Grund: Eine kurze Restlaufzeit ist ein Grund
gegen Risiko, eine lange aber kein Grund dafür — ob jemand Schwankung tragen
kann und will, entscheiden weiterhin die beiden Achsen.

Warum überhaupt
---------------
Ein Rückgang lässt sich nur aussitzen, wenn Zeit bleibt. Wer in zwanzig Jahren
Geld braucht, kann eine Krise durchstehen; wer es in achtzehn Monaten braucht,
kann es nicht. Entscheidend ist nicht die erwartete Rendite, sondern die Frage,
ob im Zeitraum bis zum Zieltermin eine Erholung realistisch ist.

Der Effekt ist unabhängig vom Kunden: Er hängt allein an der Restlaufzeit und
wirkt deshalb auch dann, wenn Tragfähigkeit und Bereitschaft hoch sind. Genau
das macht ihn zu einer harten Deckelung und nicht zu einem weiteren Kriterium
im Punktesystem — dort ließe er sich gegen andere Kriterien aufrechnen, und ein
Kunde mit hohem Einkommen käme kurz vor dem Zieltermin weiterhin auf eine hohe
Aktienquote.

Der Anlagehorizont geht zusätzlich als **eines von sechs** Kriterien in die
Tragfähigkeit ein (``policies/capacity_v1.yaml``). Das ist keine Doppelzählung
mit unterschiedlicher Wirkung: Dort beschreibt er die Belastbarkeit zu Beginn,
hier begrenzt er das Risiko zum Ende hin. Beide Wirkungen sind einseitig nach
unten gerichtet, weshalb die Kombination nie zu einer höheren Klasse führen kann.

Die Stufen stehen in ``policies/glidepath_v1.yaml``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from candor.domain.errors import PolicyError
from candor.domain.goal import Goal
from candor.domain.policy import GlidepathPolicy
from candor.domain.result import Axis, RiskProfile
from candor.trace.events import Rule
from candor.trace.recorder import TraceRecorder

#: Rechtsgrundlage der Begrenzung über den Anlagehorizont.
LEGAL_BASIS: Final = "Art. 54 Abs. 2 Buchst. a DelVO (EU) 2017/565"

RULE_GLIDEPATH = Rule(
    rule_id="GLD-CAP",
    description=(
        "Aus der Restlaufzeit bis zum Zieltermin wird über die Stufen des Regelwerks "
        "die höchste zulässige Risikoklasse bestimmt; sie deckelt das Ergebnis der "
        "Profilierung, kann es aber nie anheben."
    ),
    legal_basis=LEGAL_BASIS,
)


def max_class_for_remaining_months(remaining_months: int, policy: GlidepathPolicy) -> int:
    """Ermittelt die höchste zulässige Risikoklasse für eine Restlaufzeit.

    Maßgeblich ist die erste Stufe, deren Untergrenze die Restlaufzeit erreicht.
    Die Stufen sind laut Regelwerksmodell absteigend sortiert und reichen bis
    Restlaufzeit null, sodass stets genau eine Stufe greift.

    Args:
        remaining_months: Restlaufzeit in vollen Monaten.
        policy: Regelwerk des Glide-Path.

    Returns:
        Die höchste zulässige Risikoklasse.

    Raises:
        PolicyError: Wenn die Restlaufzeit negativ ist oder keine Stufe greift.
    """
    if remaining_months < 0:
        raise PolicyError(
            f"Die Restlaufzeit darf nicht negativ sein, ist aber {remaining_months} Monate."
        )
    for stufe in policy.stufen:
        if remaining_months >= stufe.restlaufzeit_monate_ab:
            return stufe.max_risikoklasse
    raise PolicyError(  # pragma: no cover — durch das Modell ausgeschlossen
        f"Keine Glide-Path-Stufe greift für {remaining_months} Monate Restlaufzeit."
    )


def apply_glidepath(
    profile: RiskProfile,
    remaining_months: int,
    policy: GlidepathPolicy,
    recorder: TraceRecorder,
) -> RiskProfile:
    """Deckelt ein Risikoprofil anhand der Restlaufzeit.

    Args:
        profile: Ergebnis der Profilierung ohne Glide-Path.
        remaining_months: Restlaufzeit in vollen Monaten.
        policy: Regelwerk des Glide-Path.
        recorder: Recorder, in den die Deckelung geschrieben wird. Es muss
            derselbe sein, mit dem das Profil erstellt wurde, damit der Trace
            zusammenhängend bleibt.

    Returns:
        Ein neues Risikoprofil. Greift die Deckelung, sinkt die Klasse und die
        bindende Achse wechselt auf :attr:`~candor.domain.result.Axis.GLIDEPATH`.

    Raises:
        PolicyError: Wenn die Restlaufzeit negativ ist.
    """
    deckel = max_class_for_remaining_months(remaining_months, policy)
    greift = deckel < profile.risk_class
    endklasse = min(profile.risk_class, deckel)

    recorder.record(
        RULE_GLIDEPATH,
        {
            "restlaufzeit_monate": remaining_months,
            "klasse_vor_deckelung": profile.risk_class,
            "max_risikoklasse_aus_restlaufzeit": deckel,
        },
        {
            "risikoklasse": endklasse,
            "deckelung_greift": greift,
            "bindende_achse": (Axis.GLIDEPATH.value if greift else profile.binding_axis.value),
        },
        policy.version,
    )

    return replace(
        profile,
        risk_class=endklasse,
        binding_axis=Axis.GLIDEPATH if greift else profile.binding_axis,
        glidepath_class=deckel,
        trace=recorder.events,
        trace_hash=recorder.content_hash(),
    )


def apply_glidepath_for_goal(
    profile: RiskProfile,
    goal: Goal,
    policy: GlidepathPolicy,
    recorder: TraceRecorder,
) -> RiskProfile:
    """Deckelt ein Risikoprofil anhand der Restlaufzeit des Anlageziels.

    Args:
        profile: Ergebnis der Profilierung ohne Glide-Path.
        goal: Anlageziel, aus dem sich die Restlaufzeit ergibt.
        policy: Regelwerk des Glide-Path.
        recorder: Recorder des laufenden Trace.

    Returns:
        Das gedeckelte Risikoprofil.
    """
    return apply_glidepath(profile, goal.horizon_months, policy, recorder)
