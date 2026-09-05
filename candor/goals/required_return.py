"""Erforderliche Rendite eines Anlageziels.

Gesucht ist die Rendite, bei der Startkapital und Sparplan zum Zieltermin genau
den Zielbetrag erreichen. Grundlage ist die **nachschüssige Rentenendwertformel**
auf Monatsbasis.

Formel::

    FV(i) = PV · (1+i)^n + PMT · ((1+i)^n − 1) / i        für i ≠ 0
    FV(0) = PV + PMT · n                                   für i = 0

mit ``PV`` Startkapital, ``PMT`` monatliche Sparrate (jeweils am Monatsende),
``n`` Zahl der Monate und ``i`` Monatsrendite. Gesucht ist die Nullstelle von
``f(i) = FV(i) − Zielbetrag``.

Warum nachschüssig
------------------
Eine Sparrate wird typischerweise am Monatsende abgebucht und angelegt. Die
vorschüssige Variante unterstellt die Einzahlung am Monatsanfang und damit einen
Monat zusätzliche Verzinsung je Rate. Über zwanzig Jahre und 240 Raten
überschätzt sie den Endwert spürbar. Die vorsichtigere Annahme ist hier die
richtige — sie führt zu einer *höheren* erforderlichen Rendite und damit eher zur
Konfliktdiagnose als zur Empfehlung.

Lösungsweg
----------
``f`` ist auf ``i > −1`` streng monoton steigend, sofern ``PV > 0`` oder
``PMT > 0``. Es gibt daher höchstens eine Nullstelle. Analytisch gelöst wird,
wo eine geschlossene Form existiert:

* ``PMT = 0`` und ``PV > 0``  ⇒  ``i = (FV/PV)^(1/n) − 1``
* ``n = 1`` und ``PV > 0``    ⇒  ``i = (FV − PMT)/PV − 1``
* ``FV = PV + PMT·n``          ⇒  ``i = 0`` exakt

Sonst kommt das Brent-Verfahren zum Einsatz — Bisektion mit inverser
quadratischer Interpolation, garantiert konvergent bei Vorzeichenwechsel im
Intervall und ohne Ableitung.

Nominal und real
----------------
Beide Größen werden **getrennt** ausgewiesen und nie vermischt:

* nominale Sicht — der Zielbetrag steht so am Zieltermin auf dem Konto,
* Kaufkraftsicht — der Zielbetrag wird mit der Inflationsannahme hochgerechnet,
  sodass er am Zieltermin dieselbe Kaufkraft hat wie heute.

Die Inflationsannahme steht in ``policies/goals_v1.yaml`` und ist ausdrücklich
eine Annahme, keine Prognose.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Final

from scipy.optimize import brentq

from candor.domain.goal import Goal
from candor.domain.policy import GoalPolicy
from candor.domain.result import (
    RequiredReturn,
    ReturnCase,
    ReturnSolution,
    SolutionMethod,
    deflate,
)
from candor.trace.events import Rule
from candor.trace.recorder import TraceRecorder

#: Monate je Jahr.
MONTHS_PER_YEAR: Final = 12

#: Genauigkeit der ausgewiesenen Renditen: sechs Nachkommastellen entsprechen
#: einem Zehntausendstel Prozentpunkt und liegen damit weit unter jeder
#: wirtschaftlich bedeutsamen Schwelle.
RATE_QUANT: Final = Decimal("0.000001")

#: Genauigkeit der ausgewiesenen Geldbeträge.
EURO_QUANT: Final = Decimal("0.01")

#: Untere Grenze der Nullstellensuche: −99 % je Monat. Darunter ist die
#: Fragestellung wirtschaftlich sinnlos; die Grenze wird als Schranke
#: ausgewiesen statt als Ergebnis behauptet.
MIN_MONTHLY_RATE: Final = -0.99

#: Obere Grenze der Nullstellensuche: +100 % je Monat, also eine Verdopplung
#: Monat für Monat. Wer diese Rendite bräuchte, hat kein Renditeproblem,
#: sondern ein Zielproblem.
MAX_MONTHLY_RATE: Final = 1.0

#: Genauigkeit der numerischen Nullstellensuche.
BRENT_TOLERANCE: Final = 1e-14

#: Horizont, unterhalb dessen die Annualisierung eine Hochrechnung ist.
SHORT_HORIZON_MONTHS: Final = MONTHS_PER_YEAR

RULE_REQUIRED_RETURN = Rule(
    rule_id="GOAL-RET",
    description=(
        "Aus Startkapital, Sparplan, Horizont und Zielbetrag wird über die nachschüssige "
        "Rentenendwertformel die erforderliche Rendite bestimmt; nominale und reale "
        "Größe werden getrennt ausgewiesen."
    ),
    legal_basis="Art. 54 Abs. 2 Buchst. a DelVO (EU) 2017/565",
)


def future_value(
    initial_capital: Decimal,
    monthly_contribution: Decimal,
    monthly_rate: Decimal,
    months: int,
) -> Decimal:
    """Berechnet den Endwert eines Sparplans mit nachschüssigen Raten.

    Args:
        initial_capital: Startkapital ``PV``.
        monthly_contribution: Monatliche Sparrate ``PMT``, jeweils am Monatsende.
        monthly_rate: Monatsrendite ``i``.
        months: Zahl der Monate ``n``.

    Returns:
        Der Endwert ``FV`` zum Zieltermin.

    Raises:
        ValueError: Wenn die Monatsrendite bei oder unter −100 % liegt oder die
            Monatszahl negativ ist. Beides ist keine sinnvolle Eingabe und wird
            nicht durch einen Ersatzwert überdeckt.
    """
    if monthly_rate <= Decimal(-1):
        msg = f"Die Monatsrendite muss über −100 % liegen, ist aber {monthly_rate}."
        raise ValueError(msg)
    if months < 0:
        msg = f"Die Monatszahl darf nicht negativ sein, ist aber {months}."
        raise ValueError(msg)

    if monthly_rate == 0:
        return initial_capital + monthly_contribution * Decimal(months)

    wachstum = (Decimal(1) + monthly_rate) ** months
    return (
        initial_capital * wachstum + monthly_contribution * (wachstum - Decimal(1)) / monthly_rate
    )


def annualize(monthly_rate: Decimal) -> Decimal:
    """Rechnet eine Monatsrendite geometrisch auf ein Jahr hoch.

    ``(1 + i_jährlich) = (1 + i_monatlich)^12``. Die naheliegende Multiplikation
    mit zwölf wäre falsch: Sie unterschlägt den Zinseszins innerhalb des Jahres.

    Args:
        monthly_rate: Monatsrendite.

    Returns:
        Die annualisierte Jahresrendite, auf sechs Stellen gerundet.
    """
    faktor = (Decimal(1) + monthly_rate) ** MONTHS_PER_YEAR
    return (faktor - Decimal(1)).quantize(RATE_QUANT, rounding=ROUND_HALF_UP)


def deannualize(annual_rate: Decimal) -> Decimal:
    """Rechnet eine Jahresrendite geometrisch auf einen Monat herunter.

    Args:
        annual_rate: Jahresrendite.

    Returns:
        Die entsprechende Monatsrendite.
    """
    faktor = (Decimal(1) + annual_rate) ** (Decimal(1) / Decimal(MONTHS_PER_YEAR))
    return faktor - Decimal(1)


def _classify(
    target: Decimal, initial_capital: Decimal, value_without_return: Decimal
) -> ReturnCase:
    """Ordnet einen Zielbetrag fachlich ein.

    Args:
        target: Zielbetrag.
        initial_capital: Startkapital.
        value_without_return: Endwert ohne jede Rendite.

    Returns:
        Die fachliche Einordnung.
    """
    if initial_capital >= target:
        return ReturnCase.BEREITS_ERREICHT
    if value_without_return >= target:
        return ReturnCase.OHNE_RENDITE_ERREICHBAR
    return ReturnCase.ERREICHBAR


def solve_monthly_rate(
    target_amount: Decimal,
    initial_capital: Decimal,
    monthly_contribution: Decimal,
    months: int,
) -> tuple[Decimal, ReturnCase, SolutionMethod]:
    """Bestimmt die erforderliche Monatsrendite für einen Zielbetrag.

    Args:
        target_amount: Angestrebter Endwert ``FV``.
        initial_capital: Startkapital ``PV``.
        monthly_contribution: Monatliche Sparrate ``PMT``.
        months: Zahl der Monate ``n``.

    Returns:
        Tripel aus Monatsrendite, fachlicher Einordnung und Lösungsverfahren.
        Die Monatsrendite ist stets eine Zahl — in den Randfällen die
        ausgewiesene Suchbereichsgrenze, nie ``NaN``.

    Raises:
        ValueError: Wenn der Horizont unter einem Monat liegt oder weder
            Startkapital noch Sparrate vorhanden sind.
    """
    if months < 1:
        msg = f"Der Horizont muss mindestens einen Monat betragen, ist aber {months}."
        raise ValueError(msg)
    if initial_capital == 0 and monthly_contribution == 0:
        msg = "Ohne Startkapital und ohne Sparrate ist kein Zielbetrag erreichbar."
        raise ValueError(msg)

    ohne_rendite = initial_capital + monthly_contribution * Decimal(months)
    fall = _classify(target_amount, initial_capital, ohne_rendite)

    # Fall 1: Der Zielbetrag entspricht exakt der Summe der Einzahlungen.
    if target_amount == ohne_rendite:
        return Decimal(0), ReturnCase.OHNE_RENDITE_ERREICHBAR, SolutionMethod.EXAKT_NULL

    # Fall 2: Keine Sparrate — reine Aufzinsung des Startkapitals.
    if monthly_contribution == 0:
        faktor = target_amount / initial_capital
        rate = faktor ** (Decimal(1) / Decimal(months)) - Decimal(1)
        return rate.quantize(RATE_QUANT, rounding=ROUND_HALF_UP), fall, SolutionMethod.GESCHLOSSEN

    # Fall 3: Genau ein Monat — die Formel ist linear in i.
    if months == 1 and initial_capital > 0:
        rate = (target_amount - monthly_contribution) / initial_capital - Decimal(1)
        return rate.quantize(RATE_QUANT, rounding=ROUND_HALF_UP), fall, SolutionMethod.GESCHLOSSEN

    # Fall 4: Numerisch. f ist streng monoton steigend, also genügt der
    # Vorzeichenwechsel an den Intervallgrenzen zur Existenzprüfung.
    ziel = float(target_amount)
    pv = float(initial_capital)
    pmt = float(monthly_contribution)

    def abweichung(rate: float) -> float:
        """Differenz zwischen Endwert und Zielbetrag.

        Args:
            rate: Monatsrendite.

        Returns:
            Der Abstand zum Zielbetrag.
        """
        wachstum = (1.0 + rate) ** months
        return pv * wachstum + pmt * (wachstum - 1.0) / rate - ziel

    if abweichung(MIN_MONTHLY_RATE) > 0:
        return (
            Decimal(str(MIN_MONTHLY_RATE)),
            ReturnCase.UEBERERFUELLT,
            SolutionMethod.SCHRANKE,
        )
    if abweichung(MAX_MONTHLY_RATE) < 0:
        return (
            Decimal(str(MAX_MONTHLY_RATE)),
            ReturnCase.NICHT_ERREICHBAR,
            SolutionMethod.SCHRANKE,
        )

    wurzel = brentq(abweichung, MIN_MONTHLY_RATE, MAX_MONTHLY_RATE, xtol=BRENT_TOLERANCE)
    rate = Decimal(str(wurzel)).quantize(RATE_QUANT, rounding=ROUND_HALF_UP)
    return rate, fall, SolutionMethod.BRENT


def _solution(
    target_amount: Decimal,
    initial_capital: Decimal,
    monthly_contribution: Decimal,
    months: int,
) -> ReturnSolution:
    """Bildet eine vollständige Lösung für einen Zielbetrag.

    Args:
        target_amount: Angestrebter Endwert.
        initial_capital: Startkapital.
        monthly_contribution: Monatliche Sparrate.
        months: Zahl der Monate.

    Returns:
        Die Lösung mit Monats- und Jahresrendite.
    """
    rate, fall, verfahren = solve_monthly_rate(
        target_amount, initial_capital, monthly_contribution, months
    )
    return ReturnSolution(
        target_amount=target_amount,
        monthly_rate=rate,
        annual_rate=annualize(rate),
        case=fall,
        method=verfahren,
    )


def inflate(amount: Decimal, annual_inflation: Decimal, months: int) -> Decimal:
    """Rechnet einen Betrag mit der Inflationsannahme auf den Zieltermin hoch.

    Args:
        amount: Betrag in heutiger Kaufkraft.
        annual_inflation: Jährliche Inflationsannahme.
        months: Zahl der Monate bis zum Zieltermin.

    Returns:
        Der nominale Betrag, der am Zieltermin dieselbe Kaufkraft hat.
    """
    jahre = Decimal(months) / Decimal(MONTHS_PER_YEAR)
    faktor = (Decimal(1) + annual_inflation) ** jahre
    return (amount * faktor).quantize(EURO_QUANT, rounding=ROUND_HALF_UP)


def required_return(
    goal: Goal,
    policy: GoalPolicy,
    inflation_annual: Decimal | None = None,
    recorder: TraceRecorder | None = None,
) -> RequiredReturn:
    """Ermittelt die erforderliche Rendite eines Anlageziels.

    Args:
        goal: Geprüftes Anlageziel.
        policy: Zielpolicy, aus der die Inflationsannahme stammt.
        inflation_annual: Abweichende Inflationsannahme. Ohne Angabe gilt der
            Wert aus dem Regelwerk.
        recorder: Optionaler Recorder für den Trace.

    Returns:
        Die erforderliche Rendite, nominal und real getrennt ausgewiesen.
    """
    inflation = (
        inflation_annual if inflation_annual is not None else policy.inflationsannahme_jaehrlich
    )
    monate = goal.horizon_months
    ohne_rendite = goal.initial_capital + goal.monthly_contribution * Decimal(monate)

    nominal = _solution(goal.target_amount, goal.initial_capital, goal.monthly_contribution, monate)
    kaufkraftziel = inflate(goal.target_amount, inflation, monate)
    kaufkraft = _solution(kaufkraftziel, goal.initial_capital, goal.monthly_contribution, monate)

    hinweise: list[str] = []
    if monate < SHORT_HORIZON_MONTHS:
        hinweise.append(
            f"Der Horizont beträgt nur {monate} Monate. Die auf ein Jahr hochgerechnete "
            "Rendite ist eine Extrapolation über den Anlagezeitraum hinaus und als "
            "Vergleichsgröße, nicht als Erwartung zu lesen."
        )
    if goal.monthly_contribution == 0:
        hinweise.append(
            "Ohne Sparrate hängt die Zielerreichung allein an der Verzinsung des "
            "Startkapitals; ein Ausgleich durch höhere Einzahlungen ist nicht möglich."
        )
    if nominal.case is ReturnCase.BEREITS_ERREICHT:
        hinweise.append(
            "Das Startkapital erreicht den Zielbetrag bereits. Die erforderliche Rendite "
            "ist negativ; die Frage ist hier nicht die Rendite, sondern der Kapitalerhalt."
        )
    if nominal.case is ReturnCase.NICHT_ERREICHBAR:
        hinweise.append(
            "Der Zielbetrag ist auch bei einer Verdopplung Monat für Monat nicht "
            "erreichbar. Ausgewiesen ist die Suchbereichsgrenze, kein Ergebnis."
        )
    if nominal.annual_rate <= 0 < kaufkraft.annual_rate:
        hinweise.append(
            "Nominal genügen die Einzahlungen, kaufkraftbereinigt nicht. Der Unterschied "
            "ist die Inflationsannahme."
        )

    ergebnis = RequiredReturn(
        horizon_months=monate,
        initial_capital=goal.initial_capital,
        monthly_contribution=goal.monthly_contribution,
        inflation_annual=inflation,
        value_without_return=ohne_rendite.quantize(EURO_QUANT, rounding=ROUND_HALF_UP),
        nominal=nominal,
        purchasing_power=kaufkraft,
        warnings=tuple(hinweise),
        policy_version=policy.version,
    )

    if recorder is not None:
        recorder.record(
            RULE_REQUIRED_RETURN,
            {
                "zielbetrag": goal.target_amount,
                "startkapital": goal.initial_capital,
                "sparrate_monatlich": goal.monthly_contribution,
                "horizont_monate": monate,
                "inflationsannahme_jaehrlich": inflation,
                "rentenformel": "nachschüssig",
            },
            {
                "erforderliche_rendite_nominal_pa": nominal.annual_rate,
                "erforderliche_rendite_real_pa": deflate(nominal.annual_rate, inflation),
                "zielbetrag_kaufkrafterhaltend": kaufkraftziel,
                "erforderliche_rendite_kaufkraft_pa": kaufkraft.annual_rate,
                "fall": nominal.case.value,
                "verfahren": nominal.method.value,
                "hinweise": list(hinweise),
            },
            policy.version,
        )

    return ergebnis
