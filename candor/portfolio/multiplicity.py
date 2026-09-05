"""Korrektur für multiples Testen beim Vergleich mehrerer Risikoklassen gegen 1/N.

**Warum das nötig ist.** Der Backtest prüft fünf Risikoklassen einzeln gegen
dasselbe 1/N-Referenzportfolio. Jeder Einzeltest hält das Niveau von 5 %, die
**Familie** der fünf Tests hält es nicht: Bei fünf unabhängigen Tests beträgt die
Wahrscheinlichkeit, mindestens einmal fälschlich zu verwerfen,
1 − 0,95⁵ ≈ 22,6 % statt der ausgewiesenen 5 %.

Ohne Korrektur ließe sich aus fünf Tests ein signifikantes Ergebnis
herausgreifen und als Beleg führen — genau der Fehler, den die Prüfung über alle
fünf Klassen (statt nur der Vorgabeklasse) eigentlich vermeiden soll.

**Die drei Verfahren und was sie kontrollieren**

| Verfahren | Kontrolliert | Schärfe |
|---|---|---|
| Bonferroni | familienweise Fehlerrate (FWER) | am strengsten |
| Holm-Bonferroni | FWER | gleichwertig gültig, nie schwächer als Bonferroni |
| Benjamini-Hochberg | Falscherkennungsrate (FDR) | am mildesten |

**Abhängigkeit der Tests — für die Einordnung wesentlich.** Die fünf Tests sind
**nicht unabhängig**: Alle vergleichen gegen dieselbe 1/N-Reihe, und die
MVO-Portfolios der Klassen überschneiden sich in ihren Assetklassen erheblich.
Daraus folgt:

* Die Formel für die FWER unterstellt Unabhängigkeit und ist deshalb eine
  **Obergrenze**. Die tatsächliche familienweise Fehlerrate liegt darunter.
* **Bonferroni und Holm sind bei positiv abhängigen Tests konservativ** — sie
  verwerfen seltener, als das Niveau erlaubte, und verlieren Trennschärfe.
* **Benjamini-Hochberg ist unter positiver Abhängigkeit gültig** (Benjamini &
  Yekutieli, 2001) und damit hier die **angemessenere Untergrenze**: Wenn schon
  BH nicht verwirft, verwirft keines der drei Verfahren.

Alle drei werden ausgewiesen, nicht nur das günstigste. Welches
gewichtet wird, entscheidet der Verfasser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

FAMILY_LEVEL: Final = 0.05


@dataclass(frozen=True, slots=True)
class AdjustedResult:
    """Ergebnis eines Einzeltests nach Korrektur für multiples Testen.

    Attributes:
        label: Kennung des Einzeltests, etwa die Risikoklasse.
        p_raw: Unkorrigierter p-Wert.
        p_bonferroni: Nach Bonferroni angepasster p-Wert, bei 1,0 gekappt.
        p_holm: Nach Holm-Bonferroni angepasster p-Wert, bei 1,0 gekappt.
        p_benjamini_hochberg: Nach Benjamini-Hochberg angepasster p-Wert (q-Wert).
        significant_raw: Ob der unkorrigierte Test verwirft.
        significant_bonferroni: Ob Bonferroni verwirft.
        significant_holm: Ob Holm-Bonferroni verwirft.
        significant_benjamini_hochberg: Ob Benjamini-Hochberg verwirft.
    """

    label: str
    p_raw: float
    p_bonferroni: float
    p_holm: float
    p_benjamini_hochberg: float
    significant_raw: bool
    significant_bonferroni: bool
    significant_holm: bool
    significant_benjamini_hochberg: bool


@dataclass(frozen=True, slots=True)
class MultiplicityReport:
    """Gesamtergebnis der Korrektur über eine Testfamilie.

    Attributes:
        results: Ergebnis je Einzeltest, in der Reihenfolge der Eingabe.
        level: Verwendetes Niveau je Familie.
        family_size: Zahl der Tests in der Familie.
        fwer_uncorrected: Familienweise Fehlerrate ohne Korrektur bei
            unterstellter Unabhängigkeit — eine Obergrenze.
        count_raw: Zahl der ohne Korrektur verworfenen Nullhypothesen.
        count_bonferroni: Zahl nach Bonferroni.
        count_holm: Zahl nach Holm-Bonferroni.
        count_benjamini_hochberg: Zahl nach Benjamini-Hochberg.
    """

    results: tuple[AdjustedResult, ...]
    level: float
    family_size: int
    fwer_uncorrected: float
    count_raw: int
    count_bonferroni: int
    count_holm: int
    count_benjamini_hochberg: int


def familywise_error_rate(family_size: int, level: float = FAMILY_LEVEL) -> float:
    """Berechnet die familienweise Fehlerrate ohne Korrektur.

    Formel: ``1 - (1 - alpha)^m`` für ``m`` Tests zum Niveau ``alpha``.

    Der Wert unterstellt **unabhängige** Tests. Sind die Tests positiv abhängig —
    wie hier durch das gemeinsame 1/N-Referenzportfolio —, liegt die tatsächliche
    Rate darunter. Die Zahl ist deshalb als Obergrenze zu lesen, nicht als
    Schätzung.

    Args:
        family_size: Zahl der Tests in der Familie.
        level: Signifikanzniveau je Einzeltest.

    Returns:
        Die familienweise Fehlerrate als Anteil.

    Raises:
        ValueError: Wenn die Familiengröße kleiner als 1 ist oder das Niveau
            außerhalb von (0, 1) liegt.
    """
    if family_size < 1:
        msg = f"Die Familiengröße muss mindestens 1 betragen, war {family_size}."
        raise ValueError(msg)
    if not 0.0 < level < 1.0:
        msg = f"Das Niveau muss zwischen 0 und 1 liegen, war {level}."
        raise ValueError(msg)
    return 1.0 - (1.0 - level) ** family_size


def adjust(
    p_values: dict[str, float],
    level: float = FAMILY_LEVEL,
) -> MultiplicityReport:
    """Wendet Bonferroni, Holm-Bonferroni und Benjamini-Hochberg an.

    Alle drei Verfahren werden auf dieselbe Familie angewandt und
    nebeneinandergestellt. Es wird **kein** Verfahren ausgewählt — die Auswahl
    wäre eine inhaltliche Entscheidung und gehört in die Auswertung, nicht in den
    Code.

    Args:
        p_values: p-Werte je Einzeltest, Schlüssel ist die Kennung.
        level: Signifikanzniveau der Familie.

    Returns:
        Der vollständige Bericht.

    Raises:
        ValueError: Wenn die Familie leer ist oder ein p-Wert außerhalb von
            [0, 1] liegt.
    """
    if not p_values:
        msg = "Die Testfamilie ist leer."
        raise ValueError(msg)
    for kennung, wert in p_values.items():
        if not 0.0 <= wert <= 1.0:
            msg = f"p-Wert für {kennung} liegt außerhalb von [0, 1]: {wert}."
            raise ValueError(msg)

    anzahl = len(p_values)
    # Aufsteigend sortiert; bei Gleichstand nach Kennung, damit das Ergebnis
    # deterministisch ist (Prinzip P1).
    geordnet = sorted(p_values.items(), key=lambda paar: (paar[1], paar[0]))

    # --- Bonferroni: p · m ---------------------------------------------------
    bonferroni = {kennung: min(1.0, wert * anzahl) for kennung, wert in p_values.items()}

    # --- Holm-Bonferroni: schrittweise absteigende Schwellen ------------------
    # Angepasster p-Wert des i-ten (aufsteigend) Tests: (m − i + 1) · p_(i),
    # danach monoton nach oben durchgereicht.
    holm: dict[str, float] = {}
    bisher = 0.0
    for rang, (kennung, wert) in enumerate(geordnet, start=1):
        kandidat = (anzahl - rang + 1) * wert
        bisher = max(bisher, kandidat)
        holm[kennung] = min(1.0, bisher)

    # --- Benjamini-Hochberg: (m / k) · p_(k), monoton nach unten --------------
    bh: dict[str, float] = {}
    kleinster = 1.0
    for rang in range(anzahl, 0, -1):
        kennung, wert = geordnet[rang - 1]
        kandidat = (anzahl / rang) * wert
        kleinster = min(kleinster, kandidat)
        bh[kennung] = min(1.0, kleinster)

    ergebnisse = tuple(
        AdjustedResult(
            label=kennung,
            p_raw=wert,
            p_bonferroni=bonferroni[kennung],
            p_holm=holm[kennung],
            p_benjamini_hochberg=bh[kennung],
            significant_raw=wert < level,
            significant_bonferroni=bonferroni[kennung] < level,
            significant_holm=holm[kennung] < level,
            significant_benjamini_hochberg=bh[kennung] < level,
        )
        for kennung, wert in p_values.items()
    )

    return MultiplicityReport(
        results=ergebnisse,
        level=level,
        family_size=anzahl,
        fwer_uncorrected=familywise_error_rate(anzahl, level),
        count_raw=sum(1 for e in ergebnisse if e.significant_raw),
        count_bonferroni=sum(1 for e in ergebnisse if e.significant_bonferroni),
        count_holm=sum(1 for e in ergebnisse if e.significant_holm),
        count_benjamini_hochberg=sum(1 for e in ergebnisse if e.significant_benjamini_hochberg),
    )
