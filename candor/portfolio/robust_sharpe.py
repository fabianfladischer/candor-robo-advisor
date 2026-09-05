"""Robuster Sharpe-Vergleichstest nach Ledoit & Wolf (2008).

**Warum dieser Test neben dem vorhandenen steht.** Der in
``benchmark.sharpe_difference_test`` umgesetzte Test von Jobson & Korkie (1981)
in der Fassung von Memmel (2003) unterstellt normalverteilte, unabhängige
Renditen. Beides ist in den Daten dieses Projekts geprüft worden:

* **Normalverteilung:** in **15 von 15** Strategiereihen abgelehnt
  (Jarque-Bera, 5 %). Exzess-Wölbung 0,77 bis 2,08, Schiefe −0,65 bis −0,37.
* **Volatilitätscluster:** in **15 von 15** Reihen nachgewiesen
  (Ljung-Box über die quadrierten Renditen, 5 %).
* **Lineare Autokorrelation:** in **keiner** Reihe signifikant.

Damit liegt ein innerer Widerspruch vor: Der Block-Bootstrap der Simulation
wird mit Fat Tails und Volatilitätsclustern begründet, der
Signifikanztest der Portfolioevaluation unterstellt deren Abwesenheit — auf
denselben Daten. Ledoit & Wolf (2008) lösen genau diesen Fall.

**Das Verfahren.** Studentisierter Block-Bootstrap:

1. Auf den Originaldaten die Differenz der Sharpe Ratios ``Δ`` und ihren
   HAC-Standardfehler ``s`` bestimmen; ``d = Δ / s``.
2. ``B`` Mal Blöcke **paarweise** aus beiden Reihen ziehen — die gemeinsame
   Ziehung erhält die Korrelation zwischen den Strategien, die den Test
   überhaupt erst nötig macht.
3. Je Ziehung ``d*_b = (Δ*_b − Δ) / s*_b``. Die Zentrierung an ``Δ`` erzeugt die
   Verteilung unter der Nullhypothese.
4. Zweiseitiger p-Wert: Anteil der ``|d*_b| ≥ |d|``.

**Abweichung von der Vorlage, ausdrücklich benannt.** Ledoit & Wolf verwenden
für den Standardfehler einen vorgeweißten QS-Kern mit automatischer Bandbreite.
Hier steht ein Bartlett-Kern mit der Bandbreitenregel von Newey & West,
``⌊4·(T/100)^(2/9)⌋`` — bei 156 Monaten sind das 4 Verzögerungen. Die
Vereinfachung betrifft nur die Studentisierung; die Robustheit gegen Fat Tails
und Cluster trägt der Block-Bootstrap, nicht der Kern. Wer den Unterschied
beziffern will, müsste beide Kerne gegenrechnen – das ist nicht geschehen.

**Determinismus (Prinzip P1).** Der Seed wird ausdrücklich übergeben und in das
Ergebnis geschrieben. Gleicher Seed, gleiche Daten, gleiche Blocklänge ⇒
bitidentischer p-Wert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from candor.engine.simulate import BlockBootstrapSampler
from candor.portfolio.benchmark import PerformanceError

#: Vorgabe der Zahl der Bootstrap-Ziehungen. Ungerade Zahl, damit der Anteil
#: (1 + Treffer) / (1 + B) keine Bindung an eine runde Schwelle erzeugt.
DEFAULT_ITERATIONS: Final = 4999

#: Blocklänge in Monaten. Identisch mit der Simulation, damit
#: Simulation und Test dieselbe Annahme über die Erinnerungslänge treffen.
DEFAULT_BLOCK_LENGTH: Final = 12

#: Mindestzahl an Monaten, unterhalb derer der Test nicht sinnvoll ist.
MIN_PERIODS: Final = 24

_MOMENT_COUNT: Final = 4


@dataclass(frozen=True, slots=True)
class RobustSharpeComparison:
    """Ergebnis des robusten Sharpe-Vergleichs.

    Attributes:
        sharpe_a: Annualisierte Sharpe Ratio der ersten Strategie.
        sharpe_b: Annualisierte Sharpe Ratio der zweiten Strategie.
        difference: Differenz ``sharpe_a − sharpe_b``.
        standard_error: HAC-Standardfehler der Differenz, annualisiert.
        studentized_statistic: Studentisierte Teststatistik der Originaldaten.
        p_value: Zweiseitiger Bootstrap-p-Wert.
        iterations: Zahl der Bootstrap-Ziehungen.
        block_length: Verwendete Blocklänge in Monaten.
        bandwidth: Zahl der Verzögerungen im Bartlett-Kern.
        seed: Verwendeter Seed.
        months: Zahl der ausgewerteten Monate.
    """

    sharpe_a: float
    sharpe_b: float
    difference: float
    standard_error: float
    studentized_statistic: float
    p_value: float
    iterations: int
    block_length: int
    bandwidth: int
    seed: int
    months: int

    def is_significant(self, level: float) -> bool:
        """Prüft, ob der Unterschied auf dem gegebenen Niveau gesichert ist.

        Args:
            level: Signifikanzniveau, etwa ``0.05``.

        Returns:
            ``True``, wenn der p-Wert das Niveau unterschreitet.
        """
        return self.p_value < level

    @property
    def label(self) -> str:
        """Bezeichnung des Verfahrens für Tabellen und Protokolle."""
        return (
            f"Ledoit-Wolf-Bootstrap (2008), {self.iterations} Ziehungen, "
            f"Blocklänge {self.block_length} Monate, Bartlett-Kern mit "
            f"{self.bandwidth} Verzögerungen, Seed {self.seed}"
        )


def _bandwidth(months: int) -> int:
    """Bestimmt die Bandbreite des Bartlett-Kerns nach Newey & West.

    Regel: ``⌊4 · (T / 100)^(2/9)⌋``, mindestens 1.

    Args:
        months: Zahl der Beobachtungen.

    Returns:
        Zahl der zu berücksichtigenden Verzögerungen.
    """
    return max(1, int(4.0 * (months / 100.0) ** (2.0 / 9.0)))


def _sharpe_and_gradient(
    excess_a: np.ndarray,
    excess_b: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Berechnet monatliche Sharpe Ratios und den Gradienten der Differenz.

    Die Sharpe Ratio ist ``mu / sqrt(g - mu^2)`` mit ``g = E[r^2]``. Der
    Gradient bezieht sich auf den Momentvektor ``(mu_a, mu_b, g_a, g_b)`` und
    wird für den Standardfehler nach der Delta-Methode gebraucht.

    Args:
        excess_a: Überschussrenditen der ersten Strategie.
        excess_b: Überschussrenditen der zweiten Strategie.

    Returns:
        Monatliche Sharpe Ratio a, monatliche Sharpe Ratio b, Gradient.

    Raises:
        PerformanceError: Wenn eine Reihe keine Streuung aufweist.
    """
    mu_a, mu_b = float(excess_a.mean()), float(excess_b.mean())
    gamma_a, gamma_b = float((excess_a**2).mean()), float((excess_b**2).mean())

    var_a, var_b = gamma_a - mu_a**2, gamma_b - mu_b**2
    if var_a <= 0.0 or var_b <= 0.0:
        msg = "Eine der Reihen hat keine Streuung; eine Sharpe Ratio ist nicht definiert."
        raise PerformanceError(msg)

    sd_a, sd_b = np.sqrt(var_a), np.sqrt(var_b)
    gradient = np.array(
        [
            gamma_a / (var_a * sd_a),
            -gamma_b / (var_b * sd_b),
            -mu_a / (2.0 * var_a * sd_a),
            mu_b / (2.0 * var_b * sd_b),
        ]
    )
    return mu_a / sd_a, mu_b / sd_b, gradient


def _hac_standard_error(
    excess_a: np.ndarray,
    excess_b: np.ndarray,
    gradient: np.ndarray,
    bandwidth: int,
) -> float:
    """Schätzt den Standardfehler der Sharpe-Differenz mit Bartlett-Kern.

    Args:
        excess_a: Überschussrenditen der ersten Strategie.
        excess_b: Überschussrenditen der zweiten Strategie.
        gradient: Gradient der Differenz nach dem Momentvektor.
        bandwidth: Zahl der Verzögerungen.

    Returns:
        Der Standardfehler der monatlichen Sharpe-Differenz.
    """
    momente = np.column_stack([excess_a, excess_b, excess_a**2, excess_b**2])
    zentriert = momente - momente.mean(axis=0)
    anzahl = zentriert.shape[0]

    psi = (zentriert.T @ zentriert) / anzahl
    for verzoegerung in range(1, min(bandwidth, anzahl - 1) + 1):
        gewicht = 1.0 - verzoegerung / (bandwidth + 1.0)
        gamma = (zentriert[verzoegerung:].T @ zentriert[:-verzoegerung]) / anzahl
        psi += gewicht * (gamma + gamma.T)

    varianz = float(gradient @ psi @ gradient) / anzahl
    return float(np.sqrt(max(varianz, np.finfo(float).tiny)))


def sharpe_difference_bootstrap(  # noqa: PLR0913 — Daten, Konvention und drei
    # Verfahrensparameter sind sachlich getrennt; ein Sammelobjekt verschleierte,
    # dass Seed, Ziehungszahl und Blocklänge das Ergebnis bestimmen (Prinzip P1).
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    risk_free_monthly: float,
    months_per_year: int,
    *,
    seed: int,
    iterations: int = DEFAULT_ITERATIONS,
    block_length: int = DEFAULT_BLOCK_LENGTH,
) -> RobustSharpeComparison:
    """Prüft die Gleichheit zweier Sharpe Ratios ohne Verteilungsannahme.

    Umsetzung des studentisierten Block-Bootstraps nach Ledoit & Wolf (2008).
    Anders als der Test von Jobson & Korkie in Memmels Fassung setzt er weder
    Normalverteilung noch Unabhängigkeit der Renditen voraus.

    Die Blöcke werden **paarweise** gezogen: Beide Reihen erhalten in jeder
    Ziehung dieselben Zeitindizes. Nur so bleibt die Korrelation zwischen den
    Strategien erhalten, die den gemeinsamen Test überhaupt erst erfordert.

    Args:
        returns_a: Monatsrenditen der ersten Strategie.
        returns_b: Monatsrenditen der zweiten Strategie.
        risk_free_monthly: Monatlicher risikofreier Zins.
        months_per_year: Monate je Jahr für die Annualisierung.
        seed: Seed des Zufallsgenerators. Pflichtangabe nach Prinzip P1.
        iterations: Zahl der Bootstrap-Ziehungen.
        block_length: Blocklänge in Monaten.

    Returns:
        Das Testergebnis.

    Raises:
        PerformanceError: Wenn die Reihen verschieden lang, zu kurz oder ohne
            Streuung sind oder wenn die Parameter unzulässig sind.
    """
    if returns_a.shape != returns_b.shape:
        msg = f"Die Reihen haben verschiedene Länge: {returns_a.shape} und {returns_b.shape}."
        raise PerformanceError(msg)
    if returns_a.ndim != 1:
        msg = f"Die Reihen müssen eindimensional sein, haben aber {returns_a.ndim} Achsen."
        raise PerformanceError(msg)
    if returns_a.size < MIN_PERIODS:
        msg = f"Nur {returns_a.size} Monate; mindestens {MIN_PERIODS} sind nötig."
        raise PerformanceError(msg)
    if iterations < 1:
        msg = f"Die Zahl der Ziehungen muss positiv sein, war {iterations}."
        raise PerformanceError(msg)
    if not 1 <= block_length <= returns_a.size:
        msg = f"Die Blocklänge muss zwischen 1 und {returns_a.size} liegen, war {block_length}."
        raise PerformanceError(msg)

    ueber_a = np.asarray(returns_a, dtype=float) - risk_free_monthly
    ueber_b = np.asarray(returns_b, dtype=float) - risk_free_monthly
    anzahl = int(ueber_a.size)
    wurzel = float(np.sqrt(months_per_year))
    bandbreite = _bandwidth(anzahl)

    sharpe_a, sharpe_b, gradient = _sharpe_and_gradient(ueber_a, ueber_b)
    differenz = sharpe_a - sharpe_b
    standardfehler = _hac_standard_error(ueber_a, ueber_b, gradient, bandbreite)
    statistik = differenz / standardfehler

    # Paarweise Blockziehung über den Simulations-Sampler: Eine Zeile der Historie
    # ist ein Monat mit BEIDEN Strategierenditen, deshalb bleibt ihre
    # Korrelation erhalten.
    historie = np.column_stack([ueber_a, ueber_b])
    sampler = BlockBootstrapSampler(historie, block_length, circular=True)
    generator = np.random.default_rng(seed)
    ziehungen = sampler.sample(generator, paths=iterations, horizon=anzahl)

    treffer = 0
    betrag = abs(statistik)
    for lauf in range(iterations):
        stichprobe_a = ziehungen[lauf, :, 0]
        stichprobe_b = ziehungen[lauf, :, 1]
        try:
            s_a, s_b, grad = _sharpe_and_gradient(stichprobe_a, stichprobe_b)
        except PerformanceError:
            # Entartete Ziehung ohne Streuung: zählt nicht als Treffer, wird
            # aber auch nicht verworfen — sonst hinge der Nenner am Zufall.
            continue
        fehler = _hac_standard_error(stichprobe_a, stichprobe_b, grad, bandbreite)
        if abs((s_a - s_b - differenz) / fehler) >= betrag:
            treffer += 1

    p_wert = (1.0 + treffer) / (1.0 + iterations)

    return RobustSharpeComparison(
        sharpe_a=sharpe_a * wurzel,
        sharpe_b=sharpe_b * wurzel,
        difference=differenz * wurzel,
        standard_error=standardfehler * wurzel,
        studentized_statistic=statistik,
        p_value=p_wert,
        iterations=iterations,
        block_length=block_length,
        bandwidth=bandbreite,
        seed=seed,
        months=anzahl,
    )
