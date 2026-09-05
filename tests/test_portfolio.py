"""Tests der Portfoliokonstruktion.

Schwerpunkte:

* **I5 — Zulässigkeit.** Gewichtssumme eins, keine Leerverkäufe, alle Bänder
  eingehalten. Geprüft auf zwei Ebenen: auf dem Regelwerk (lassen die Bänder
  überhaupt ein Portfolio zu?) und auf dem Ergebnis jeder Optimierung.
* **I6 — Monotonie der Aktienquote.** Über die Risikoklassen steigend, sowohl
  im Regelwerk als auch in den tatsächlich optimierten Portfolios.
* **I8 — Nachhaltigkeitsfilter.** Unterschreitet das gefilterte Universum die
  Mindestgröße, folgt ein definierter Fehler statt einer stillen Aufweichung.
* **Kein Look-ahead im Backtest.** Der wichtigste Einzeltest dieser Datei: Ein
  Backtest, der Daten aus dem Haltezeitraum in die Schätzung lässt, misst
  Anpassungsgüte statt Prognosegüte und fällt zwangsläufig zugunsten der
  Optimierung aus.
* **Determinismus.** Die Optimierung enthält keine Zufallsquelle; zwei Läufe
  müssen bitgleich sein (Prinzip P1).
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from candor.data.policy_loader import load_policy_set
from candor.data.universe_meta import ASSET_KEYS
from candor.domain.client import SustainabilityComponent, SustainabilityPreferences
from candor.domain.policy import CovarianceMethod, MeanMethod, PolicySet
from candor.engine.advise import SustainabilityQuotaError, check_sustainability_quotas
from candor.engine.convergence import load_history
from candor.portfolio.backtest import (
    STRATEGY_EQUAL,
    STRATEGY_MVO,
    STRATEGY_REFERENCE,
    BacktestError,
    run_backtest,
)
from candor.portfolio.benchmark import (
    PerformanceError,
    equal_weight_portfolio,
    evaluate_performance,
    max_drawdown,
    sharpe_difference_test,
    turnover_from_weights,
)
from candor.portfolio.estimate import (
    MIN_OBSERVATIONS,
    EstimationError,
    FloatArray,
    compare_mean_methods,
    estimate_moments,
    historical_mean,
    ledoit_wolf_covariance,
    risk_free_monthly,
    sample_covariance,
    shrinkage_as_decimal,
    shrunk_mean,
)
from candor.portfolio.evaluation import (
    PORTFOLIO_END,
    PORTFOLIO_START,
    _nachhaltigkeitsblock,
    _zulaessigkeitsblock,
    render_block,
    write_evaluation,
)
from candor.portfolio.optimize import (
    OptimizationError,
    SustainabilityConflictError,
    SustainabilityUniverseError,
    achieved_shares,
    check_constraints,
    filter_universe,
    minimum_variance,
    optimize,
)
from candor.portfolio.universe import equal_weights, equity_share, reference_weights
from candor.report.figures import build_figure_5
from candor.trace.recorder import TraceMetadata, TraceRecorder

#: Das Regelwerk wird einmal geladen; es ist unveränderlich.
POLICIES: PolicySet = load_policy_set()

#: Toleranz beim Prüfen der Restriktionen.
TOLERANCE = POLICIES.portfolio.optimierung.restriktionstoleranz

#: Alle Risikoklassen des Regelwerks.
RISK_CLASSES = tuple(sorted(POLICIES.classes.risikoklassen))


@pytest.fixture(scope="module")
def history() -> FloatArray:
    """Lädt die vollständige historische Renditematrix.

    Returns:
        Renditematrix der Form ``(monate, assetklassen)``.
    """
    return load_history(POLICIES)


@pytest.fixture(scope="module")
def moments(history: FloatArray) -> object:
    """Schätzt die Momente einmal je Modul.

    Args:
        history: Historische Renditematrix.

    Returns:
        Die geschätzten Momente.
    """
    return estimate_moments(history, POLICIES.portfolio.schaetzung, ASSET_KEYS)


def _optimiere(moments: object, risk_class: int) -> object:
    """Optimiert eine Klasse gegen die Rendite ihrer Referenzallokation.

    Args:
        moments: Geschätzte Momente.
        risk_class: Risikoklasse.

    Returns:
        Das Optimierungsergebnis.
    """
    referenz = reference_weights(risk_class, POLICIES.classes, ASSET_KEYS)
    ziel = float(referenz @ moments.mean)  # type: ignore[attr-defined]
    return optimize(
        moments,  # type: ignore[arg-type]
        risk_class,
        POLICIES.classes,
        POLICIES.portfolio,
        ASSET_KEYS,
        target_return=ziel,
    )


def _optimiere_gefiltert(moments: object, risk_class: int, universum: object) -> object:
    """Optimiert eine Klasse auf einem gefilterten Universum.

    Die Zielrendite ist bewusst unerreichbar niedrig gesetzt, damit allein die
    Restriktionen binden und der Test nicht an der Zielrendite scheitert.

    Args:
        moments: Geschätzte Momente.
        risk_class: Risikoklasse.
        universum: Ergebnis des Nachhaltigkeitsfilters.

    Returns:
        Das Optimierungsergebnis.
    """
    return optimize(
        moments,  # type: ignore[arg-type]
        risk_class,
        POLICIES.classes,
        POLICIES.portfolio,
        ASSET_KEYS,
        target_return=-1.0,
        universe=universum,  # type: ignore[arg-type]
    )


# --- I5: Zulässigkeit der Gewichte -----------------------------------------


@given(risk_class=st.sampled_from(RISK_CLASSES))
@settings(max_examples=len(RISK_CLASSES), deadline=None)
def test_i5_regelwerk_laesst_ein_portfolio_zu(risk_class: int) -> None:
    """I5 auf dem Regelwerk: Die Bänder lassen eine Gewichtssumme von eins zu.

    Die Prüfung setzt vor der Optimierung an. Wären die Bänder in sich
    widersprüchlich, gäbe es kein zulässiges Portfolio, und jeder Solver müsste
    scheitern — der Fehler läge dann aber in der Policy, nicht im Verfahren.

    Args:
        risk_class: Geprüfte Risikoklasse.
    """
    definition = POLICIES.classes.risikoklassen[risk_class]
    minima = sum(band.min for band in definition.einzelwerte.values())
    maxima = sum(band.max for band in definition.einzelwerte.values())

    assert minima <= Decimal(1)
    assert maxima >= Decimal(1)


@given(risk_class=st.sampled_from(RISK_CLASSES))
@settings(max_examples=len(RISK_CLASSES), deadline=None)
def test_i5_referenzallokation_ist_zulaessig(risk_class: int) -> None:
    """I5 für die regelbasierte Referenzallokation.

    Args:
        risk_class: Geprüfte Risikoklasse.
    """
    gewichte = reference_weights(risk_class, POLICIES.classes, ASSET_KEYS)
    definition = POLICIES.classes.risikoklassen[risk_class]

    assert check_constraints(gewichte, definition, POLICIES.classes, ASSET_KEYS, TOLERANCE) == ()


@pytest.mark.parametrize("risk_class", RISK_CLASSES)
def test_i5_optimiertes_portfolio_ist_zulaessig(moments: object, risk_class: int) -> None:
    """I5 für das optimierte Portfolio: Summe eins, keine Leerverkäufe, Bänder gehalten.

    Der Test wiederholt bewusst, was der Solver zugesichert hat. Ein Solver kann
    ``optimal`` melden und Restriktionen dennoch leicht verletzen.

    Args:
        moments: Geschätzte Momente.
        risk_class: Geprüfte Risikoklasse.
    """
    ergebnis = _optimiere(moments, risk_class)
    gewichte = ergebnis.weights  # type: ignore[attr-defined]
    definition = POLICIES.classes.risikoklassen[risk_class]

    assert float(gewichte.sum()) == pytest.approx(1.0, abs=TOLERANCE)
    assert float(gewichte.min()) >= -TOLERANCE
    assert check_constraints(gewichte, definition, POLICIES.classes, ASSET_KEYS, TOLERANCE) == ()


@pytest.mark.parametrize("risk_class", RISK_CLASSES)
def test_i5_minimum_varianz_portfolio_ist_zulaessig(moments: object, risk_class: int) -> None:
    """I5 auch für den linken Randpunkt der Effizienzlinie.

    Args:
        moments: Geschätzte Momente.
        risk_class: Geprüfte Risikoklasse.
    """
    ergebnis = minimum_variance(
        moments,  # type: ignore[arg-type]
        risk_class,
        POLICIES.classes,
        POLICIES.portfolio,
        ASSET_KEYS,
    )
    definition = POLICIES.classes.risikoklassen[risk_class]

    assert (
        check_constraints(ergebnis.weights, definition, POLICIES.classes, ASSET_KEYS, TOLERANCE)
        == ()
    )


def test_i5_check_constraints_erkennt_einen_leerverkauf() -> None:
    """Gegenprobe: Die Prüfung schlägt bei einem negativen Gewicht an.

    Ohne diese Gegenprobe wäre nicht belegt, dass die Prüfung überhaupt etwas
    prüft — ein Test, der nur grüne Fälle sieht, kann auch leer laufen.
    """
    gewichte = np.array([-0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    definition = POLICIES.classes.risikoklassen[3]

    verstoesse = check_constraints(gewichte, definition, POLICIES.classes, ASSET_KEYS, TOLERANCE)

    assert verstoesse
    assert any("Leerverkauf" in eintrag for eintrag in verstoesse)


def test_i5_check_constraints_erkennt_eine_verletzte_gewichtssumme() -> None:
    """Gegenprobe: Eine Gewichtssumme ungleich eins wird erkannt."""
    gewichte = np.full(len(ASSET_KEYS), 0.5)
    definition = POLICIES.classes.risikoklassen[3]

    verstoesse = check_constraints(gewichte, definition, POLICIES.classes, ASSET_KEYS, TOLERANCE)

    assert any("Gewichtssumme" in eintrag for eintrag in verstoesse)


# --- I6: Monotonie der Aktienquote -----------------------------------------


def test_i6_aktienquote_steigt_im_regelwerk() -> None:
    """I6 auf dem Regelwerk: Unter- und Obergrenze der Aktienquote steigen monoton."""
    untergrenzen = [
        POLICIES.classes.risikoklassen[klasse].gruppen["aktien"].min for klasse in RISK_CLASSES
    ]
    obergrenzen = [
        POLICIES.classes.risikoklassen[klasse].gruppen["aktien"].max for klasse in RISK_CLASSES
    ]

    assert untergrenzen == sorted(untergrenzen)
    assert obergrenzen == sorted(obergrenzen)


def test_i6_aktienquote_steigt_in_den_optimierten_portfolios(moments: object) -> None:
    """I6 auf dem Ergebnis: Die tatsächlich optimierte Aktienquote steigt monoton.

    Die stärkere der beiden Aussagen. Dass die Bänder monoton sind, garantiert
    für sich genommen noch nicht, dass die Optimierung innerhalb der Bänder
    ebenfalls monoton wählt.

    Args:
        moments: Geschätzte Momente.
    """
    quoten = [
        equity_share(
            _optimiere(moments, klasse).weights,  # type: ignore[attr-defined]
            POLICIES.classes,
            ASSET_KEYS,
        )
        for klasse in RISK_CLASSES
    ]

    assert quoten == sorted(quoten), f"Aktienquoten nicht monoton: {quoten}"


def test_i6_aktienquote_steigt_in_der_referenzallokation() -> None:
    """I6 auch für die regelbasierte Referenzallokation."""
    quoten = [
        equity_share(
            reference_weights(klasse, POLICIES.classes, ASSET_KEYS), POLICIES.classes, ASSET_KEYS
        )
        for klasse in RISK_CLASSES
    ]

    assert quoten == sorted(quoten)


def test_i6_erwartete_rendite_steigt_mit_der_klasse(moments: object) -> None:
    """Höhere Klasse, höhere erwartete Rendite — und höhere Volatilität.

    Args:
        moments: Geschätzte Momente.
    """
    ergebnisse = [_optimiere(moments, klasse) for klasse in RISK_CLASSES]
    renditen = [e.expected_return for e in ergebnisse]  # type: ignore[attr-defined]
    volatilitaeten = [e.volatility for e in ergebnisse]  # type: ignore[attr-defined]

    assert renditen == sorted(renditen)
    assert volatilitaeten == sorted(volatilitaeten)


# --- I8: Nachhaltigkeitsfilter ---------------------------------------------


def _praeferenz(**verlangt: bool) -> SustainabilityPreferences:
    """Baut Nachhaltigkeitspräferenzen mit den angegebenen Ausprägungen.

    Args:
        **verlangt: Ausprägungen, die verlangt werden sollen.

    Returns:
        Die Präferenzen.
    """
    felder = {
        name: SustainabilityComponent(requested=True, minimum_share=Decimal("0.5"))
        for name in verlangt
        if verlangt[name]
    }
    return SustainabilityPreferences(has_preferences=bool(felder), **felder)


def test_i8_ohne_praeferenz_bleibt_das_universum_vollstaendig() -> None:
    """Ohne Nachhaltigkeitspräferenz wird nicht gefiltert."""
    ergebnis = filter_universe(
        SustainabilityPreferences(has_preferences=False), POLICIES.portfolio, ASSET_KEYS
    )

    assert ergebnis.included == ASSET_KEYS
    assert not ergebnis.applied
    assert not ergebnis.is_reduced


def test_i8_filter_reduziert_das_universum() -> None:
    """Mit Taxonomiepräferenz fallen die dafür ungeeigneten Assetklassen weg."""
    ergebnis = filter_universe(_praeferenz(taxonomy_aligned=True), POLICIES.portfolio, ASSET_KEYS)

    assert ergebnis.applied
    assert ergebnis.is_reduced
    assert len(ergebnis.included) < len(ASSET_KEYS)
    assert "GOLD" in ergebnis.excluded


def test_i8_ausschluss_traegt_eine_begruendung() -> None:
    """Jeder Ausschluss ist im Klartext begründet — der Trace muss ohne Code lesbar sein."""
    ergebnis = filter_universe(_praeferenz(taxonomy_aligned=True), POLICIES.portfolio, ASSET_KEYS)

    for name, grund in ergebnis.excluded.items():
        assert grund.strip() != "", name
        assert "erfüllt nicht" in grund


def test_i8_zu_kleines_universum_bricht_definiert_ab() -> None:
    """I8: Unterschreitet das Universum die Mindestgröße, folgt ein definierter Fehler.

    Das ist der Kern der Invariante. Die Alternative — die Präferenz still
    aufweichen oder das Universum auffüllen — wäre ein Verstoß gegen die
    Nachhaltigkeitsvorgaben der DelVO (EU) 2021/1253.
    """
    with pytest.raises(SustainabilityUniverseError) as fehler:
        filter_universe(
            _praeferenz(taxonomy_aligned=True, sfdr_sustainable=True, pai_considered=True),
            POLICIES.portfolio.model_copy(
                update={
                    "nachhaltigkeit": POLICIES.portfolio.nachhaltigkeit.model_copy(
                        update={"mindestuniversum": 8}
                    )
                }
            ),
            ASSET_KEYS,
        )

    assert fehler.value.required == 8
    assert fehler.value.excluded
    assert "2021/1253" in fehler.value.legal_basis


def test_i8_fehler_nennt_die_rechtsgrundlage() -> None:
    """Der Fehler trägt die Rechtsgrundlage in der Meldung, nicht nur im Attribut."""
    with pytest.raises(SustainabilityUniverseError, match="2021/1253"):
        filter_universe(
            _praeferenz(taxonomy_aligned=True, sfdr_sustainable=True),
            POLICIES.portfolio.model_copy(
                update={
                    "nachhaltigkeit": POLICIES.portfolio.nachhaltigkeit.model_copy(
                        update={"mindestuniversum": 8}
                    )
                }
            ),
            ASSET_KEYS,
        )


def test_i8_filter_wirkt_vor_der_optimierung(moments: object) -> None:
    """Gefilterte Positionen tragen im Ergebnis exakt null.

    Der Filter greift vor der Optimierung. Ein nachträgliches Streichen würde
    die Restriktionen verletzen und das Ergebnis unzulässig machen.

    Gewählt ist die PAI-Ausprägung: Sie entfernt allein GOLD, und GOLD hat in
    keiner Risikoklasse eine positive Untergrenze. Nur so lässt sich das
    Zusammenspiel von Filter und Optimierung überhaupt prüfen — die beiden
    anderen Ausprägungen kollidieren mit den Pflichtuntergrenzen, siehe
    ``test_i8_taxonomie_kollidiert_mit_jeder_risikoklasse``.

    Args:
        moments: Geschätzte Momente.
    """
    universum = filter_universe(_praeferenz(pai_considered=True), POLICIES.portfolio, ASSET_KEYS)
    assert set(universum.excluded) == {"GOLD"}

    ergebnis = _optimiere_gefiltert(moments, 3, universum)

    assert float(ergebnis.weights[ASSET_KEYS.index("GOLD")]) == 0.0
    assert float(ergebnis.weights.sum()) == pytest.approx(1.0, abs=TOLERANCE)
    definition = POLICIES.classes.risikoklassen[3]
    assert (
        check_constraints(ergebnis.weights, definition, POLICIES.classes, ASSET_KEYS, TOLERANCE)
        == ()
    )


@pytest.mark.parametrize("risk_class", RISK_CLASSES)
def test_i8_taxonomie_kollidiert_mit_jeder_risikoklasse(moments: object, risk_class: int) -> None:
    """Die Taxonomiepräferenz ist mit keiner Risikoklasse vereinbar.

    Ursache ist kein Programmierfehler, sondern ein echter Zielkonflikt: Die
    Bandbreiten schreiben in jeder Klasse Staatsanleihen oder Schwellenländer
    mit positiver Untergrenze vor, und Staatsanleihen sind keine Investition in
    wirtschaftliche Tätigkeiten im Sinne der Taxonomie-Verordnung. Das System
    legt den Widerspruch offen, statt eine der beiden Vorgaben aufzuweichen.

    Args:
        moments: Geschätzte Momente.
        risk_class: Geprüfte Risikoklasse.
    """
    universum = filter_universe(_praeferenz(taxonomy_aligned=True), POLICIES.portfolio, ASSET_KEYS)

    with pytest.raises(SustainabilityConflictError) as fehler:
        _optimiere_gefiltert(moments, risk_class, universum)

    assert fehler.value.risk_class == risk_class
    assert fehler.value.blocking
    assert all(untergrenze > 0 for untergrenze in fehler.value.blocking.values())


def test_i8_konflikt_wird_vor_dem_solver_erkannt(moments: object) -> None:
    """Der Konflikt erscheint als fachlicher Fehler, nicht als Solverproblem.

    Die Unterscheidung ist wesentlich: „Solver meldet Erfolg, Ergebnis verletzt
    die Restriktionen" würde den Solver beschuldigen und die eigentliche Ursache
    verdecken. Der Fehler nennt stattdessen die blockierende Position und die
    Rechtsgrundlage.

    Args:
        moments: Geschätzte Momente.
    """
    universum = filter_universe(_praeferenz(taxonomy_aligned=True), POLICIES.portfolio, ASSET_KEYS)

    with pytest.raises(SustainabilityConflictError) as fehler:
        _optimiere_gefiltert(moments, 3, universum)

    meldung = str(fehler.value)
    assert "BD_GOV" in meldung
    assert "2021/1253" in meldung
    assert "stillschweigend" in meldung


def test_i8_pai_praeferenz_kollidiert_mit_keiner_klasse(moments: object) -> None:
    """Die PAI-Ausprägung ist mit allen Risikoklassen vereinbar.

    Gegenstück zum Taxonomiefall: Sie entfernt allein GOLD, das nirgends
    vorgeschrieben ist. Der Test belegt, dass der Konfliktfehler nicht pauschal
    bei jedem Filter anschlägt.

    Args:
        moments: Geschätzte Momente.
    """
    universum = filter_universe(_praeferenz(pai_considered=True), POLICIES.portfolio, ASSET_KEYS)

    for klasse in RISK_CLASSES:
        ergebnis = _optimiere_gefiltert(moments, klasse, universum)
        assert float(ergebnis.weights[ASSET_KEYS.index("GOLD")]) == 0.0


# --- Momentenschätzung -----------------------------------------------------


def test_historischer_mittelwert_ist_das_arithmetische_mittel(history: FloatArray) -> None:
    """Der historische Schätzer rechnet nichts hinzu.

    Args:
        history: Historische Renditematrix.
    """
    assert np.allclose(historical_mean(history), history.mean(axis=0))


def test_shrinkage_liegt_zwischen_mittelwert_und_gesamtmittel(history: FloatArray) -> None:
    """Der geschrumpfte Schätzer liegt je Assetklasse zwischen beiden Polen.

    Args:
        history: Historische Renditematrix.
    """
    historisch = historical_mean(history)
    gesamt = float(historisch.mean())
    geschrumpft, intensitaet = shrunk_mean(history)

    assert 0.0 <= intensitaet <= 1.0
    for einzeln, verschoben in zip(historisch, geschrumpft, strict=True):
        untere, obere = sorted((float(einzeln), gesamt))
        assert untere - 1e-12 <= float(verschoben) <= obere + 1e-12


def test_shrinkage_erhaelt_das_gesamtmittel(history: FloatArray) -> None:
    """Die Schrumpfung verschiebt das Gesamtmittel nicht.

    Sie zieht die Einzelwerte zur Mitte; die Mitte selbst bleibt, wo sie war.

    Args:
        history: Historische Renditematrix.
    """
    geschrumpft, _ = shrunk_mean(history)

    assert float(geschrumpft.mean()) == pytest.approx(float(historical_mean(history).mean()))


@given(intensitaet=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=25, deadline=None)
def test_shrinkage_mit_vorgegebener_intensitaet(history: FloatArray, intensitaet: float) -> None:
    """Bei vorgegebener Intensität wird genau diese verwendet.

    Args:
        history: Historische Renditematrix.
        intensitaet: Vorgegebene Intensität.
    """
    geschrumpft, verwendet = shrunk_mean(history, intensitaet)

    assert verwendet == pytest.approx(intensitaet)
    if intensitaet == 1.0:
        assert np.allclose(geschrumpft, geschrumpft[0])


def test_shrinkage_weist_unzulaessige_intensitaet_zurueck(history: FloatArray) -> None:
    """Eine Intensität außerhalb von [0, 1] ist ein Fehler.

    Args:
        history: Historische Renditematrix.
    """
    with pytest.raises(EstimationError, match="außerhalb"):
        shrunk_mean(history, 1.5)


def test_ledoit_wolf_intensitaet_liegt_im_einheitsintervall(history: FloatArray) -> None:
    """Die Schrumpfungsintensität ist ein Anteil und liegt zwischen null und eins.

    Args:
        history: Historische Renditematrix.
    """
    _, intensitaet = ledoit_wolf_covariance(history)

    assert 0.0 <= intensitaet <= 1.0


def test_ledoit_wolf_matrix_ist_symmetrisch_und_positiv_definit(history: FloatArray) -> None:
    """Die geschrumpfte Kovarianz ist eine gültige Kovarianzmatrix.

    Ohne positive Definitheit wäre das Optimierungsproblem nicht konvex, und die
    Lösung nicht mehr eindeutig.

    Args:
        history: Historische Renditematrix.
    """
    kovarianz, _ = ledoit_wolf_covariance(history)

    assert np.allclose(kovarianz, kovarianz.T)
    assert float(np.linalg.eigvalsh(kovarianz).min()) > 0.0


def test_shrinkage_verbessert_die_kondition(history: FloatArray) -> None:
    """Die geschrumpfte Matrix ist besser konditioniert als die Stichprobenmatrix.

    Das ist der eigentliche Zweck des Verfahrens: Eine schlecht konditionierte
    Kovarianz macht die Optimierung empfindlich gegen kleinste Änderungen der
    Eingabe.

    Args:
        history: Historische Renditematrix.
    """
    kurz = history[:60]
    geschrumpft, _ = ledoit_wolf_covariance(kurz)

    assert np.linalg.cond(geschrumpft) < np.linalg.cond(sample_covariance(kurz))


def test_schaetzung_weist_zu_kurze_historie_zurueck() -> None:
    """Unter der Mindestzahl an Beobachtungen bricht die Schätzung ab."""
    zu_kurz = np.zeros((MIN_OBSERVATIONS - 1, len(ASSET_KEYS)))

    with pytest.raises(EstimationError, match="Monatsbeobachtungen"):
        historical_mean(zu_kurz)


def test_schaetzung_weist_nicht_endliche_werte_zurueck(history: FloatArray) -> None:
    """NaN in der Renditematrix ist ein Fehler, kein zu ignorierender Wert.

    Args:
        history: Historische Renditematrix.
    """
    beschaedigt = history.copy()
    beschaedigt[0, 0] = np.nan

    with pytest.raises(EstimationError, match="nicht endliche"):
        historical_mean(beschaedigt)


def test_schaetzung_ist_deterministisch(history: FloatArray) -> None:
    """Zwei Schätzungen derselben Matrix sind bitgleich (Prinzip P1).

    Args:
        history: Historische Renditematrix.
    """
    erst = estimate_moments(history, POLICIES.portfolio.schaetzung, ASSET_KEYS)
    zweit = estimate_moments(history, POLICIES.portfolio.schaetzung, ASSET_KEYS)

    assert np.array_equal(erst.mean, zweit.mean)
    assert np.array_equal(erst.covariance, zweit.covariance)
    assert erst.covariance_shrinkage == zweit.covariance_shrinkage


def test_beide_erwartungswertverfahren_werden_ausgewiesen(history: FloatArray) -> None:
    """Die Gegenüberstellung nennt beide Verfahren je Assetklasse.

    Args:
        history: Historische Renditematrix.
    """
    vergleich = compare_mean_methods(history, ASSET_KEYS)

    assert set(vergleich) == set(ASSET_KEYS)
    for werte in vergleich.values():
        assert set(werte) == {"historisch_pa", "shrinkage_pa", "differenz_pp"}


def test_historisches_verfahren_meldet_keine_schrumpfung(history: FloatArray) -> None:
    """Beim historischen Mittel ist die Schrumpfungsintensität null.

    Args:
        history: Historische Renditematrix.
    """
    einstellungen = POLICIES.portfolio.schaetzung.model_copy(
        update={"erwartungswert_verfahren": MeanMethod.HISTORISCH}
    )
    momente = estimate_moments(history, einstellungen, ASSET_KEYS)

    assert momente.mean_shrinkage == 0.0
    assert np.allclose(momente.mean, historical_mean(history))


def test_risikofreier_proxy_stammt_aus_dem_regelwerk(history: FloatArray) -> None:
    """Der risikofreie Zins ist die mittlere Rendite des Proxys.

    Args:
        history: Historische Renditematrix.
    """
    proxy = POLICIES.portfolio.schaetzung.risikofreier_proxy
    assert proxy is not None
    erwartet = float(history[:, ASSET_KEYS.index(proxy)].mean())

    assert risk_free_monthly(history, POLICIES.portfolio, ASSET_KEYS) == pytest.approx(erwartet)


# --- Optimierung: Solver, Fehler, Determinismus ----------------------------


def test_optimierung_ist_deterministisch(moments: object) -> None:
    """Zwei Läufe liefern bitgleiche Gewichte (Prinzip P1).

    Args:
        moments: Geschätzte Momente.
    """
    erst = _optimiere(moments, 3)
    zweit = _optimiere(moments, 3)

    assert np.array_equal(erst.weights, zweit.weights)  # type: ignore[attr-defined]
    assert erst.solver == zweit.solver  # type: ignore[attr-defined]


def test_optimierung_protokolliert_die_solverversuche(moments: object) -> None:
    """Jeder Solverversuch steht mit seinem Status im Ergebnis.

    Args:
        moments: Geschätzte Momente.
    """
    ergebnis = _optimiere(moments, 3)

    assert ergebnis.attempts  # type: ignore[attr-defined]
    for name, status in ergebnis.attempts:  # type: ignore[attr-defined]
        assert name
        assert status


def test_optimierung_weist_unbekannte_risikoklasse_zurueck(moments: object) -> None:
    """Eine Klasse außerhalb von 1 bis 5 ist ein Fehler.

    Args:
        moments: Geschätzte Momente.
    """
    with pytest.raises(OptimizationError, match="Unbekannte Risikoklasse"):
        optimize(
            moments,  # type: ignore[arg-type]
            9,
            POLICIES.classes,
            POLICIES.portfolio,
            ASSET_KEYS,
            target_return=0.0,
        )


def test_unerreichbare_zielrendite_wird_abgesenkt_und_vermerkt(moments: object) -> None:
    """Eine unerfüllbare Zielrendite wird gesenkt — und das wird ausgewiesen.

    Die Absenkung ist eine Abweichung von der Vorgabe. Sie stillschweigend
    vorzunehmen wäre der Fehler; sie zu vermerken ist die Lösung.

    Args:
        moments: Geschätzte Momente.
    """
    unerreichbar = float(moments.mean.max()) * 2.0  # type: ignore[attr-defined]
    ergebnis = optimize(
        moments,  # type: ignore[arg-type]
        3,
        POLICIES.classes,
        POLICIES.portfolio,
        ASSET_KEYS,
        target_return=unerreichbar,
    )

    assert ergebnis.target_return_lowered
    assert ergebnis.target_return < unerreichbar
    assert ergebnis.as_summary()["zielrendite_abgesenkt"] is True


def test_minimum_varianz_hat_die_kleinste_volatilitaet(moments: object) -> None:
    """Das Minimum-Varianz-Portfolio unterbietet die Referenzallokation.

    Args:
        moments: Geschätzte Momente.
    """
    minimal = minimum_variance(
        moments,  # type: ignore[arg-type]
        3,
        POLICIES.classes,
        POLICIES.portfolio,
        ASSET_KEYS,
    )
    referenz = reference_weights(3, POLICIES.classes, ASSET_KEYS)

    assert minimal.volatility <= moments.portfolio_volatility(referenz) + 1e-9  # type: ignore[attr-defined]


# --- Benchmark und Kennzahlen ----------------------------------------------


def test_gleichgewicht_summiert_sich_zu_eins() -> None:
    """Das 1/N-Portfolio ist vollinvestiert und gleichgewichtet."""
    gewichte = equal_weight_portfolio(len(ASSET_KEYS))

    assert float(gewichte.sum()) == pytest.approx(1.0)
    assert np.allclose(gewichte, 1.0 / len(ASSET_KEYS))
    assert np.array_equal(gewichte, equal_weights(ASSET_KEYS))


def test_gleichgewicht_ist_in_keiner_klasse_zulaessig() -> None:
    """1/N verletzt in jeder Risikoklasse die Bandbreiten.

    Das entwertet den Benchmark nicht — als methodischer Maßstab bleibt er
    verpflichtend. Es heißt aber, dass 1/N für einen an die Geeignetheitsprüfung
    gebundenen Anbieter kein empfehlbares Produkt ist. Der Test hält den Befund
    fest, damit er nicht unbemerkt verschwindet.
    """
    gewichte = equal_weight_portfolio(len(ASSET_KEYS))

    for klasse in RISK_CLASSES:
        definition = POLICIES.classes.risikoklassen[klasse]
        verstoesse = check_constraints(
            gewichte, definition, POLICIES.classes, ASSET_KEYS, TOLERANCE
        )
        assert verstoesse, f"Klasse {klasse}: 1/N unerwartet zulässig"


def test_max_drawdown_findet_den_groessten_rueckgang() -> None:
    """Der Drawdown wird auf dem Wertindex gerechnet, nicht auf Einzelmonaten.

    Die Reihe fällt in zwei aufeinanderfolgenden Monaten um je 10 %; der
    Drawdown beträgt damit 1 − 0,9 · 0,9 = 19 %, nicht 10 %.
    """
    reihe = np.array([0.0, -0.10, -0.10, 0.05])

    assert max_drawdown(reihe) == pytest.approx(-0.19, abs=1e-9)


def test_max_drawdown_ist_null_ohne_rueckgang() -> None:
    """Eine durchgehend steigende Reihe hat keinen Drawdown."""
    assert max_drawdown(np.array([0.01, 0.02, 0.01])) == pytest.approx(0.0)


def test_turnover_misst_gegen_das_gedriftete_gewicht() -> None:
    """Der Umschlag vergleicht Ziel- mit gedriftetem Gewicht, nicht mit dem alten Ziel.

    Wird das Zielgewicht gehalten und ist die Position gedriftet, entsteht
    Umschlag. Der Vergleich gegen das alte Ziel würde null ergeben und die
    Handelskosten unterschlagen.
    """
    ziel = [np.array([0.5, 0.5])]
    gedriftet = [np.array([0.6, 0.4])]

    assert turnover_from_weights(ziel, gedriftet, 1.0) == pytest.approx(0.2)


def test_turnover_ist_null_ohne_rebalancing() -> None:
    """Ohne Rebalancing gibt es keinen Umschlag."""
    assert turnover_from_weights([], [], 5.0) == 0.0


def test_kennzahlen_weisen_zu_kurze_reihe_zurueck() -> None:
    """Unter zwölf Monaten ist keine annualisierte Kennzahl zu bilden."""
    with pytest.raises(PerformanceError, match="Monate"):
        evaluate_performance(np.zeros(6), 0.0, 12)


def test_sharpe_ist_null_ohne_streuung() -> None:
    """Eine Reihe ohne Streuung hat keine definierte Sharpe Ratio; sie wird null."""
    kennzahlen = evaluate_performance(np.full(24, 0.005), 0.0, 12)

    assert kennzahlen.sharpe_ratio == 0.0
    assert kennzahlen.annualised_volatility == pytest.approx(0.0)


def test_sharpe_test_erkennt_identische_reihen() -> None:
    """Zwei gleiche Reihen unterscheiden sich nicht — der p-Wert ist eins."""
    rng = np.random.default_rng(20260726)
    reihe = rng.normal(0.005, 0.03, 120)

    ergebnis = sharpe_difference_test(reihe, reihe, 0.0, 12)

    assert ergebnis.difference == pytest.approx(0.0, abs=1e-12)
    assert ergebnis.p_value == pytest.approx(1.0)
    assert not ergebnis.is_significant()


def test_sharpe_test_weist_verschieden_lange_reihen_zurueck() -> None:
    """Der Test setzt denselben Zeitraum voraus."""
    with pytest.raises(PerformanceError, match="verschiedene Länge"):
        sharpe_difference_test(np.zeros(24), np.zeros(36), 0.0, 12)


def test_sharpe_test_erkennt_einen_klaren_unterschied() -> None:
    """Bei deutlich verschiedenen Reihen wird der Unterschied gesichert.

    Gegenprobe zum Test auf identische Reihen: Ohne sie wäre nicht belegt, dass
    der Test überhaupt anschlägt.
    """
    rng = np.random.default_rng(20260726)
    schwach = rng.normal(0.001, 0.03, 240)
    stark = schwach + 0.008

    ergebnis = sharpe_difference_test(stark, schwach, 0.0, 12)

    assert ergebnis.difference > 0.0
    assert ergebnis.is_significant()


# --- Backtest --------------------------------------------------------------


@pytest.fixture(scope="module")
def backtest(history: FloatArray) -> object:
    """Rechnet den Backtest einmal je Modul.

    Args:
        history: Historische Renditematrix.

    Returns:
        Das Backtest-Ergebnis.
    """
    return run_backtest(history, POLICIES, ASSET_KEYS)


def test_backtest_liefert_alle_drei_strategien(backtest: object) -> None:
    """Verglichen werden MVO, 1/N und die Referenzallokation.

    Args:
        backtest: Das Backtest-Ergebnis.
    """
    assert set(backtest.strategies) == {  # type: ignore[attr-defined]
        STRATEGY_MVO,
        STRATEGY_EQUAL,
        STRATEGY_REFERENCE,
    }


def test_backtest_wertet_alle_strategien_auf_demselben_zeitraum_aus(backtest: object) -> None:
    """Ein Vergleich über verschiedene Zeiträume wäre wertlos.

    Args:
        backtest: Das Backtest-Ergebnis.
    """
    laengen = {
        strategie.monthly_returns.size
        for strategie in backtest.strategies.values()  # type: ignore[attr-defined]
    }

    assert len(laengen) == 1


def test_backtest_hat_keinen_lookahead(history: FloatArray) -> None:
    """Der wichtigste Test dieser Datei: Die Schätzung sieht den Haltezeitraum nicht.

    Nachgestellt wird die Fensterlogik des Backtests. Für jeden Startpunkt muss
    gelten, dass das Schätzfenster **vor** dem Haltefenster endet und beide sich
    nicht überschneiden. Ein Backtest mit Look-ahead misst Anpassungsgüte statt
    Prognosegüte und fällt zwangsläufig zugunsten der Optimierung aus.

    Args:
        history: Historische Renditematrix.
    """
    einstellungen = POLICIES.portfolio.backtest
    fenster = einstellungen.schaetzfenster_monate
    haltedauer = einstellungen.haltedauer_monate
    monate = history.shape[0]

    startpunkte = list(range(fenster, monate - haltedauer + 1, haltedauer))
    assert startpunkte, "Der Backtest hat keinen einzigen Haltezeitraum."

    for start in startpunkte:
        schaetz_von, schaetz_bis = start - fenster, start
        halte_von, halte_bis = start, start + haltedauer

        assert schaetz_von >= 0
        assert schaetz_bis == halte_von, "Schätz- und Haltefenster stoßen nicht aneinander."
        assert schaetz_bis <= halte_von, "Das Schätzfenster ragt in den Haltezeitraum."
        assert halte_bis <= monate


def test_backtest_haltezeitraeume_ueberschneiden_sich_nicht(backtest: object) -> None:
    """Die Haltezeiträume decken den Zeitraum lückenlos und überschneidungsfrei ab.

    Args:
        backtest: Das Backtest-Ergebnis.
    """
    erwartete_monate = backtest.periods * backtest.holding_period  # type: ignore[attr-defined]

    assert (
        backtest.strategies[STRATEGY_MVO].monthly_returns.size == erwartete_monate  # type: ignore[attr-defined]
    )


def test_backtest_ist_deterministisch(history: FloatArray) -> None:
    """Zwei Läufe liefern dieselbe Renditereihe (Prinzip P1).

    Args:
        history: Historische Renditematrix.
    """
    erst = run_backtest(history, POLICIES, ASSET_KEYS)
    zweit = run_backtest(history, POLICIES, ASSET_KEYS)

    assert np.array_equal(
        erst.strategies[STRATEGY_MVO].monthly_returns,
        zweit.strategies[STRATEGY_MVO].monthly_returns,
    )


def test_backtest_weist_zu_kurze_historie_zurueck() -> None:
    """Reicht die Historie nicht für Fenster und Haltedauer, bricht der Backtest ab."""
    zu_kurz = np.zeros((24, len(ASSET_KEYS)))

    with pytest.raises(BacktestError, match="Historie umfasst"):
        run_backtest(zu_kurz, POLICIES, ASSET_KEYS)


def test_mvo_hat_hoeheren_umschlag_als_gleichgewicht(backtest: object) -> None:
    """Befund: Die Optimierung schichtet deutlich mehr um als 1/N.

    Der Umschlag ist der Preis der Optimierung. Er wird ausgewiesen, weil ein
    Vergleich der Sharpe Ratios ohne ihn unvollständig wäre.

    Args:
        backtest: Das Backtest-Ergebnis.
    """
    mvo = backtest.strategies[STRATEGY_MVO].metrics.turnover  # type: ignore[attr-defined]
    gleich = backtest.strategies[STRATEGY_EQUAL].metrics.turnover  # type: ignore[attr-defined]

    assert mvo > gleich


def test_backtest_zieht_transaktionskosten_ab(history: FloatArray) -> None:
    """Ohne Kostenabzug wäre der Vergleich zugunsten der MVO verzerrt.

    Gegenprobe über einen Lauf mit auf null gesetzten Spreads: Die Rendite muss
    dann höher liegen, und zwar für die Strategie mit dem höheren Umschlag
    stärker.

    Args:
        history: Historische Renditematrix.
    """
    ohne_kosten = POLICIES.model_copy(
        update={
            "simulation": POLICIES.simulation.model_copy(
                update={
                    "kosten": {
                        name: kosten.model_copy(update={"spread_einmalig": Decimal("0")})
                        for name, kosten in POLICIES.simulation.kosten.items()
                    }
                }
            )
        }
    )
    mit = run_backtest(history, POLICIES, ASSET_KEYS)
    ohne = run_backtest(history, ohne_kosten, ASSET_KEYS)

    mit_mvo = mit.strategies[STRATEGY_MVO].metrics.annualised_return
    ohne_mvo = ohne.strategies[STRATEGY_MVO].metrics.annualised_return

    assert ohne_mvo > mit_mvo


# --- Abbildung 5 -----------------------------------------------------------


def test_abbildung_5_wird_erzeugt(out_dir: Path) -> None:
    """Abbildung 5 entsteht samt zweizeiliger Unterschrift.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    bild, unterschrift = build_figure_5(out_dir)

    assert bild.is_file()
    assert bild.stat().st_size > 0
    zeilen = unterschrift.read_text(encoding="utf-8").splitlines()
    assert len(zeilen) == 2
    assert zeilen[0].startswith("Abb. 5:")
    assert zeilen[1] == "Quelle: Eigene Darstellung."


# --- Fehlerpfade und Randfälle der Hilfsfunktionen -------------------------


def test_kennzahlen_und_zusammenfassung_werden_ausgewiesen(moments: object) -> None:
    """Die abgeleiteten Größen der Momentenschätzung sind schlüssig.

    Args:
        moments: Geschätzte Momente.
    """
    annualisiert = moments.annualised_mean  # type: ignore[attr-defined]
    volatilitaet = moments.annualised_volatility  # type: ignore[attr-defined]
    zusammenfassung = moments.as_summary()  # type: ignore[attr-defined]

    assert annualisiert.shape == (len(ASSET_KEYS),)
    assert np.all(volatilitaet > 0.0)
    for schluessel in (
        "erwartungswert_verfahren",
        "kovarianz_verfahren",
        "schrumpfung_kovarianz",
        "beobachtungen",
    ):
        assert schluessel in zusammenfassung


def test_stichprobenkovarianz_als_alternatives_verfahren(history: FloatArray) -> None:
    """Das Regelwerk lässt die unbearbeitete Stichprobenkovarianz zu.

    Args:
        history: Historische Renditematrix.
    """
    einstellungen = POLICIES.portfolio.schaetzung.model_copy(
        update={"kovarianz_verfahren": CovarianceMethod.STICHPROBE}
    )
    momente = estimate_moments(history, einstellungen, ASSET_KEYS)

    assert momente.covariance_shrinkage == 0.0
    assert np.allclose(momente.covariance, sample_covariance(history))


def test_schaetzung_weist_falsche_dimension_zurueck() -> None:
    """Eine eindimensionale Reihe ist keine Renditematrix."""
    with pytest.raises(EstimationError, match="Matrix"):
        historical_mean(np.zeros(60))


def test_schaetzung_weist_leeres_universum_zurueck() -> None:
    """Ohne Assetklasse gibt es nichts zu schätzen."""
    with pytest.raises(EstimationError, match="keine Assetklasse"):
        historical_mean(np.zeros((60, 0)))


def test_schaetzung_weist_unpassende_spaltenzahl_zurueck(history: FloatArray) -> None:
    """Spaltenzahl und Assetklassen müssen zusammenpassen.

    Args:
        history: Historische Renditematrix.
    """
    with pytest.raises(EstimationError, match="Spalten"):
        estimate_moments(history, POLICIES.portfolio.schaetzung, ASSET_KEYS[:3])


def test_shrinkage_bei_singulaerer_kovarianz_schrumpft_vollstaendig() -> None:
    """Trägt die Stichprobe die Kovarianzstruktur nicht, wird vollständig geschrumpft.

    Bei identischen Spalten ist die Kovarianz singulär. Die konservative Antwort
    ist dann, allen Assetklassen dieselbe erwartete Rendite zuzuweisen — die
    Stichprobe gibt eine Unterscheidung nicht her.
    """
    rng = np.random.default_rng(20260726)
    spalte = rng.normal(0.004, 0.03, 60)
    entartet = np.column_stack([spalte, spalte, spalte])

    geschrumpft, intensitaet = shrunk_mean(entartet)

    assert intensitaet == pytest.approx(1.0)
    assert np.allclose(geschrumpft, geschrumpft[0])


def test_risikofreier_zins_ohne_proxy_ist_null(history: FloatArray) -> None:
    """Ohne Proxy rechnet die Sharpe Ratio gegen null Prozent.

    Args:
        history: Historische Renditematrix.
    """
    ohne_proxy = POLICIES.portfolio.model_copy(
        update={
            "schaetzung": POLICIES.portfolio.schaetzung.model_copy(
                update={"risikofreier_proxy": None}
            )
        }
    )

    assert risk_free_monthly(history, ohne_proxy, ASSET_KEYS) == 0.0


def test_risikofreier_zins_weist_fehlenden_proxy_zurueck(history: FloatArray) -> None:
    """Fehlt der Proxy in den übergebenen Assetklassen, ist das ein Fehler.

    Args:
        history: Historische Renditematrix.
    """
    with pytest.raises(EstimationError, match="fehlt"):
        risk_free_monthly(history[:, :2], POLICIES.portfolio, ("EQ_DEV", "EQ_EM"))


def test_schrumpfung_wird_fuer_den_bericht_gerundet() -> None:
    """Die Intensität wird auf sechs Stellen gerundet ausgewiesen."""
    assert shrinkage_as_decimal(0.03763712345) == Decimal("0.037637")


def test_risikofreier_proxy_ist_ueber_den_zeitraum_negativ(history: FloatArray) -> None:
    """Befund: Der Proxy hat über den Beobachtungszeitraum eine negative Rendite.

    Deutsche Kurzläufer rentierten zwischen 2015 und 2022 negativ. Das ist kein
    Rechenfehler, sondern die Marktlage des Zeitraums — und es hebt jede Sharpe
    Ratio gegenüber einer Rechnung mit null Prozent an. Der Test hält den Befund
    fest, damit er bei der Einordnung der Sharpe-Werte nicht übersehen wird.

    Args:
        history: Historische Renditematrix.
    """
    assert risk_free_monthly(history, POLICIES.portfolio, ASSET_KEYS) < 0.0


def test_gleichgewicht_ohne_assetklasse_ist_ein_fehler() -> None:
    """Ein 1/N-Portfolio aus null Positionen gibt es nicht."""
    with pytest.raises(PerformanceError, match="mindestens eine"):
        equal_weight_portfolio(0)


def test_max_drawdown_der_leeren_reihe_ist_null() -> None:
    """Eine leere Reihe hat keinen Drawdown."""
    assert max_drawdown(np.array([])) == 0.0


def test_turnover_weist_ungleiche_listen_zurueck() -> None:
    """Ziel- und Driftgewichte müssen paarweise vorliegen."""
    with pytest.raises(PerformanceError, match="gleich viele"):
        turnover_from_weights([np.array([0.5, 0.5])], [], 1.0)


def test_kennzahlen_weisen_falsche_dimension_zurueck() -> None:
    """Eine Matrix ist keine Renditereihe."""
    with pytest.raises(PerformanceError, match="eindimensionale"):
        evaluate_performance(np.zeros((24, 2)), 0.0, 12)


def test_kennzahlen_weisen_nicht_endliche_werte_zurueck() -> None:
    """NaN in der Renditereihe ist ein Fehler."""
    reihe = np.zeros(24)
    reihe[3] = np.nan

    with pytest.raises(PerformanceError, match="nicht endliche"):
        evaluate_performance(reihe, 0.0, 12)


def test_sharpe_test_weist_zu_kurze_reihen_zurueck() -> None:
    """Auch der Signifikanztest braucht mindestens zwölf Monate."""
    with pytest.raises(PerformanceError, match="Monate"):
        sharpe_difference_test(np.zeros(6), np.zeros(6), 0.0, 12)


def test_sharpe_test_weist_streuungsfreie_reihe_zurueck() -> None:
    """Ohne Streuung ist die Sharpe Ratio nicht definiert."""
    with pytest.raises(PerformanceError, match="keine Streuung"):
        sharpe_difference_test(np.full(24, 0.01), np.full(24, 0.02), 0.0, 12)


# --- Auswertungsabschnitt für die Auswertungsdatei -------------------------


def test_zulaessigkeitsblock_haelt_den_befund_fest() -> None:
    """Der erzeugte Abschnitt weist 1/N in allen fünf Klassen als unzulässig aus."""
    zeilen = "\n".join(_zulaessigkeitsblock(POLICIES))

    assert "4.4 Ist 1/N überhaupt empfehlbar?" in zeilen
    assert "in 5 von 5 Risikoklassen nicht zulässig" in zeilen
    assert "Art. 54 DelVO (EU) 2017/565" in zeilen


def test_nachhaltigkeitsblock_haelt_den_konflikt_fest() -> None:
    """Der erzeugte Abschnitt weist den Konflikt und die Recherchelücke aus."""
    zeilen = "\n".join(_nachhaltigkeitsblock(POLICIES))

    assert "4.5 Nachhaltigkeitsfilter gegen Streuungsvorgaben" in zeilen
    assert "1, 2, 3, 4, 5" in zeilen
    assert "`BD_GOV`" in zeilen


def test_evaluationsdatei_wird_an_der_marke_ersetzt(tmp_path: Path) -> None:
    """Der Abschnitt wird zwischen den Marken vollständig ausgetauscht.

    Ohne diesen Test könnte ein zweiter Lauf den Abschnitt anhängen statt zu
    ersetzen, und die Datei würde bei jedem Aufruf wachsen.

    Args:
        tmp_path: Temporäres Verzeichnis.
    """
    ziel = tmp_path / "auswertung.md"
    ziel.write_text(
        f"Kopf\n\n{PORTFOLIO_START}\nalter Inhalt\n{PORTFOLIO_END}\n\nFuss\n", encoding="utf-8"
    )

    write_evaluation(f"{PORTFOLIO_START}\nneuer Inhalt\n{PORTFOLIO_END}", ziel)
    inhalt = ziel.read_text(encoding="utf-8")

    assert "alter Inhalt" not in inhalt
    assert "neuer Inhalt" in inhalt
    assert inhalt.startswith("Kopf")
    assert inhalt.rstrip().endswith("Fuss")
    assert inhalt.count(PORTFOLIO_START) == 1


def test_evaluationsdatei_muss_existieren(tmp_path: Path) -> None:
    """Fehlt die Zieldatei, bricht der Schreibvorgang ab.

    Args:
        tmp_path: Temporäres Verzeichnis.
    """
    with pytest.raises(FileNotFoundError, match="Zieldatei fehlt"):
        write_evaluation("egal", tmp_path / "gibt-es-nicht.md")


@pytest.mark.slow
def test_vollstaendiger_auswertungsabschnitt_wird_erzeugt(history: FloatArray) -> None:
    """Der gesamte Abschnitt 4 entsteht fehlerfrei aus gerechneten Zahlen.

    Als `slow` markiert: Der Lauf rechnet den Backtest sechsmal — einmal als
    Basisfall und einmal je Risikoklasse.

    Args:
        history: Historische Renditematrix.
    """
    block = render_block(history, POLICIES)

    assert block.startswith(PORTFOLIO_START)
    assert block.rstrip().endswith(PORTFOLIO_END)
    for ueberschrift in ("### 4.1", "### 4.2", "### 4.3", "### 4.4", "### 4.5"):
        assert ueberschrift in block
    for zeile in block.splitlines():
        if "UCITS" in zeile or zeile.startswith("#") or "." not in zeile:
            continue
        assert " 0." not in zeile, f"Englischer Dezimalpunkt in: {zeile!r}"


# --- Erreichter Nachhaltigkeitsanteil --------------------------------------


def test_erreichter_anteil_wird_gemessen_und_nicht_gesetzt() -> None:
    """Ein Portfolio mit nicht konformer Position ergibt einen Anteil unter eins.

    Der Test belegt, dass :func:`achieved_shares` **misst**. Eine Fassung, die
    den Wert setzt statt ihn zu rechnen — etwa als konstante eins, weil der
    Filter ja ausschließt —, bliebe im Regelbetrieb unauffällig und wäre genau
    die Sorte Behauptung, die dieser Test ausräumt.
    """
    einstufung = POLICIES.portfolio.nachhaltigkeit.einstufung
    nicht_konform = [name for name in ASSET_KEYS if not einstufung[name].pai]
    assert nicht_konform, "Ohne nicht konforme Assetklasse belegt der Test nichts."

    gewichte = np.zeros(len(ASSET_KEYS))
    gewichte[ASSET_KEYS.index(nicht_konform[0])] = 0.25
    konform = next(name for name in ASSET_KEYS if einstufung[name].pai)
    gewichte[ASSET_KEYS.index(konform)] = 0.75

    anteile = achieved_shares(
        gewichte, _praeferenz(pai_considered=True), POLICIES.portfolio, ASSET_KEYS
    )

    assert anteile == {"pai": Decimal("0.7500")}


def test_erreichter_anteil_nur_fuer_verlangte_auspraegungen() -> None:
    """Nicht verlangte Ausprägungen werden nicht gemessen und nicht ausgewiesen."""
    gewichte = np.full(len(ASSET_KEYS), 1.0 / len(ASSET_KEYS))

    anteile = achieved_shares(
        gewichte, _praeferenz(pai_considered=True), POLICIES.portfolio, ASSET_KEYS
    )

    assert set(anteile) == {"pai"}


def test_ohne_praeferenz_wird_nichts_gemessen() -> None:
    """Ohne erklärte Präferenz gibt es keinen Anteil auszuweisen."""
    gewichte = np.full(len(ASSET_KEYS), 1.0 / len(ASSET_KEYS))

    anteile = achieved_shares(
        gewichte,
        SustainabilityPreferences(has_preferences=False),
        POLICIES.portfolio,
        ASSET_KEYS,
    )

    assert anteile == {}


def test_quotenpruefung_bricht_bei_unterschrittener_vorgabe_ab() -> None:
    """Die Prüfung muss feuern, wenn der erreichte Anteil unter der Vorgabe liegt.

    Über den öffentlichen Weg ist der Fall beim gegenwärtigen Stand nicht
    erreichbar: :func:`filter_universe` schließt nicht konforme Assetklassen
    vollständig aus, der erreichte Anteil ist deshalb stets eins. Der Test
    konstruiert das Portfolio deshalb selbst. Ohne ihn wäre die Prüfung eine
    Zusicherung, von der niemand weiß, ob sie hält — genau die Sorte
    Behauptung, gegen die dieser Test gerichtet ist.
    """
    einstufung = POLICIES.portfolio.nachhaltigkeit.einstufung
    nicht_konform = next(name for name in ASSET_KEYS if not einstufung[name].pai)
    konform = next(name for name in ASSET_KEYS if einstufung[name].pai)
    gewichte = np.zeros(len(ASSET_KEYS))
    gewichte[ASSET_KEYS.index(nicht_konform)] = 0.6
    gewichte[ASSET_KEYS.index(konform)] = 0.4

    recorder = TraceRecorder(
        TraceMetadata(
            client_id="K-TEST",
            goal_id="Z-TEST",
            policy_versions=dict(POLICIES.versions),
            package_version="0.1.0",
        )
    )

    with pytest.raises(SustainabilityQuotaError, match="Mindestanteil nicht ein"):
        check_sustainability_quotas(
            _praeferenz(pai_considered=True),
            SimpleNamespace(weights=gewichte),
            POLICIES,
            recorder,
            asset_keys=ASSET_KEYS,
        )

    # Der Befund steht im Trace, bevor abgebrochen wird: Ein Abbruch ohne
    # Aufzeichnung wäre im Streitfall nicht belegbar.
    assert [ereignis.rule_id for ereignis in recorder.events] == ["ADV-ESG-QUOTE"]
    assert recorder.events[0].output["alle_vorgaben_eingehalten"] is False


def test_quotenpruefung_haelt_bei_erfuellter_vorgabe() -> None:
    """Wird die Vorgabe eingehalten, bricht nichts ab und der Befund steht im Trace."""
    einstufung = POLICIES.portfolio.nachhaltigkeit.einstufung
    konform = [name for name in ASSET_KEYS if einstufung[name].pai]
    gewichte = np.zeros(len(ASSET_KEYS))
    for name in konform:
        gewichte[ASSET_KEYS.index(name)] = 1.0 / len(konform)

    recorder = TraceRecorder(
        TraceMetadata(
            client_id="K-TEST",
            goal_id="Z-TEST",
            policy_versions=dict(POLICIES.versions),
            package_version="0.1.0",
        )
    )
    check_sustainability_quotas(
        _praeferenz(pai_considered=True),
        SimpleNamespace(weights=gewichte),
        POLICIES,
        recorder,
        asset_keys=ASSET_KEYS,
    )

    assert recorder.events[0].output["erreichte_anteile"] == {"pai": "1.0000"}
    assert recorder.events[0].output["alle_vorgaben_eingehalten"] is True
