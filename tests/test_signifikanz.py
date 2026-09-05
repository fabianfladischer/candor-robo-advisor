"""Tests der Korrektur für multiples Testen und des robusten Sharpe-Tests.

Drei Gruppen:

1. **Mehrfachvergleichskorrektur** — gegen von Hand nachgerechnete Werte. Die
   Rechnung steht als Kommentar im Test, damit sie prüfbar ist.
2. **Robuster Sharpe-Test** — Determinismus, Randfälle und die *Größe* des
   Tests unter der Nullhypothese. Der Größencheck ist der wichtigste: Ein Test,
   dessen Verwerfungsrate unter H0 nicht dem Niveau entspricht, liefert
   p-Werte, die nichts bedeuten.
3. **Trennung von Evaluation und Kundendokument** — Backtest-Signifikanzen
   dürfen nicht in die Geeignetheitserklärung gelangen (Art. 54 Abs. 12 DelVO
   verlangt eine Darstellung der Eignung für den Kunden, keine
   Methodenevaluation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from candor.portfolio.benchmark import PerformanceError, sharpe_difference_test
from candor.portfolio.multiplicity import adjust, familywise_error_rate
from candor.portfolio.robust_sharpe import sharpe_difference_bootstrap
from candor.report.examples import main as beispiele_erzeugen

ROOT = Path(__file__).resolve().parent.parent

# Die fünf p-Werte des Backtests.
P_WERTE = {"1": 0.4418, "2": 0.5319, "3": 0.0402, "4": 0.0491, "5": 0.1160}


# --- Mehrfachvergleichskorrektur -------------------------------------------


def test_familienweise_fehlerrate_von_hand() -> None:
    """FWER = 1 - (1 - alpha)^m.

    Nachrechnung für m = 5, alpha = 0,05:
        0,95^5 = 0,7737809375
        1 − 0,7737809375 = 0,2262190625
    """
    assert familywise_error_rate(5, 0.05) == pytest.approx(0.2262190625, abs=1e-9)
    assert familywise_error_rate(1, 0.05) == pytest.approx(0.05, abs=1e-12)


def test_familienweise_fehlerrate_weist_unsinn_ab() -> None:
    """Familiengröße unter 1 und Niveaus außerhalb (0, 1) sind Fehler."""
    with pytest.raises(ValueError, match="Familiengröße"):
        familywise_error_rate(0)
    with pytest.raises(ValueError, match="Niveau"):
        familywise_error_rate(5, 1.0)


def test_bonferroni_von_hand() -> None:
    """Bonferroni multipliziert jeden p-Wert mit der Familiengröße.

    Nachrechnung (m = 5):
        Klasse 3: 0,0402 · 5 = 0,2010
        Klasse 4: 0,0491 · 5 = 0,2455
        Klasse 5: 0,1160 · 5 = 0,5800
        Klasse 1: 0,4418 · 5 = 2,2090 -> gekappt auf 1,0
        Klasse 2: 0,5319 · 5 = 2,6595 -> gekappt auf 1,0
    """
    bericht = adjust(P_WERTE)
    nach_kennung = {e.label: e for e in bericht.results}
    assert nach_kennung["3"].p_bonferroni == pytest.approx(0.2010, abs=1e-9)
    assert nach_kennung["4"].p_bonferroni == pytest.approx(0.2455, abs=1e-9)
    assert nach_kennung["5"].p_bonferroni == pytest.approx(0.5800, abs=1e-9)
    assert nach_kennung["1"].p_bonferroni == pytest.approx(1.0, abs=1e-12)
    assert bericht.count_bonferroni == 0


def test_holm_von_hand() -> None:
    """Holm-Bonferroni: (m − i + 1) · p_(i), danach monoton steigend.

    Aufsteigend sortiert: 0,0402(3) 0,0491(4) 0,1160(5) 0,4418(1) 0,5319(2)
        Rang 1: 5 · 0,0402 = 0,2010                     -> 0,2010
        Rang 2: 4 · 0,0491 = 0,1964, max(0,2010; …)     -> 0,2010
        Rang 3: 3 · 0,1160 = 0,3480                     -> 0,3480
        Rang 4: 2 · 0,4418 = 0,8836                     -> 0,8836
        Rang 5: 1 · 0,5319 = 0,5319, max(0,8836; …)     -> 0,8836
    """
    nach = {e.label: e for e in adjust(P_WERTE).results}
    assert nach["3"].p_holm == pytest.approx(0.2010, abs=1e-9)
    assert nach["4"].p_holm == pytest.approx(0.2010, abs=1e-9)
    assert nach["5"].p_holm == pytest.approx(0.3480, abs=1e-9)
    assert nach["1"].p_holm == pytest.approx(0.8836, abs=1e-9)
    assert nach["2"].p_holm == pytest.approx(0.8836, abs=1e-9)


def test_benjamini_hochberg_von_hand() -> None:
    """Benjamini-Hochberg: (m / k) · p_(k), von hinten monoton fallend.

    Absteigend über die Ränge:
        Rang 5 (0,5319): 5/5 · 0,5319 = 0,5319          -> 0,5319
        Rang 4 (0,4418): 5/4 · 0,4418 = 0,55225, min    -> 0,5319
        Rang 3 (0,1160): 5/3 · 0,1160 = 0,193333        -> 0,193333
        Rang 2 (0,0491): 5/2 · 0,0491 = 0,12275         -> 0,12275
        Rang 1 (0,0402): 5/1 · 0,0402 = 0,2010, min     -> 0,12275
    """
    nach = {e.label: e for e in adjust(P_WERTE).results}
    assert nach["2"].p_benjamini_hochberg == pytest.approx(0.5319, abs=1e-9)
    assert nach["1"].p_benjamini_hochberg == pytest.approx(0.5319, abs=1e-9)
    assert nach["5"].p_benjamini_hochberg == pytest.approx(0.1933333333, abs=1e-9)
    assert nach["4"].p_benjamini_hochberg == pytest.approx(0.12275, abs=1e-9)
    assert nach["3"].p_benjamini_hochberg == pytest.approx(0.12275, abs=1e-9)


def test_alle_drei_verfahren_beseitigen_die_randnahen_befunde() -> None:
    """Der zentrale Befund: keine der drei Korrekturen lässt einen Test übrig.

    Ohne Korrektur gelten zwei von fünf Klassen als gesichert. Beide p-Werte
    liegen dicht an der Schwelle (0,0402 und 0,0491) und überstehen keine der
    drei Korrekturen — auch nicht Benjamini-Hochberg, das mildeste Verfahren.
    """
    bericht = adjust(P_WERTE)
    assert bericht.count_raw == 2
    assert bericht.count_bonferroni == 0
    assert bericht.count_holm == 0
    assert bericht.count_benjamini_hochberg == 0


def test_holm_ist_nie_strenger_als_bonferroni() -> None:
    """Holm-Bonferroni verwirft mindestens so oft wie Bonferroni.

    Das ist die definierende Eigenschaft des Verfahrens: gleiche Kontrolle der
    FWER bei höherer Trennschärfe.
    """
    rng = np.random.default_rng(4242)
    for _ in range(200):
        werte = {str(i): float(rng.uniform(0.0, 0.3)) for i in range(6)}
        bericht = adjust(werte)
        assert bericht.count_holm >= bericht.count_bonferroni
        for eintrag in bericht.results:
            assert eintrag.p_holm <= eintrag.p_bonferroni + 1e-12


def test_benjamini_hochberg_ist_nie_strenger_als_holm() -> None:
    """BH kontrolliert die FDR und verwirft daher mindestens so oft wie Holm."""
    rng = np.random.default_rng(31337)
    for _ in range(200):
        werte = {str(i): float(rng.uniform(0.0, 0.3)) for i in range(6)}
        bericht = adjust(werte)
        assert bericht.count_benjamini_hochberg >= bericht.count_holm


def test_leere_familie_ist_ein_fehler() -> None:
    """Eine leere Testfamilie ist keine gültige Eingabe (Prinzip P5)."""
    with pytest.raises(ValueError, match="leer"):
        adjust({})


def test_p_wert_ausserhalb_des_intervalls_ist_ein_fehler() -> None:
    """p-Werte außerhalb von [0, 1] werden laut abgewiesen, nicht gekappt."""
    with pytest.raises(ValueError, match="außerhalb"):
        adjust({"a": 1.5})


# --- Robuster Sharpe-Test ---------------------------------------------------


def _paar(rng: np.random.Generator, months: int = 156) -> tuple[np.ndarray, np.ndarray]:
    """Erzeugt zwei korrelierte Reihen mit identischer Sharpe Ratio.

    Args:
        rng: Zufallsgenerator.
        months: Zahl der Monate.

    Returns:
        Zwei Renditereihen.
    """
    zug = rng.multivariate_normal([0.006, 0.006], [[0.0016, 0.0013], [0.0013, 0.0016]], size=months)
    return zug[:, 0], zug[:, 1]


def test_bootstrap_ist_deterministisch() -> None:
    """Gleicher Seed und gleiche Daten ergeben denselben p-Wert (Prinzip P1)."""
    a, b = _paar(np.random.default_rng(7))
    erst = sharpe_difference_bootstrap(a, b, 0.0, 12, seed=99, iterations=499)
    zweit = sharpe_difference_bootstrap(a, b, 0.0, 12, seed=99, iterations=499)
    assert erst.p_value == zweit.p_value
    assert erst.studentized_statistic == zweit.studentized_statistic


def test_verschiedene_seeds_ergeben_verschiedene_p_werte() -> None:
    """Ohne diesen Test bliebe unbemerkt, wenn der Seed nicht wirkt."""
    a, b = _paar(np.random.default_rng(8))
    erst = sharpe_difference_bootstrap(a, b, 0.0, 12, seed=1, iterations=499)
    zweit = sharpe_difference_bootstrap(a, b, 0.0, 12, seed=2, iterations=499)
    assert erst.p_value != zweit.p_value


def test_identische_reihen_ergeben_keinen_unterschied() -> None:
    """Zwei identische Reihen haben dieselbe Sharpe Ratio; p muss groß sein."""
    a, _ = _paar(np.random.default_rng(11))
    ergebnis = sharpe_difference_bootstrap(a, a.copy(), 0.0, 12, seed=5, iterations=499)
    assert ergebnis.difference == pytest.approx(0.0, abs=1e-12)
    assert ergebnis.p_value > 0.5


def test_grosse_des_tests_unter_der_nullhypothese() -> None:
    """Die Verwerfungsrate unter H0 muss dem Niveau entsprechen.

    **Der wichtigste Test dieser Gruppe.** Ein Verfahren, dessen Größe nicht
    stimmt, liefert p-Werte ohne Bedeutung — unabhängig davon, wie robust es
    gegen Verteilungsannahmen ist.

    Geprüft wird gegen ein großzügiges Band: 120 Wiederholungen erlauben nur
    eine grobe Aussage (Standardfehler rund 2 Prozentpunkte). Der Test soll
    einen groben Fehler in der Zentrierung oder Studentisierung aufdecken.

    **Er bestimmt die Größe des Verfahrens ausdrücklich nicht.** Auch der
    umfangreichere Lauf mit 400 Wiederholungen reicht dafür nicht: Dort ergibt
    sich z = 0,73 gegen den Sollwert von 5 %
    (p = 0,463); für 80 % Power auf die beobachtete Abweichung von
    0,8 Prozentpunkten wären rund 5 825 Wiederholungen nötig. Ein hier
    bestandener Test belegt daher, dass **kein grober** Fehler vorliegt — nicht,
    dass die Größe stimmt.
    """
    rng = np.random.default_rng(20260727)
    wiederholungen = 120
    verworfen = 0
    for lauf in range(wiederholungen):
        a, b = _paar(rng)
        ergebnis = sharpe_difference_bootstrap(
            a, b, 0.0, 12, seed=6000 + lauf, iterations=299, block_length=6
        )
        verworfen += ergebnis.p_value < 0.05
    rate = verworfen / wiederholungen
    assert 0.005 < rate < 0.15, (
        f"Verwerfungsrate unter H0 liegt bei {rate:.3f}; erwartet wird ein Wert "
        "nahe 0,05. Ein Wert weit darüber deutet auf einen Fehler in der "
        "Zentrierung an der Originaldifferenz hin, ein Wert nahe null auf einen "
        "zu großen Standardfehler."
    )


def test_bootstrap_und_parametrischer_test_stimmen_auf_normalen_daten_ueberein() -> None:
    """Auf i.i.d.-normalen Daten dürfen beide Verfahren nicht weit auseinanderliegen.

    Dort sind die Annahmen des parametrischen Tests erfüllt; ein grober
    Unterschied wäre ein Hinweis auf einen Implementierungsfehler im
    Bootstrap, nicht auf Robustheit.
    """
    rng = np.random.default_rng(2024)
    for _ in range(5):
        zug = rng.multivariate_normal(
            [0.009, 0.005], [[0.0016, 0.0011], [0.0011, 0.0018]], size=240
        )
        a, b = zug[:, 0], zug[:, 1]
        parametrisch = sharpe_difference_test(a, b, 0.0, 12).p_value
        robust = sharpe_difference_bootstrap(
            a, b, 0.0, 12, seed=77, iterations=999, block_length=3
        ).p_value
        assert abs(parametrisch - robust) < 0.20


def test_bootstrap_weist_unzulaessige_eingaben_ab() -> None:
    """Fail loud statt stiller Notbehelf (Prinzip P5)."""
    a, b = _paar(np.random.default_rng(13))
    with pytest.raises(PerformanceError, match="verschiedene Länge"):
        sharpe_difference_bootstrap(a, b[:-1], 0.0, 12, seed=1)
    with pytest.raises(PerformanceError, match="mindestens"):
        sharpe_difference_bootstrap(a[:10], b[:10], 0.0, 12, seed=1)
    with pytest.raises(PerformanceError, match="Blocklänge"):
        sharpe_difference_bootstrap(a, b, 0.0, 12, seed=1, block_length=0)
    with pytest.raises(PerformanceError, match="Ziehungen"):
        sharpe_difference_bootstrap(a, b, 0.0, 12, seed=1, iterations=0)


def test_ergebnis_nennt_seed_und_verfahren() -> None:
    """Das Ergebnis muss sich ohne den Code nachvollziehen lassen (Prinzip P2)."""
    a, b = _paar(np.random.default_rng(17))
    ergebnis = sharpe_difference_bootstrap(a, b, 0.0, 12, seed=4711, iterations=299)
    assert ergebnis.seed == 4711
    assert "4711" in ergebnis.label
    assert "Ledoit-Wolf" in ergebnis.label
    assert str(ergebnis.iterations) in ergebnis.label


# --- Trennung von Evaluation und Kundendokument -----------------------------


def test_keine_backtest_signifikanz_in_den_kundendokumenten(tmp_path: Path) -> None:
    """Backtest-p-Werte dürfen nicht in die Geeignetheitserklärung gelangen.

    Art. 54 Abs. 12 DelVO (EU) 2017/565 verlangt eine Darstellung, **warum die
    Empfehlung für diesen Kunden geeignet ist** — nicht die Methodenevaluation
    des Anbieters. Ein Signifikanztest über die Güte des Optimierungsverfahrens
    gehört in die Methodendokumentation, nicht in ein Kundendokument.

    Geprüft werden die Dokumente, die das System aus den drei Beispielprofilen
    erzeugt – nicht abgelegte Dateien. Damit gilt der Nachweis für den jeweils
    aktuellen Stand der Textbausteine.

    Args:
        tmp_path: Temporäres Verzeichnis für die erzeugten Dokumente.
    """
    pypdf = pytest.importorskip("pypdf")

    assert beispiele_erzeugen(["--out", str(tmp_path)]) == 0
    dokumente = sorted(tmp_path.glob("*.pdf"))
    assert dokumente, "Keine Beispieldokumente erzeugt."

    verboten = (
        "p-Wert",
        "Signifikanzniveau",
        "signifikant",
        "Jobson",
        "Memmel",
        "Ledoit-Wolf-Bootstrap",
        "Backtest",
        "Nullhypothese",
        "Teststatistik",
        "1/N",
    )
    for pfad in dokumente:
        leser = pypdf.PdfReader(str(pfad))
        text = " ".join(" ".join((seite.extract_text() or "").split()) for seite in leser.pages)
        gefunden = [begriff for begriff in verboten if begriff in text]
        assert not gefunden, (
            f"{pfad.name} enthält Begriffe der Methodenevaluation: {gefunden}. "
            "Backtest-Signifikanzen gehören nach Art. 54 Abs. 12 DelVO nicht in "
            "ein Kundendokument, sondern in die Methodendokumentation."
        )
