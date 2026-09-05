"""Tests der Simulation.

Schwerpunkte:

* **Erhaltung der Korrelationsstruktur** — der entscheidende Konstruktionspunkt
  des Block-Bootstrap. Geprüft wird nicht nur, dass die Korrelation stimmt,
  sondern auch, dass eine getrennte Ziehung je Assetklasse sie zerstören würde.
* **Erwartungstreue** — der zirkuläre Bootstrap trifft das historische Mittel,
  der nicht zirkuläre nicht. Der Test hält die gefundene Verzerrung fest,
  damit sie nicht unbemerkt zurückkehrt.
* **Determinismus** — gleicher Seed, gleiches Ergebnis (Prinzip P1).
* **Blockstruktur** — aufeinanderfolgende Monate innerhalb eines Blocks stammen
  auch in der Historie aus aufeinanderfolgenden Monaten.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from candor import format_de
from candor.data.policy_loader import load_policy_set
from candor.data.universe_meta import ASSET_KEYS
from candor.domain.errors import PolicyError
from candor.domain.policy import PolicySet
from candor.engine.convergence import (
    QUICK_PATH_COUNTS,
    convergence_study,
    load_history,
    method_comparison,
    render_block,
    write_evaluation,
)
from candor.engine.convergence import main as convergence_main
from candor.engine.simulate import (
    BATCH_SIZE,
    BlockBootstrapSampler,
    FloatArray,
    IIDNormalSampler,
    simulate,
)
from candor.portfolio.universe import (
    _fill_within_bounds as fill_within_bounds,
)
from candor.portfolio.universe import (
    equal_weights,
    equity_share,
    reference_weights,
    weights_as_mapping,
)
from candor.report.figures import (
    build_figure_3,
    fan_chart,
    german_decimal,
    german_percent,
    german_thousands,
)
from candor.report.figures import main as figures_main

#: Pfadzahl der schnellen Tests.
TEST_PATHS = 400


@pytest.fixture(scope="module")
def policies() -> PolicySet:
    """Lädt das ausgelieferte Regelwerk einmal je Modul.

    Returns:
        Das geprüfte Regelwerk.
    """
    return load_policy_set()


@pytest.fixture(scope="module")
def history(policies: PolicySet) -> FloatArray:
    """Lädt die vollständige historische Renditematrix.

    Args:
        policies: Das geladene Regelwerk.

    Returns:
        Renditematrix der Form ``(monate, assetklassen)``.
    """
    return load_history(policies)


def simuliere(sampler: object, policies: PolicySet, **abweichungen: object) -> object:
    """Rechnet den Standardfall der Tests.

    Args:
        sampler: Ziehungsverfahren.
        policies: Das geladene Regelwerk.
        **abweichungen: Abweichungen von den Vorgabewerten.

    Returns:
        Das Simulationsergebnis.
    """
    felder: dict[str, object] = {
        "initial_capital": 10_000.0,
        "monthly_contribution": 300.0,
        "horizon_months": 120,
        "target_amount": 60_000.0,
        "policy": policies.simulation,
        "paths": TEST_PATHS,
        "asset_keys": ASSET_KEYS,
        "track_bands": False,
    }
    felder.update(abweichungen)
    gewichte = reference_weights(3, policies.classes, ASSET_KEYS)
    return simulate(sampler, gewichte, **felder)  # type: ignore[arg-type]


# --- Der entscheidende Punkt: Korrelationsstruktur -------------------------


def test_korrelationsstruktur_bleibt_erhalten(policies: PolicySet, history: FloatArray) -> None:
    """Die simultane Ziehung erhält die Korrelation zwischen den Assetklassen.

    Das ist der Konstruktionskern des Verfahrens. Gezogen wird ein Zeitindex für
    alle acht Assetklassen gemeinsam; die gezogenen Monate sind damit dieselben
    realen Monate wie in der Historie.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    gezogen = sampler.sample(np.random.default_rng(20260726), 3000, 60)
    flach = gezogen.reshape(-1, gezogen.shape[2])

    historisch = np.corrcoef(history, rowvar=False)
    simuliert = np.corrcoef(flach, rowvar=False)

    assert np.abs(simuliert - historisch).max() < 0.05


def test_getrennte_ziehung_wuerde_die_korrelation_zerstoeren(
    policies: PolicySet, history: FloatArray
) -> None:
    """Gegenprobe: Eine Ziehung je Assetklasse zerstört die Korrelation.

    Der Test belegt, dass die simultane Ziehung tatsächlich nötig ist und nicht
    bloß eine Vorsichtsmaßnahme. Er ist die Begründung des Verfahrens in
    Testform.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    rng = np.random.default_rng(20260726)
    laenge, assets = history.shape
    getrennt = np.empty((3000 * 60, assets))
    for spalte in range(assets):
        index = rng.integers(0, laenge, size=3000 * 60)
        getrennt[:, spalte] = history[index, spalte]

    historisch = np.corrcoef(history, rowvar=False)
    zerstoert = np.corrcoef(getrennt, rowvar=False)
    ausserhalb = ~np.eye(assets, dtype=bool)

    # Die historische Korrelation zwischen den Aktienklassen ist hoch; nach
    # getrennter Ziehung ist sie praktisch null.
    assert np.abs(historisch[ausserhalb]).max() > 0.5
    assert np.abs(zerstoert[ausserhalb]).max() < 0.1
    assert policies.simulation.blocklaenge_monate > 0


# --- Erwartungstreue: die gefundene Verzerrung ----------------------------


def test_zirkulaerer_bootstrap_ist_erwartungstreu(policies: PolicySet, history: FloatArray) -> None:
    """Der zirkuläre Bootstrap trifft das historische Mittel.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    gezogen = sampler.sample(np.random.default_rng(1), 4000, 120)

    assert np.abs(gezogen.mean(axis=(0, 1)) - history.mean(axis=0)).max() < 0.0004


def test_nicht_zirkulaerer_bootstrap_ueberschaetzt(
    policies: PolicySet, history: FloatArray
) -> None:
    """Ohne Umlauf sind die Randmonate untergewichtet — hier die Krisenmonate 2008.

    Der Test hält die gefundene Verzerrung fest. Er schlägt fehl,
    sobald jemand die Vorgabe auf ``circular=False`` zurückstellt, ohne die
    Folgen zu bedenken.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    gewichte = reference_weights(3, policies.classes, ASSET_KEYS)
    zirkulaer = BlockBootstrapSampler(history, 12)
    ohne_umlauf = BlockBootstrapSampler(history, 12, circular=False)

    historisch = float((history @ gewichte).mean())
    mit = float((zirkulaer.sample(np.random.default_rng(2), 8000, 120) @ gewichte).mean())
    ohne = float((ohne_umlauf.sample(np.random.default_rng(2), 8000, 120) @ gewichte).mean())

    assert abs(mit - historisch) < 0.00005
    assert ohne - historisch > 0.0002


def test_blockstruktur_bleibt_zusammenhaengend(history: FloatArray) -> None:
    """Aufeinanderfolgende Monate eines Blocks stammen aus aufeinanderfolgenden Monaten.

    Geprüft über eine künstliche Historie, in der jede Zeile ihren eigenen Index
    trägt: Die gezogene Reihe muss dann innerhalb eines Blocks um genau eins
    steigen.

    Args:
        history: Historische Renditematrix, hier nur für die Form.
    """
    kuenstlich = np.arange(120, dtype=np.float64).reshape(-1, 1)
    sampler = BlockBootstrapSampler(kuenstlich, 12)
    gezogen = sampler.sample(np.random.default_rng(3), 50, 24)[:, :, 0]

    differenz = np.diff(gezogen[:, :12], axis=1)
    # Innerhalb eines Blocks steigt der Index um eins — außer beim Umlauf, der
    # von 119 auf 0 springt.
    assert np.all((differenz == 1) | (differenz == -119))
    assert history.shape[1] == len(ASSET_KEYS)


# --- Sampler-Prüfungen ----------------------------------------------------


def test_normalsampler_trifft_momente(history: FloatArray) -> None:
    """Die Normalziehung übernimmt Erwartungswerte und Kovarianz.

    Args:
        history: Historische Renditematrix.
    """
    sampler = IIDNormalSampler(history)
    gezogen = sampler.sample(np.random.default_rng(4), 4000, 60)
    flach = gezogen.reshape(-1, gezogen.shape[2])

    assert np.abs(flach.mean(axis=0) - history.mean(axis=0)).max() < 0.0005
    assert (
        np.abs(np.corrcoef(flach, rowvar=False) - np.corrcoef(history, rowvar=False)).max() < 0.05
    )


def test_zu_kurze_historie_wird_abgewiesen() -> None:
    """Eine Historie kürzer als ein Block ist unzulässig."""
    with pytest.raises(ValueError, match="kürzer"):
        BlockBootstrapSampler(np.zeros((6, 3)), 12)


def test_fehlende_werte_werden_abgewiesen() -> None:
    """Fehlende Werte machen eine simultane Ziehung unmöglich."""
    kaputt = np.zeros((60, 3))
    kaputt[10, 1] = np.nan

    with pytest.raises(ValueError, match="fehlende"):
        BlockBootstrapSampler(kaputt, 12)
    with pytest.raises(ValueError, match="fehlende"):
        IIDNormalSampler(kaputt)


def test_unzulaessige_blocklaenge_wird_abgewiesen() -> None:
    """Eine Blocklänge unter eins ist unzulässig."""
    with pytest.raises(ValueError, match="positiv"):
        BlockBootstrapSampler(np.zeros((60, 3)), 0)


# --- Determinismus --------------------------------------------------------


def test_gleicher_seed_gleiches_ergebnis(policies: PolicySet, history: FloatArray) -> None:
    """Zwei Läufe mit demselben Seed liefern identische Endwerte (Prinzip P1).

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    erst = simuliere(sampler, policies, seed=42)
    zweit = simuliere(sampler, policies, seed=42)

    assert np.array_equal(erst.final_values, zweit.final_values)  # type: ignore[attr-defined]
    assert erst.success_probability == zweit.success_probability  # type: ignore[attr-defined]


def test_anderer_seed_anderes_ergebnis(policies: PolicySet, history: FloatArray) -> None:
    """Ein anderer Seed liefert eine andere Stichprobe.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    erst = simuliere(sampler, policies, seed=42)
    zweit = simuliere(sampler, policies, seed=43)

    assert not np.array_equal(erst.final_values, zweit.final_values)  # type: ignore[attr-defined]


def test_pfadzahl_ueber_der_blockgroesse(policies: PolicySet, history: FloatArray) -> None:
    """Auch über mehrere Rechenblöcke hinweg entsteht die volle Pfadzahl.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    ergebnis = simuliere(sampler, policies, paths=BATCH_SIZE + 137, horizon_months=24)

    assert ergebnis.final_values.shape == (BATCH_SIZE + 137,)  # type: ignore[attr-defined]


# --- Kosten und Mechanik --------------------------------------------------


def test_kosten_senken_den_endwert(policies: PolicySet, history: FloatArray) -> None:
    """Ohne Kosten fällt der Endwert höher aus als mit.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    kostenfrei = policies.simulation.model_copy(
        update={
            "kosten": {
                name: eintrag.model_copy(update={"ter_jaehrlich": 0, "spread_einmalig": 0})
                for name, eintrag in policies.simulation.kosten.items()
            }
        }
    )

    mit = simuliere(sampler, policies, seed=7)
    ohne = simuliere(sampler, policies, seed=7, policy=kostenfrei)

    assert float(np.median(ohne.final_values)) > float(np.median(mit.final_values))  # type: ignore[attr-defined]


def test_hoehere_sparrate_erhoeht_den_endwert(policies: PolicySet, history: FloatArray) -> None:
    """Eine höhere Sparrate führt zu einem höheren Endwert.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    klein = simuliere(sampler, policies, seed=8, monthly_contribution=100.0)
    gross = simuliere(sampler, policies, seed=8, monthly_contribution=600.0)

    assert float(np.median(gross.final_values)) > float(np.median(klein.final_values))  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("feld", "wert", "muster"),
    [
        ("horizon_months", 0, "mindestens einen Monat"),
        ("initial_capital", -1.0, "nicht negativ"),
    ],
)
def test_unzulaessige_eingaben_werden_abgewiesen(
    policies: PolicySet, history: FloatArray, feld: str, wert: object, muster: str
) -> None:
    """Unzulässige Eingaben brechen ab, statt still gerechnet zu werden.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
        feld: Zu verfälschendes Feld.
        wert: Unzulässiger Wert.
        muster: Erwartetes Fehlermuster.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)

    with pytest.raises(ValueError, match=muster):
        simuliere(sampler, policies, **{feld: wert})


def test_gewichte_muessen_sich_zu_eins_addieren(policies: PolicySet, history: FloatArray) -> None:
    """Gewichte, die sich nicht zu eins addieren, sind unzulässig.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)

    with pytest.raises(ValueError, match="zu eins addieren"):
        simulate(
            sampler,
            np.full(len(ASSET_KEYS), 0.5),
            initial_capital=1000.0,
            monthly_contribution=0.0,
            horizon_months=12,
            target_amount=2000.0,
            policy=policies.simulation,
            paths=10,
            asset_keys=ASSET_KEYS,
        )


# --- Ergebnisobjekt -------------------------------------------------------


def test_kennzahlen_sind_stimmig(policies: PolicySet, history: FloatArray) -> None:
    """Perzentile, Erfolgs- und Shortfall-Wahrscheinlichkeit passen zusammen.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    ergebnis = simuliere(sampler, policies, seed=9, track_bands=True)

    perzentile = ergebnis.percentiles()  # type: ignore[attr-defined]
    assert perzentile[5] < perzentile[25] < perzentile[50] < perzentile[75] < perzentile[95]
    assert 0.0 <= ergebnis.success_probability <= 1.0  # type: ignore[attr-defined]
    assert np.isclose(  # type: ignore[attr-defined]
        ergebnis.success_probability + ergebnis.shortfall_probability, 1.0
    )
    assert ergebnis.median_max_drawdown <= 0.0  # type: ignore[attr-defined]
    assert ergebnis.worst_max_drawdown <= ergebnis.median_max_drawdown  # type: ignore[attr-defined]
    assert set(ergebnis.value_bands) == {5, 25, 50, 75, 95}  # type: ignore[attr-defined]


def test_bedingter_fehlbetrag_mittelt_nur_ueber_misserfolge(
    policies: PolicySet, history: FloatArray
) -> None:
    """Der erwartete Fehlbetrag mittelt ausschließlich über verfehlte Pfade.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    ergebnis = simuliere(sampler, policies, seed=10, target_amount=200_000.0)

    verfehlt = ergebnis.final_values[ergebnis.final_values < 200_000.0]  # type: ignore[attr-defined]
    erwartet = float(np.mean(200_000.0 - verfehlt))

    assert ergebnis.expected_shortfall == pytest.approx(erwartet)  # type: ignore[attr-defined]
    assert ergebnis.expected_shortfall > 0  # type: ignore[attr-defined]


def test_ohne_misserfolg_ist_der_fehlbetrag_null(policies: PolicySet, history: FloatArray) -> None:
    """Erreicht jeder Pfad das Ziel, ist der Fehlbetrag null und nicht ``NaN``.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    ergebnis = simuliere(sampler, policies, seed=11, target_amount=1.0)

    assert ergebnis.success_probability == 1.0  # type: ignore[attr-defined]
    assert ergebnis.expected_shortfall == 0.0  # type: ignore[attr-defined]


def test_zusammenfassung_ist_vollstaendig(policies: PolicySet, history: FloatArray) -> None:
    """Die Zusammenfassung führt alle Kennzahlen für Trace und Bericht.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    zusammenfassung = simuliere(sampler, policies, seed=12).as_summary()  # type: ignore[attr-defined]

    for schluessel in (
        "verfahren",
        "pfade",
        "seed",
        "erfolgswahrscheinlichkeit",
        "erwarteter_fehlbetrag_im_misserfolg",
        "perzentile",
        "median_max_drawdown",
    ):
        assert schluessel in zusammenfassung


# --- Referenzallokation ---------------------------------------------------


@pytest.mark.parametrize("klasse", [1, 2, 3, 4, 5])
def test_referenzallokation_haelt_die_baender_ein(policies: PolicySet, klasse: int) -> None:
    """Die Allokation liegt in allen Bandbreiten und summiert sich zu eins (I5).

    Args:
        policies: Das geladene Regelwerk.
        klasse: Geprüfte Risikoklasse.
    """
    gewichte = reference_weights(klasse, policies.classes, ASSET_KEYS)
    definition = policies.classes.risikoklassen[klasse]

    assert np.isclose(gewichte.sum(), 1.0)
    assert (gewichte >= -1e-12).all()
    for name, gewicht in zip(ASSET_KEYS, gewichte, strict=True):
        band = definition.einzelwerte[name]
        assert float(band.min) - 1e-9 <= gewicht <= float(band.max) + 1e-9


def test_aktienquote_steigt_ueber_die_klassen(policies: PolicySet) -> None:
    """Invariante I6 gilt auch für die Referenzallokation.

    Args:
        policies: Das geladene Regelwerk.
    """
    quoten = [
        equity_share(
            reference_weights(k, policies.classes, ASSET_KEYS), policies.classes, ASSET_KEYS
        )
        for k in range(1, 6)
    ]

    assert quoten == sorted(quoten)
    assert quoten[0] < quoten[-1]


def test_unbekannte_risikoklasse_wird_abgewiesen(policies: PolicySet) -> None:
    """Eine Risikoklasse außerhalb 1 bis 5 ist unzulässig.

    Args:
        policies: Das geladene Regelwerk.
    """
    with pytest.raises(PolicyError, match="Unbekannte Risikoklasse"):
        reference_weights(9, policies.classes, ASSET_KEYS)


def test_gleichgewichtetes_portfolio() -> None:
    """Das 1/N-Portfolio verteilt gleichmäßig und summiert sich zu eins."""
    gewichte = equal_weights(ASSET_KEYS)

    assert np.isclose(gewichte.sum(), 1.0)
    assert len(set(gewichte.tolist())) == 1
    with pytest.raises(ValueError, match="mindestens eine Assetklasse"):
        equal_weights(())


def test_gewichte_als_zuordnung(policies: PolicySet) -> None:
    """Die Zuordnung für Trace und Bericht ist vollständig benannt.

    Args:
        policies: Das geladene Regelwerk.
    """
    zuordnung = weights_as_mapping(reference_weights(3, policies.classes, ASSET_KEYS), ASSET_KEYS)

    assert set(zuordnung) == set(ASSET_KEYS)


# --- Konvergenzanalyse ----------------------------------------------------


def test_standardfehler_faellt_mit_der_pfadzahl(policies: PolicySet, history: FloatArray) -> None:
    """Mehr Pfade führen zu einer stabileren Schätzung.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    punkte = convergence_study(policies, history, QUICK_PATH_COUNTS, repetitions=5)
    fehler = [punkt.empirical_se for punkt in punkte]

    assert len(punkte) == len(QUICK_PATH_COUNTS)
    assert fehler[0] > fehler[-1]
    for punkt in punkte:
        # Der empirische Fehler bleibt in der Größenordnung des theoretischen.
        assert punkt.empirical_se < punkt.theoretical_se * 3


def test_verfahrensvergleich_deckt_alle_klassen_ab(
    policies: PolicySet, history: FloatArray
) -> None:
    """Der Vergleich liefert je Risikoklasse eine Zeile.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    vergleiche = method_comparison(policies, history, paths=500)

    assert [eintrag.risk_class for eintrag in vergleiche] == [1, 2, 3, 4, 5]
    for eintrag in vergleiche:
        assert eintrag.bootstrap_p5 > 0
        assert eintrag.normal_p5 > 0


def test_konvergenzbericht_ist_vollstaendig(policies: PolicySet, history: FloatArray) -> None:
    """Der erzeugte Abschnitt enthält beide Tabellen und einen Befund.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    punkte = convergence_study(policies, history, (100, 500), repetitions=3)
    vergleiche = method_comparison(policies, history, paths=300)
    block = render_block(punkte, vergleiche, policies)

    assert "BEGIN:KONVERGENZ" in block
    assert "END:KONVERGENZ" in block
    assert "SE empirisch" in block
    assert "Block-Bootstrap gegen i.i.d.-Normalziehung" in block
    assert "**Befund:**" in block
    # Deutsches Dezimalkomma in allen Prozentangaben: Ein Punkt
    # als Dezimaltrennzeichen darf nirgends vorkommen.
    assert re.search(r"\d,\d{4} %", block) is not None
    assert re.search(r"\d\.\d{4} %", block) is None


# --- Abbildung ------------------------------------------------------------


def test_abbildung_drei_entsteht(out_dir: Path) -> None:
    """Abbildung 3 wird als PNG mit zweizeiliger Unterschrift erzeugt.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    bild, unterschrift = build_figure_3(out_dir)

    assert bild.is_file()
    assert bild.stat().st_size > 10_000
    zeilen = unterschrift.read_text(encoding="utf-8").splitlines()
    assert len(zeilen) == 2
    assert zeilen[0].startswith("Abb. 3:")
    assert zeilen[1] == "Quelle: Eigene Darstellung."


def test_achsenformat_verwendet_deutsches_dezimalkomma() -> None:
    """Die Achsenbeschriftung folgt dem deutschen Zahlformat.

    Getrennt wird einheitlich mit geschütztem Leerzeichen nach DIN 5008.
    Zuvor setzte `german_thousands` als einzige Stelle des Projekts einen Punkt;
    seit der Zusammenführung in `candor.format_de` gilt überall dieselbe Regel.
    Das Trennzeichen wird aus dem Modul bezogen und hier nicht wiederholt.
    """
    assert german_thousands(250000) == f"250{format_de.THOUSANDS_SEPARATOR}000"
    assert german_decimal(12.5) == "12,5"
    assert german_percent(0.075) == "7,5 %"


def test_fehlende_baender_werden_abgewiesen(policies: PolicySet, history: FloatArray) -> None:
    """Ohne mitgeschriebenen Verlauf lässt sich kein Fächerdiagramm zeichnen.

    Args:
        policies: Das geladene Regelwerk.
        history: Historische Renditematrix.
    """
    sampler = BlockBootstrapSampler(history, policies.simulation.blocklaenge_monate)
    ohne_baender = simuliere(sampler, policies, seed=13, track_bands=False)

    with pytest.raises(ValueError, match="keine Perzentilbänder"):
        fan_chart(ohne_baender, Path("/tmp/unbenutzt.png"))  # type: ignore[arg-type]  # noqa: S108


def test_abbildungs_kommandozeile(out_dir: Path) -> None:
    """Der Kommandozeilenaufruf erzeugt Bild und Unterschrift.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    assert figures_main(["--out", str(out_dir), "--seed", "1"]) == 0
    assert (out_dir / "abb3_faecherdiagramm.png").is_file()


# --- Schreiben in die Auswertungsdatei -------------------------------------


def test_konvergenzbericht_ersetzt_den_vorhandenen_abschnitt(out_dir: Path) -> None:
    """Ein wiederholter Lauf ersetzt den Abschnitt, statt ihn anzuhängen.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    ziel = out_dir / "auswertung.md"
    ziel.write_text(
        "# Kopf\n\n<!-- BEGIN:KONVERGENZ -->\nalt\n<!-- END:KONVERGENZ -->\n\nRest\n",
        encoding="utf-8",
    )

    write_evaluation("<!-- BEGIN:KONVERGENZ -->\nneu\n<!-- END:KONVERGENZ -->", ziel)
    inhalt = ziel.read_text(encoding="utf-8")

    assert "neu" in inhalt
    assert "alt" not in inhalt
    assert inhalt.count("BEGIN:KONVERGENZ") == 1
    assert inhalt.startswith("# Kopf")
    assert inhalt.rstrip().endswith("Rest")


def test_konvergenzbericht_fuegt_bei_fehlenden_markierungen_ein(out_dir: Path) -> None:
    """Ohne Markierungen wird der bestehende Abschnitt 2.1 ersetzt.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    ziel = out_dir / "EVALUATION_ohne_marker.md"
    ziel.write_text(
        "# Kopf\n\n### 2.1 Konvergenz über die Pfadzahl\n\nalt\n\n---\n\nRest\n",
        encoding="utf-8",
    )

    write_evaluation("NEUER BLOCK", ziel)
    inhalt = ziel.read_text(encoding="utf-8")

    assert "NEUER BLOCK" in inhalt
    assert "alt" not in inhalt
    assert "Rest" in inhalt


def test_fehlende_zieldatei_wird_abgewiesen(out_dir: Path) -> None:
    """Eine nicht vorhandene Zieldatei führt zum Fehler.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    with pytest.raises(FileNotFoundError, match="Zieldatei fehlt"):
        write_evaluation("x", out_dir / "gibt_es_nicht.md")


def test_konvergenz_kommandozeile_im_schnelldurchlauf() -> None:
    """Der Schnelldurchlauf läuft durch und schreibt nichts."""
    assert convergence_main(["--quick"]) == 0


# --- Referenzallokation: Fehlerzweige -------------------------------------


def test_nicht_verteilbare_summe_wird_abgewiesen() -> None:
    """Eine Summe außerhalb der Bänder ist nicht verteilbar."""
    with pytest.raises(PolicyError, match="außerhalb der Bandbreiten"):
        fill_within_bounds(1.0, np.array([0.0, 0.0]), np.array([0.2, 0.2]))


def test_feste_baender_werden_unveraendert_uebernommen() -> None:
    """Ohne Spielraum bleibt es bei den Untergrenzen."""
    verteilung = fill_within_bounds(0.6, np.array([0.3, 0.3]), np.array([0.3, 0.3]))

    assert verteilung.tolist() == [0.3, 0.3]
