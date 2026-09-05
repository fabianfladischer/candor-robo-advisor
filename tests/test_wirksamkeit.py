"""Tests des Wirksamkeitsnachweises.

Schwerpunkte:

* **Stresstest.** Die Rechnung selbst (Drawdown, Erholungsdauer) gegen von Hand
  nachrechenbare Reihen, und die Befunde gegen stille Veränderung.
* **Der Befund von 2022 wird festgehalten.** Dass die Verlusterwartung der
  Klassen 1 und 2 gebrochen wurde, ist das wichtigste Ergebnis. Ein
  Test hält es fest — nicht damit es so bleibt, sondern damit eine Änderung
  auffällt und begründet werden muss.
* **Sensitivität.** Die Ein-Faktor-Analyse und die Aussage, dass die beiden
  Deckel-Kriterien den größten Hebel tragen.
* **Verkürzte Fenster.** Der Snapshot beginnt nach dem Start der Finanzkrise;
  der Test belegt, dass die Verkürzung ausgewiesen und nicht verschwiegen wird.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from candor.data.loader import load_returns
from candor.data.policy_loader import load_policy_set
from candor.data.universe_meta import ASSET_KEYS
from candor.domain.goal import GoalType
from candor.domain.policy import PolicySet
from candor.engine.stress import (
    CRISIS_WINDOWS,
    MAX_RECOVERY_MONTHS,
    CrisisWindow,
    StressTestError,
    broken_promises,
    max_drawdown,
    run_all,
    run_stress_test,
)
from candor.goals.model import build_goal
from candor.portfolio.universe import reference_weights
from candor.profiling.sensitivity import (
    AXIS_CAPACITY,
    AXIS_KNOWLEDGE,
    AXIS_TOLERANCE,
    SensitivityError,
    SensitivityReport,
    analyse,
    baseline_clients,
)
from candor.report.figures import SENSITIVITY_DATE
from candor.report.wirksamkeit import (
    SENSITIVITY_END,
    SENSITIVITY_START,
    STRESS_END,
    STRESS_START,
    build_blocks,
    render_sensitivity,
    render_stress,
    write_evaluation,
)

#: Das Regelwerk wird einmal geladen.
POLICIES: PolicySet = load_policy_set()

#: Alle Risikoklassen.
RISK_CLASSES = tuple(sorted(POLICIES.classes.risikoklassen))


@pytest.fixture(scope="module")
def renditen() -> pd.DataFrame:
    """Lädt die Monatsrenditen mit Datumsindex.

    Returns:
        Die Renditen des Snapshots.
    """
    return load_returns()


@pytest.fixture(scope="module")
def ergebnisse(renditen: pd.DataFrame) -> tuple:
    """Rechnet alle Stresstests einmal je Modul.

    Args:
        renditen: Monatsrenditen.

    Returns:
        Die Ergebnisse.
    """
    return run_all(renditen, POLICIES, ASSET_KEYS)


# --- Die Rechnung selbst ---------------------------------------------------


def test_drawdown_wird_auf_dem_wertindex_gerechnet() -> None:
    """Der Drawdown misst den Rückgang vom Höchststand, nicht den größten Monat.

    Zwei Monate mit je −10 % ergeben 1 − 0,9 · 0,9 = 19 %, nicht 10 %. Der
    Unterschied ist der ganze Zweck der Kennzahl.
    """
    assert max_drawdown(np.array([0.0, -0.10, -0.10, 0.05])) == pytest.approx(-0.19, abs=1e-9)


def test_drawdown_ist_null_bei_durchgehendem_anstieg() -> None:
    """Eine nur steigende Reihe hat keinen Rückgang."""
    assert max_drawdown(np.array([0.02, 0.01, 0.03])) == pytest.approx(0.0)


def test_drawdown_der_leeren_reihe_ist_null() -> None:
    """Eine leere Reihe hat keinen Rückgang."""
    assert max_drawdown(np.array([])) == 0.0


def test_erholungsdauer_wird_ueber_das_fenster_hinaus_gesucht() -> None:
    """Die Erholung wird nach dem Fenster gesucht, nicht darin.

    Ein Krisenfenster endet definitionsgemäß im Tief. Wer die Erholung nur
    innerhalb des Fensters suchte, fände sie nie.

    Die gestellte Reihe fällt im Fenster um 20 % und steigt danach in zwei
    Schritten wieder darüber.
    """
    index = pd.period_range("2020-01", periods=6, freq="M").to_timestamp(how="end")
    daten = np.zeros((6, len(ASSET_KEYS)))
    daten[1, :] = -0.20
    daten[2:, :] = 0.12
    rahmen = pd.DataFrame(daten, index=index, columns=list(ASSET_KEYS))
    fenster = CrisisWindow("test", "Test", "2020-01", "2020-02", "nur zwei Monate")
    gewichte = np.full(len(ASSET_KEYS), 1.0 / len(ASSET_KEYS))

    ergebnis = run_stress_test(rahmen, gewichte, fenster, 3, POLICIES)

    assert ergebnis.max_drawdown == pytest.approx(-0.20)
    assert ergebnis.recovery_months == 2


def test_ausbleibende_erholung_wird_als_solche_gemeldet() -> None:
    """Wird der Höchststand nicht wiedererreicht, ist die Dauer nicht gesetzt."""
    index = pd.period_range("2020-01", periods=6, freq="M").to_timestamp(how="end")
    daten = np.full((6, len(ASSET_KEYS)), -0.05)
    rahmen = pd.DataFrame(daten, index=index, columns=list(ASSET_KEYS))
    fenster = CrisisWindow("test", "Test", "2020-01", "2020-06", "nur Rückgang")
    gewichte = np.full(len(ASSET_KEYS), 1.0 / len(ASSET_KEYS))

    assert run_stress_test(rahmen, gewichte, fenster, 3, POLICIES).recovery_months is None


def test_erholungssuche_ist_nach_oben_begrenzt() -> None:
    """Die Suche endet nach der festgelegten Frist statt am Ende der Reihe.

    Ohne diese Grenze hinge der Wert davon ab, wie lang der Snapshot zufällig
    ist — eine Kennzahl, die mit jedem Datenupdate anders ausfiele.
    """
    assert MAX_RECOVERY_MONTHS > 0
    laenge = MAX_RECOVERY_MONTHS + 24
    index = pd.period_range("2000-01", periods=laenge, freq="M").to_timestamp(how="end")
    daten = np.zeros((laenge, len(ASSET_KEYS)))
    daten[1, :] = -0.30
    daten[MAX_RECOVERY_MONTHS + 5, :] = 1.0  # Erholung erst nach der Frist
    rahmen = pd.DataFrame(daten, index=index, columns=list(ASSET_KEYS))
    fenster = CrisisWindow("test", "Test", "2000-01", "2000-02", "spät")
    gewichte = np.full(len(ASSET_KEYS), 1.0 / len(ASSET_KEYS))

    assert run_stress_test(rahmen, gewichte, fenster, 3, POLICIES).recovery_months is None


def test_fenster_ohne_daten_bricht_ab(renditen: pd.DataFrame) -> None:
    """Ein Fenster außerhalb der Historie ist ein Fehler, kein leeres Ergebnis.

    Args:
        renditen: Monatsrenditen.
    """
    fenster = CrisisWindow("leer", "Vor der Reihe", "1990-01", "1990-12", "nicht vorhanden")
    gewichte = reference_weights(3, POLICIES.classes, ASSET_KEYS)

    with pytest.raises(StressTestError, match="kein einziger Monat"):
        run_stress_test(renditen[list(ASSET_KEYS)].dropna(), gewichte, fenster, 3, POLICIES)


# --- Verkürzte Fenster -----------------------------------------------------


def test_verkuerztes_fenster_wird_ausgewiesen(ergebnisse: tuple) -> None:
    """Die Finanzkrise beginnt vor dem Snapshot — das wird nicht verschwiegen.

    Der ausgewiesene Rückgang von 2008 ist dadurch eine Untergrenze. Ein
    Stresstest, der die angeforderte Fensterlänge behauptete, wäre irreführend.

    Args:
        ergebnisse: Ergebnisse aller Stresstests.
    """
    finanzkrise = [e for e in ergebnisse if e.window.key == "finanzkrise"]

    assert finanzkrise
    for eintrag in finanzkrise:
        assert eintrag.is_truncated
        assert eintrag.months < eintrag.requested_months


def test_vollstaendige_fenster_werden_nicht_als_verkuerzt_gemeldet(ergebnisse: tuple) -> None:
    """Corona und der Zinsschock liegen vollständig im Snapshot.

    Args:
        ergebnisse: Ergebnisse aller Stresstests.
    """
    for schluessel in ("corona", "zinsschock"):
        for eintrag in [e for e in ergebnisse if e.window.key == schluessel]:
            assert not eintrag.is_truncated


# --- Die Befunde -----------------------------------------------------------


def test_verlusterwartung_steigt_monoton_ueber_die_klassen() -> None:
    """Eine höhere Klasse sagt nie weniger Verlust zu als eine niedrigere.

    Die Prüfung steckt im Regelwerksmodell; der Test belegt, dass sie greift.
    """
    zusagen = [
        POLICIES.classes.risikoklassen[klasse].erwarteter_maximalverlust for klasse in RISK_CLASSES
    ]

    assert zusagen == sorted(zusagen)


def test_verlusterwartung_folgt_der_verlustszenario_leiter() -> None:
    """Die Zusage je Klasse entspricht der Stufe S_N der Bereitschaftserhebung.

    Ohne diese Bindung wären Befragung und Zusage zwei unverbundene Zahlenreihen:
    Der Kunde nähme im Fragebogen 20 % hin und bekäme eine Klasse zugewiesen, die
    ihm einen ganz anderen Wert verspricht.
    """
    stufen = {
        stufe.id: stufe.verlust_anteil for stufe in POLICIES.tolerance.verlustszenarien.stufen
    }

    for klasse in RISK_CLASSES:
        zusage = POLICIES.classes.risikoklassen[klasse].erwarteter_maximalverlust
        assert zusage == stufen[f"S{klasse}"], f"Klasse {klasse}"


def test_befund_2022_gebrochene_zusage_bleibt_festgehalten(ergebnisse: tuple) -> None:
    """Der Kernbefund: 2022 brach die Zusage der konservativen Klassen.

    Der Test hält den Befund fest, damit eine Änderung auffällt und begründet
    werden muss. Er behauptet nicht, dass der Befund so bleiben *soll* —
    ändern sich Daten oder Regelwerk, ist das Ergebnis neu einzuordnen.

    Args:
        ergebnisse: Ergebnisse aller Stresstests.
    """
    gebrochen = broken_promises(ergebnisse)
    betroffen = {(e.window.key, e.risk_class) for e in gebrochen}

    assert betroffen == {("zinsschock", 1), ("zinsschock", 2)}, (
        "Die Menge der gebrochenen Zusagen hat sich geändert. Das ist ein fachlicher Befund "
        "und keine Formatierungsfrage."
    )


def test_befund_2022_differenziert_die_klassifizierung_kaum(ergebnisse: tuple) -> None:
    """2022 lagen alle fünf Klassen im Rückgang dicht beieinander.

    Der zweite Kernbefund: Die Risikoklassifizierung leistete in dieser Krise
    fast keine Differenzierung. Aktien und Anleihen fielen gemeinsam; der Puffer,
    auf dem die konservativen Klassen beruhen, wirkte nicht.

    Args:
        ergebnisse: Ergebnisse aller Stresstests.
    """
    rueckgaenge = [abs(e.max_drawdown) for e in ergebnisse if e.window.key == "zinsschock"]
    spanne = (max(rueckgaenge) - min(rueckgaenge)) * 100.0

    assert spanne < 1.5, f"Spanne 2022 beträgt {spanne:.2f} pp"


def test_2008_differenziert_die_klassifizierung_deutlich(ergebnisse: tuple) -> None:
    """Gegenprobe: In der Finanzkrise wirkte die Klassifizierung wie vorgesehen.

    Ohne diese Gegenprobe bliebe offen, ob die fehlende Differenzierung von 2022
    ein Merkmal der Krise oder ein Fehler des Systems ist.

    Args:
        ergebnisse: Ergebnisse aller Stresstests.
    """
    nach_klasse = {
        e.risk_class: abs(e.max_drawdown) for e in ergebnisse if e.window.key == "finanzkrise"
    }
    geordnet = [nach_klasse[k] for k in RISK_CLASSES]

    assert geordnet == sorted(geordnet), "Der Rückgang steigt nicht monoton mit der Klasse."
    assert (geordnet[-1] - geordnet[0]) * 100.0 > 20.0


def test_konservative_klasse_erholt_sich_nach_2022_am_langsamsten(ergebnisse: tuple) -> None:
    """Gegen die Intuition: Klasse 1 brauchte länger als Klasse 5.

    Ursache ist der Anleihenschwerpunkt: Er wurde vom Zinsanstieg getroffen und
    nahm am anschließenden Aktienaufschwung nicht teil. Der Befund gehört in die
    Grenzenliste (G-08), weil er die Anforderungen an die Betreuung im
    Krisenfall betrifft.

    Args:
        ergebnisse: Ergebnisse aller Stresstests.
    """
    nach_klasse = {
        e.risk_class: e.recovery_months for e in ergebnisse if e.window.key == "zinsschock"
    }

    assert nach_klasse[1] is not None
    assert nach_klasse[5] is not None
    assert nach_klasse[1] > nach_klasse[5]


def test_alle_fenster_und_klassen_werden_gerechnet(ergebnisse: tuple) -> None:
    """Jede Kombination aus Fenster und Klasse liegt vor.

    Args:
        ergebnisse: Ergebnisse aller Stresstests.
    """
    assert len(ergebnisse) == len(CRISIS_WINDOWS) * len(RISK_CLASSES)


def test_stresstest_ist_deterministisch(renditen: pd.DataFrame) -> None:
    """Der Stresstest zieht nichts — zwei Läufe sind gleich (Prinzip P1).

    Args:
        renditen: Monatsrenditen.
    """
    erst = run_all(renditen, POLICIES, ASSET_KEYS)
    zweit = run_all(renditen, POLICIES, ASSET_KEYS)

    assert [e.max_drawdown for e in erst] == [e.max_drawdown for e in zweit]


# --- Sensitivitätsanalyse --------------------------------------------------


@pytest.fixture(scope="module")
def sensitivitaet() -> list[SensitivityReport]:
    """Rechnet die Sensitivitätsanalyse einmal je Modul.

    Returns:
        Die Berichte je Basisprofil.
    """
    anlageziel = build_goal(
        "Z-SENS-A",
        GoalType.ALTERSVORSORGE,
        Decimal("300000.00"),
        SENSITIVITY_DATE,
        POLICIES.goals,
        initial_capital=Decimal("20000.00"),
        monthly_contribution=Decimal("500.00"),
    )
    return [
        analyse(kunde, anlageziel, POLICIES, name) for name, kunde in baseline_clients().items()
    ]


def test_drei_basisprofile_werden_geprueft(sensitivitaet: list[SensitivityReport]) -> None:
    """Die Analyse läuft über mehrere Basisprofile.

    Ein einzelnes Profil genügt nicht: Der Hebel eines Items hängt davon ab, ob
    seine Achse gerade bindet.

    Args:
        sensitivitaet: Die Berichte.
    """
    assert len(sensitivitaet) == 3
    assert {bericht.profile_name for bericht in sensitivitaet} == {
        "ausgewogen",
        "vorsichtig",
        "offensiv",
    }


def test_jedes_item_wird_ueber_alle_auspraegungen_variiert(
    sensitivitaet: list[SensitivityReport],
) -> None:
    """Jedes Item wird mit mindestens zwei Ausprägungen geprüft.

    Args:
        sensitivitaet: Die Berichte.
    """
    for bericht in sensitivitaet:
        assert bericht.items
        for eintrag in bericht.items:
            assert eintrag.variants >= 2
            assert eintrag.min_class <= eintrag.baseline_class <= eintrag.max_class


def test_alle_drei_achsen_sind_vertreten(sensitivitaet: list[SensitivityReport]) -> None:
    """Die Analyse deckt Tragfähigkeit, Bereitschaft und Kenntnisse ab.

    Args:
        sensitivitaet: Die Berichte.
    """
    achsen = {eintrag.axis for bericht in sensitivitaet for eintrag in bericht.items}

    assert achsen == {AXIS_CAPACITY, AXIS_TOLERANCE, AXIS_KNOWLEDGE}


def test_befund_die_deckel_kriterien_tragen_den_groessten_hebel(
    sensitivitaet: list[SensitivityReport],
) -> None:
    """Kernbefund: Kenntnisdeckel und Liquiditäts-K.-o. bewegen am meisten.

    Beide sind **keine** Bestandteile eines gewichteten Punktescores. Das am
    stärksten gewichtete Item der Bereitschaftserhebung (35 %) bewegt weniger.
    Der Test hält den Befund fest, weil er für die Erhebungspraxis folgenreich
    ist (Grenze G-05).

    Args:
        sensitivitaet: Die Berichte.
    """
    groesster: dict[str, int] = {}
    for bericht in sensitivitaet:
        for eintrag in bericht.items:
            groesster[eintrag.item] = max(groesster.get(eintrag.item, 0), eintrag.span)

    spitze = max(groesster.values())
    fuehrend = {item for item, spanne in groesster.items() if spanne == spitze}

    assert fuehrend == {"knowledge_level", "liquidity_reserve_months"}
    assert groesster["verlustszenario_betrag"] < spitze


def test_hebel_ist_ohne_wirkung_wenn_die_achse_nicht_bindet(
    sensitivitaet: list[SensitivityReport],
) -> None:
    """Im vorsichtigen Profil bewegt fast nichts die Klasse.

    Das Profil liegt bereits bei Klasse 1 und wird zusätzlich vom Kenntnisdeckel
    begrenzt. Der Befund macht anschaulich, was das bindende Minimum bedeutet:
    Solange eine andere Achse bindet, bleibt jede Verbesserung folgenlos.

    Args:
        sensitivitaet: Die Berichte.
    """
    vorsichtig = next(b for b in sensitivitaet if b.profile_name == "vorsichtig")
    ohne_wirkung = [eintrag for eintrag in vorsichtig.items if eintrag.span == 0]

    assert len(ohne_wirkung) >= len(vorsichtig.items) - 2


def test_bericht_ordnet_nach_hebel(sensitivitaet: list[SensitivityReport]) -> None:
    """Die Items stehen nach Spannweite absteigend.

    Args:
        sensitivitaet: Die Berichte.
    """
    for bericht in sensitivitaet:
        spannen = [eintrag.span for eintrag in bericht.items]
        assert spannen == sorted(spannen, reverse=True)
        assert bericht.strongest.span == spannen[0]


def test_leerer_bericht_hat_kein_staerkstes_item() -> None:
    """Ohne Items gibt es kein stärkstes — das ist ein Fehler, kein None."""
    leer = SensitivityReport("leer", 3, "Risikotragfähigkeit", ())

    with pytest.raises(SensitivityError, match="kein einziges Item"):
        _ = leer.strongest


def test_sensitivitaet_ist_deterministisch(sensitivitaet: list[SensitivityReport]) -> None:
    """Zwei Läufe liefern dieselben Hebel (Prinzip P1).

    Args:
        sensitivitaet: Die Berichte.
    """
    anlageziel = build_goal(
        "Z-SENS-A",
        GoalType.ALTERSVORSORGE,
        Decimal("300000.00"),
        SENSITIVITY_DATE,
        POLICIES.goals,
        initial_capital=Decimal("20000.00"),
        monthly_contribution=Decimal("500.00"),
    )
    erneut = [
        analyse(kunde, anlageziel, POLICIES, name) for name, kunde in baseline_clients().items()
    ]

    for vorher, nachher in zip(sensitivitaet, erneut, strict=True):
        assert [e.span for e in vorher.items] == [e.span for e in nachher.items]


# --- Auswertungsabschnitte -------------------------------------------------


def test_stressabschnitt_nennt_die_verkuerzung(ergebnisse: tuple) -> None:
    """Der erzeugte Abschnitt weist die unvollständige Historie aus.

    Args:
        ergebnisse: Ergebnisse aller Stresstests.
    """
    block = render_stress(ergebnisse, POLICIES)

    assert block.startswith(STRESS_START)
    assert block.rstrip().endswith(STRESS_END)
    assert "Untergrenze" in block
    assert "13 von 18" in block


def test_stressabschnitt_markiert_gebrochene_zusagen(ergebnisse: tuple) -> None:
    """Gebrochene Zusagen sind im Abschnitt gekennzeichnet und aufgelistet.

    Args:
        ergebnisse: Ergebnisse aller Stresstests.
    """
    block = render_stress(ergebnisse, POLICIES)

    assert "⚠" in block
    assert "Hielt die Zusage?" in block
    assert "Zins- und Inflationsschock 2022" in block


def test_sensitivitaetsabschnitt_nennt_den_befund(
    sensitivitaet: list[SensitivityReport],
) -> None:
    """Der erzeugte Abschnitt benennt die beiden Deckel-Kriterien.

    Args:
        sensitivitaet: Die Berichte.
    """
    block = render_sensitivity(sensitivitaet)

    assert block.startswith(SENSITIVITY_START)
    assert block.rstrip().endswith(SENSITIVITY_END)
    assert "Kenntnisstand (Deckel)" in block
    assert "Liquiditätsreserve (K.-o.)" in block
    assert "Art. 55 DelVO" in block


def test_beide_abschnitte_verwenden_deutsche_zahlen(ergebnisse: tuple) -> None:
    """Kein englischer Dezimalpunkt in den erzeugten Abschnitten.

    Args:
        ergebnisse: Ergebnisse aller Stresstests.
    """
    block = render_stress(ergebnisse, POLICIES)

    # Ein Punkt zwischen zwei Ziffern ist im deutschen Fliesstext stets ein
    # Fehler. Ausgenommen sind Gliederungsnummern in Ueberschriften ("### 3.1")
    # und Versionsangaben ("1.1.0") — beides sind keine gerechneten Zahlen.
    for zeile in block.splitlines():
        if zeile.startswith("#"):
            continue
        ohne_versionen = re.sub(r"\d+\.\d+\.\d+", "", zeile)
        treffer = re.search(r"\d\.\d", ohne_versionen)
        assert treffer is None, f"Englischer Dezimalpunkt in: {zeile!r}"


def test_abschnitte_werden_an_den_marken_ersetzt(tmp_path: Path) -> None:
    """Ein zweiter Lauf ersetzt die Abschnitte, statt sie anzuhängen.

    Args:
        tmp_path: Temporäres Verzeichnis.
    """
    ziel = tmp_path / "auswertung.md"
    ziel.write_text(
        f"Kopf\n\n{STRESS_START}\nalt A\n{STRESS_END}\n\n"
        f"{SENSITIVITY_START}\nalt B\n{SENSITIVITY_END}\n\nFuss\n",
        encoding="utf-8",
    )

    write_evaluation(
        f"{STRESS_START}\nneu A\n{STRESS_END}",
        f"{SENSITIVITY_START}\nneu B\n{SENSITIVITY_END}",
        ziel,
    )
    inhalt = ziel.read_text(encoding="utf-8")

    assert "alt A" not in inhalt
    assert "alt B" not in inhalt
    assert "neu A" in inhalt
    assert "neu B" in inhalt
    assert inhalt.count(STRESS_START) == 1
    assert inhalt.rstrip().endswith("Fuss")


def test_fehlende_zieldatei_bricht_ab(tmp_path: Path) -> None:
    """Ohne Zieldatei wird nichts geschrieben.

    Args:
        tmp_path: Temporäres Verzeichnis.
    """
    with pytest.raises(FileNotFoundError, match="Zieldatei fehlt"):
        write_evaluation("a", "b", tmp_path / "gibt-es-nicht.md")


@pytest.mark.slow
def test_beide_abschnitte_entstehen_gemeinsam() -> None:
    """``build_blocks`` liefert beide Abschnitte fehlerfrei."""
    stress, sensitivitaet = build_blocks()

    assert STRESS_START in stress
    assert SENSITIVITY_START in sensitivitaet
