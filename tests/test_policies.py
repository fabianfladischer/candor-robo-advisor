"""Tests des Regelwerks und seines Loaders.

Zwei Nachweise stehen im Vordergrund:

1. Das ausgelieferte Regelwerk ist in sich stimmig und lädt.
2. **Fehlerhaftes YAML wird abgewiesen** — und zwar auch dann, wenn der Fehler
   nur ein Tippfehler in einem Schlüsselnamen ist. Genau dieser Fall ist der
   gefährliche: Er bliebe unbemerkt und veränderte trotzdem jedes Ergebnis.

Die Prüfungen arbeiten auf einer Kopie des Regelwerks in einem temporären
Verzeichnis, damit die ausgelieferten Dateien unangetastet bleiben.
"""

from __future__ import annotations

import re
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from candor.data.policy_loader import POLICY_FILES, default_policy_dir, load_policy_set
from candor.data.universe_meta import ASSET_KEYS as UNIVERSE_KEYS
from candor.domain.errors import PolicyError
from candor.domain.goal import GoalPriority
from candor.domain.policy import ASSET_KEYS as POLICY_KEYS
from candor.domain.policy import MAX_RISK_CLASS, MIN_RISK_CLASS, Band, CapacityDimension, ScoreBand


@pytest.fixture
def policy_kopie(out_dir: Path) -> Path:
    """Legt eine beschreibbare Kopie des Regelwerks an.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.

    Returns:
        Pfad des kopierten Regelwerksverzeichnisses.
    """
    ziel = out_dir / "policies"
    shutil.copytree(default_policy_dir(), ziel)
    return ziel


def aendere(pfad: Path, mutation: Any) -> None:
    """Lädt eine YAML-Datei, wendet eine Änderung an und schreibt sie zurück.

    Args:
        pfad: Pfad der Datei.
        mutation: Funktion, die den geladenen Inhalt verändert.
    """
    inhalt = yaml.safe_load(pfad.read_text(encoding="utf-8"))
    mutation(inhalt)
    pfad.write_text(yaml.safe_dump(inhalt, allow_unicode=True, sort_keys=False), encoding="utf-8")


# --- Das ausgelieferte Regelwerk ------------------------------------------


def test_ausgeliefertes_regelwerk_laedt() -> None:
    """Alle sechs Policies lassen sich laden und sind in sich stimmig."""
    regelwerk = load_policy_set()

    assert set(regelwerk.versions) == set(POLICY_FILES)
    # Bewusst gegen das Muster statt gegen einen festen Wert: Eine Policy soll
    # sich fortschreiben lassen, ohne dass ein Test nachgezogen werden muss.
    assert all(re.match(r"^\d+\.\d+\.\d+$", v) for v in regelwerk.versions.values())


def test_kopf_ist_je_datei_vollstaendig() -> None:
    """Jede Policy trägt Version, Gültigkeitsdatum, Beschreibung und Quelle."""
    regelwerk = load_policy_set()

    for name in POLICY_FILES:
        policy = getattr(regelwerk, name)
        assert policy.version
        assert policy.gueltig_ab is not None
        assert len(policy.beschreibung) >= 10
        assert len(policy.quelle) >= 5


def test_versionskennung_ist_verdichtet_darstellbar() -> None:
    """Die verdichtete Versionskennung nennt jede Teilpolicy."""
    regelwerk = load_policy_set()
    kennung = regelwerk.combined_version

    for name, version in regelwerk.versions.items():
        assert f"{name}={version}" in kennung


def test_assetklassen_stimmen_mit_dem_universum_ueberein() -> None:
    """Regelwerk und Datenschicht verwenden dieselben Assetklassen.

    Der Domänenkern darf die Datenschicht nicht importieren (Prinzip P4), führt
    die Kennungen deshalb selbst. Dieser Test hält beide Listen zusammen.
    """
    assert POLICY_KEYS == UNIVERSE_KEYS


def test_alle_risikoklassen_sind_definiert() -> None:
    """Die Klassen 1 bis 5 sind vollständig hinterlegt."""
    klassen = load_policy_set().classes.risikoklassen

    assert set(klassen) == set(range(MIN_RISK_CLASS, MAX_RISK_CLASS + 1))


def test_aktienquote_steigt_ueber_die_klassen() -> None:
    """Invariante I6 gilt bereits im Regelwerk, nicht erst im Ergebnis."""
    klassen = load_policy_set().classes.risikoklassen
    quoten = [klassen[k].gruppen["aktien"].max for k in sorted(klassen)]

    assert quoten == sorted(quoten)
    assert quoten[0] < quoten[-1]


def test_baender_lassen_ein_zulaessiges_portfolio_zu() -> None:
    """Invariante I5: Untergrenzen ≤ 1 ≤ Obergrenzen je Risikoklasse."""
    klassen = load_policy_set().classes.risikoklassen

    for nummer, definition in klassen.items():
        summe_min = sum(band.min for band in definition.einzelwerte.values())
        summe_max = sum(band.max for band in definition.einzelwerte.values())
        assert summe_min <= Decimal(1), f"Klasse {nummer}: Untergrenzen zu hoch"
        assert summe_max >= Decimal(1), f"Klasse {nummer}: Obergrenzen zu niedrig"


def test_muss_ziel_verlangt_mehr_als_wunsch_ziel() -> None:
    """Der Schwellenwert für MUSS-Ziele liegt über dem für WUNSCH-Ziele."""
    goals = load_policy_set().goals

    assert goals.minimum_success_probability(GoalPriority.MUSS) > goals.minimum_success_probability(
        GoalPriority.WUNSCH
    )


def test_eigene_vorgabe_kann_die_schwelle_nur_anheben() -> None:
    """Eine individuelle Vorgabe darf den Schwellenwert nicht unterschreiten.

    Sonst ließe sich die Konfliktdiagnose durch eine niedrigere Eigenvorgabe
    umgehen — genau die Beschönigung, die das System vermeiden soll.
    """
    goals = load_policy_set().goals
    schwelle = goals.minimum_success_probability(GoalPriority.MUSS)

    assert goals.minimum_success_probability(GoalPriority.MUSS, Decimal("0.10")) == schwelle
    assert goals.minimum_success_probability(GoalPriority.MUSS, Decimal("0.99")) == Decimal("0.99")


# --- Fehlerhafte Regelwerke -----------------------------------------------


def test_unbekannter_schluessel_wird_abgewiesen(policy_kopie: Path) -> None:
    """Ein Tippfehler im Schlüsselnamen führt zum Abbruch.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "goals_v1.yaml",
        lambda d: d.update({"mindest_erfolgswahrscheinlikeit": {"MUSS": 0.9}}),
    )

    with pytest.raises(PolicyError, match="unbekannter Schlüssel"):
        load_policy_set(policy_kopie)


def test_fehlende_datei_wird_gemeldet(policy_kopie: Path) -> None:
    """Eine fehlende Policy-Datei bricht den Ladevorgang ab.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    (policy_kopie / "classes_v1.yaml").unlink()

    with pytest.raises(PolicyError, match="fehlt"):
        load_policy_set(policy_kopie)


def test_ungueltiges_yaml_wird_gemeldet(policy_kopie: Path) -> None:
    """Syntaktisch fehlerhaftes YAML bricht den Ladevorgang ab.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    (policy_kopie / "goals_v1.yaml").write_text(
        "version: '1.0.0'\n  : [unbalanced", encoding="utf-8"
    )

    with pytest.raises(PolicyError, match="kein gültiges YAML"):
        load_policy_set(policy_kopie)


def test_leere_datei_wird_gemeldet(policy_kopie: Path) -> None:
    """Eine leere Policy-Datei bricht den Ladevorgang ab.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    (policy_kopie / "glidepath_v1.yaml").write_text("", encoding="utf-8")

    with pytest.raises(PolicyError, match="leer"):
        load_policy_set(policy_kopie)


def test_fehlender_pflichtschluessel_wird_gemeldet(policy_kopie: Path) -> None:
    """Fehlt die Versionsangabe, bricht der Ladevorgang ab.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(policy_kopie / "capacity_v1.yaml", lambda d: d.pop("version"))

    with pytest.raises(PolicyError, match="version"):
        load_policy_set(policy_kopie)


def test_gewichtssumme_ungleich_eins_wird_abgewiesen(policy_kopie: Path) -> None:
    """Dimensionsgewichte müssen sich zu eins addieren.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "capacity_v1.yaml",
        lambda d: d["dimensionen"]["liquiditaetsreserve"].update({"gewicht": 0.5}),
    )

    with pytest.raises(PolicyError, match="Dimensionsgewichte"):
        load_policy_set(policy_kopie)


def test_fehlende_beschaeftigungsart_wird_abgewiesen(policy_kopie: Path) -> None:
    """Jede Beschäftigungsart muss bewertet sein.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "capacity_v1.yaml",
        lambda d: d["dimensionen"]["einkommensstabilitaet"]["optionen"].pop("VERBEAMTET"),
    )

    with pytest.raises(PolicyError, match="einkommensstabilitaet"):
        load_policy_set(policy_kopie)


def test_nicht_offenes_letztes_band_wird_abgewiesen(policy_kopie: Path) -> None:
    """Das letzte Band einer stetigen Dimension muss nach oben offen sein.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "capacity_v1.yaml",
        lambda d: d["dimensionen"]["liquiditaetsreserve"]["baender"][-1].update({"bis": 99.0}),
    )

    with pytest.raises(PolicyError, match="nach oben offen"):
        load_policy_set(policy_kopie)


def test_unsortierte_klassenschwellen_werden_abgewiesen(policy_kopie: Path) -> None:
    """Klassenschwellen müssen streng aufsteigend sortiert sein.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "tolerance_v1.yaml",
        lambda d: d["klassenschwellen"].reverse(),
    )

    with pytest.raises(PolicyError, match="aufsteigend"):
        load_policy_set(policy_kopie)


def test_fehlende_assetklasse_wird_abgewiesen(policy_kopie: Path) -> None:
    """Jede Risikoklasse muss alle acht Assetklassen führen.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "classes_v1.yaml",
        lambda d: d["risikoklassen"][3]["einzelwerte"].pop("GOLD"),
    )

    with pytest.raises(PolicyError, match="einzelwerte"):
        load_policy_set(policy_kopie)


def test_untergrenzen_ueber_eins_werden_abgewiesen(policy_kopie: Path) -> None:
    """Untergrenzen, die zusammen über eins liegen, lassen kein Portfolio zu.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "classes_v1.yaml",
        lambda d: d["risikoklassen"][3]["einzelwerte"]["BD_GOV"].update({"min": 0.95, "max": 0.95}),
    )

    with pytest.raises(PolicyError):
        load_policy_set(policy_kopie)


def test_nicht_monotone_aktienquote_wird_abgewiesen(policy_kopie: Path) -> None:
    """Eine über die Klassen fallende Aktienquote verletzt Invariante I6.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "classes_v1.yaml",
        lambda d: d["risikoklassen"][5]["gruppen"]["aktien"].update({"min": 0.00, "max": 0.05}),
    )

    with pytest.raises(PolicyError, match="monoton"):
        load_policy_set(policy_kopie)


def test_glidepath_ohne_untere_stufe_wird_abgewiesen(policy_kopie: Path) -> None:
    """Der Glide-Path muss bis Restlaufzeit null reichen.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(policy_kopie / "glidepath_v1.yaml", lambda d: d["stufen"].pop())

    with pytest.raises(PolicyError, match="Restlaufzeit 0"):
        load_policy_set(policy_kopie)


def test_steigende_risikoklasse_bei_kuerzerer_restlaufzeit_wird_abgewiesen(
    policy_kopie: Path,
) -> None:
    """Mit kürzerer Restlaufzeit darf die zulässige Klasse nicht steigen.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "glidepath_v1.yaml",
        lambda d: d["stufen"][-1].update({"max_risikoklasse": 5}),
    )

    with pytest.raises(PolicyError, match="nicht steigen"):
        load_policy_set(policy_kopie)


def test_muss_schwelle_unter_wunsch_schwelle_wird_abgewiesen(policy_kopie: Path) -> None:
    """Ein MUSS-Ziel darf nicht weniger verlangen als ein WUNSCH-Ziel.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "goals_v1.yaml",
        lambda d: d["mindest_erfolgswahrscheinlichkeit"].update({"MUSS": 0.50}),
    )

    with pytest.raises(PolicyError, match="höhere Mindestwahrscheinlichkeit"):
        load_policy_set(policy_kopie)


def test_unbekannte_instrumentengattung_wird_abgewiesen(policy_kopie: Path) -> None:
    """Eine erfundene Instrumentengattung führt zum Abbruch.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "knowledge_v1.yaml",
        lambda d: d["gattungen"].update(
            {
                "KRYPTO": {
                    "schaltet_risikoklasse_frei": 5,
                    "mindestkenntnis": "umfassende Kenntnisse",
                    "mindesterfahrung_jahre": 1,
                    "mindesttransaktionen_pro_jahr": 1,
                }
            }
        ),
    )

    with pytest.raises(PolicyError, match="unbekannt"):
        load_policy_set(policy_kopie)


# --- Stimmigkeitsprüfungen der Modelle direkt -----------------------------
#
# Die folgenden Fälle prüfen die Fehlerzweige der Regelwerksmodelle unmittelbar,
# ohne den Umweg über YAML. Über Dateien wären sie nur umständlich erreichbar,
# und jeder Zweig soll einzeln belegt sein: Eine Stimmigkeitsprüfung, die nie
# ausgelöst hat, ist keine Prüfung, sondern eine Behauptung.


def test_dimension_ohne_bewertungsform_wird_abgewiesen() -> None:
    """Eine Dimension braucht entweder Bänder oder Optionen."""
    with pytest.raises(PolicyError, match="genau eines"):
        CapacityDimension(beschreibung="Testdimension", gewicht=Decimal("1.0"))


def test_dimension_mit_beiden_bewertungsformen_wird_abgewiesen() -> None:
    """Bänder und Optionen zugleich sind unzulässig."""
    with pytest.raises(PolicyError, match="genau eines"):
        CapacityDimension(
            beschreibung="Testdimension",
            gewicht=Decimal("1.0"),
            baender=(ScoreBand(bis=None, punkte=10),),
            optionen={"A": 1},
        )


def test_dimension_ohne_baender_wird_abgewiesen() -> None:
    """Eine stetige Dimension ohne jedes Band ist unzulässig."""
    with pytest.raises(PolicyError, match="mindestens ein Band"):
        CapacityDimension(beschreibung="Testdimension", gewicht=Decimal("1.0"), baender=())


def test_offenes_band_in_der_mitte_wird_abgewiesen() -> None:
    """Nur das letzte Band darf nach oben offen sein."""
    with pytest.raises(PolicyError, match="Nur das letzte Band"):
        CapacityDimension(
            beschreibung="Testdimension",
            gewicht=Decimal("1.0"),
            baender=(
                ScoreBand(bis=None, punkte=10),
                ScoreBand(bis=Decimal("5"), punkte=20),
                ScoreBand(bis=None, punkte=30),
            ),
        )


def test_unsortierte_bandgrenzen_werden_abgewiesen() -> None:
    """Bandgrenzen müssen streng aufsteigend sein."""
    with pytest.raises(PolicyError, match="aufsteigend"):
        CapacityDimension(
            beschreibung="Testdimension",
            gewicht=Decimal("1.0"),
            baender=(
                ScoreBand(bis=Decimal("9"), punkte=10),
                ScoreBand(bis=Decimal("3"), punkte=20),
                ScoreBand(bis=None, punkte=30),
            ),
        )


def test_band_mit_untergrenze_ueber_obergrenze_wird_abgewiesen() -> None:
    """Eine Untergrenze über der Obergrenze ist unzulässig."""
    with pytest.raises(PolicyError, match="Bandbreite unzulässig"):
        Band(min=Decimal("0.8"), max=Decimal("0.2"))


def test_klassenschwellen_muessen_bei_null_beginnen(policy_kopie: Path) -> None:
    """Ohne Schwelle bei null bliebe die unterste Punktzahl unbewertet.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "capacity_v1.yaml",
        lambda d: d["klassenschwellen"][0].update({"ab_punkte": 5}),
    )

    with pytest.raises(PolicyError, match="bei 0 Punkten"):
        load_policy_set(policy_kopie)


def test_nicht_erreichbare_risikoklasse_wird_abgewiesen(policy_kopie: Path) -> None:
    """Jede Klasse von 1 bis 5 muss über eine Schwelle erreichbar sein.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(policy_kopie / "capacity_v1.yaml", lambda d: d["klassenschwellen"].pop())

    with pytest.raises(PolicyError, match="Nicht erreichbare Risikoklassen"):
        load_policy_set(policy_kopie)


def test_positionsobergrenze_wird_durchgesetzt(policy_kopie: Path) -> None:
    """Keine Einzelobergrenze darf über der Positionsobergrenze liegen.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "classes_v1.yaml",
        lambda d: d["risikoklassen"][3]["einzelwerte"]["EQ_DEV"].update({"max": 0.99}),
    )

    with pytest.raises(PolicyError, match="Positionsobergrenze"):
        load_policy_set(policy_kopie)


def test_obergrenzen_unter_eins_werden_abgewiesen(policy_kopie: Path) -> None:
    """Obergrenzen, die zusammen unter eins liegen, lassen kein Portfolio zu.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """

    def kappe(inhalt: Any) -> None:
        for band in inhalt["risikoklassen"][3]["einzelwerte"].values():
            band["min"] = 0.0
            band["max"] = 0.05

    aendere(policy_kopie / "classes_v1.yaml", kappe)

    with pytest.raises(PolicyError, match="Obergrenzen summieren"):
        load_policy_set(policy_kopie)


def test_doppelte_gruppenzuordnung_wird_abgewiesen(policy_kopie: Path) -> None:
    """Eine Assetklasse darf nicht zwei Gruppen zugeordnet sein.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "classes_v1.yaml",
        lambda d: d["gruppenzuordnung"]["sachwerte"].append("EQ_DEV"),
    )

    with pytest.raises(PolicyError):
        load_policy_set(policy_kopie)


def test_glidepath_ohne_stufen_wird_abgewiesen(policy_kopie: Path) -> None:
    """Ein Glide-Path ohne jede Stufe ist unzulässig.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(policy_kopie / "glidepath_v1.yaml", lambda d: d.update({"stufen": []}))

    with pytest.raises(PolicyError, match="mindestens eine Stufe"):
        load_policy_set(policy_kopie)


def test_unsortierte_glidepath_stufen_werden_abgewiesen(policy_kopie: Path) -> None:
    """Die Stufen müssen streng absteigend nach Restlaufzeit sortiert sein.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(policy_kopie / "glidepath_v1.yaml", lambda d: d["stufen"].reverse())

    with pytest.raises(PolicyError, match="absteigend"):
        load_policy_set(policy_kopie)


def test_fehlende_zielprioritaet_wird_abgewiesen(policy_kopie: Path) -> None:
    """Jede Zielpriorität braucht einen Schwellenwert.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    aendere(
        policy_kopie / "goals_v1.yaml",
        lambda d: d["mindest_erfolgswahrscheinlichkeit"].pop("WUNSCH"),
    )

    with pytest.raises(PolicyError, match="fehlend"):
        load_policy_set(policy_kopie)


def test_regelwerk_aus_nicht_zuordnung_wird_abgewiesen(policy_kopie: Path) -> None:
    """Eine YAML-Liste statt einer Zuordnung bricht den Ladevorgang ab.

    Args:
        policy_kopie: Beschreibbare Kopie des Regelwerks.
    """
    (policy_kopie / "goals_v1.yaml").write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(PolicyError, match="Zuordnung erwartet"):
        load_policy_set(policy_kopie)
