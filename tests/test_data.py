"""Tests der Datenschicht.

Der Schwerpunkt liegt auf den Eigenschaften, die für die Reproduzierbarkeit
tragend sind:

* Das Universum ist widerspruchsfrei und jede ISIN ist als geprüft ausgewiesen.
* Der Loader liest den Snapshot und geht **nicht** von selbst ins Netz.
* Das Manifest erkennt jede Veränderung einer Snapshot-Datei.
* Die Kennzahlen sind gegen analytisch bekannte Fälle geprüft, nicht gegen
  sich selbst.

Alle Tests laufen ohne Netzzugang. Der einzige Live-Test ist mit ``netz``
markiert und wird nur ausgeführt, wenn ``CANDOR_NETZTEST=1`` gesetzt ist.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from candor.data.fetch import (
    DataSourceError,
    fetch_prices_eur,
    to_monthly_returns,
)
from candor.data.loader import SnapshotMissingError, load_prices, load_returns, snapshot_info
from candor.data.manifest import ManifestError, read_manifest, sha256_of, verify_snapshot
from candor.data.snapshot import (
    PRICES_FILE,
    RETURNS_FILE,
    SnapshotError,
    build_snapshot,
    find_internal_gaps,
)
from candor.data.stats import compute_stats, correlation_matrix, format_de, max_drawdown
from candor.data.universe_meta import (
    ASSET_KEYS,
    BASE_CURRENCY,
    UNIVERSE,
    Instrument,
    IsinStatus,
    get_instrument,
)

#: Aufbau einer ISIN: zwei Buchstaben Länderpräfix, neun Zeichen, eine Prüfziffer.
ISIN_MUSTER = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


# --- Hilfsmittel -----------------------------------------------------------


def _monatsreihe(werte: list[float], start: str = "2020-01") -> pd.Series:
    """Baut eine Kursreihe auf Monatsenden.

    Args:
        werte: Kurswerte in zeitlicher Reihenfolge.
        start: Erster Monat im Format ``YYYY-MM``.

    Returns:
        Reihe mit Monatsenden als Index.
    """
    index = pd.period_range(start=start, periods=len(werte), freq="M").to_timestamp("M")
    return pd.Series(werte, index=index, dtype="float64")


class FakePriceSource:
    """Datenquelle für Tests — erfüllt das Protocol ``PriceSource`` ohne Netz.

    Der Test belegt damit zugleich, dass die Quelle tatsächlich austauschbar ist:
    Der Rest der Datenschicht kennt nur das Protocol.
    """

    def __init__(self, reihen: dict[str, pd.Series]) -> None:
        """Legt die Quelle an.

        Args:
            reihen: Zuordnung Symbol auf Kursreihe.
        """
        self._reihen = reihen
        self.abgefragt: list[str] = []

    @property
    def label(self) -> str:
        """Bezeichnung der Quelle."""
        return "Testquelle (fest verdrahtete Reihen)"

    def monthly_closes(self, symbol: str) -> pd.Series:
        """Liefert die hinterlegte Reihe.

        Args:
            symbol: Abgefragtes Symbol.

        Returns:
            Die hinterlegte Kursreihe.

        Raises:
            DataSourceError: Wenn für das Symbol nichts hinterlegt ist.
        """
        self.abgefragt.append(symbol)
        if symbol not in self._reihen:
            raise DataSourceError(f"Testquelle kennt {symbol!r} nicht.")
        return self._reihen[symbol]


class ExplodingPriceSource:
    """Quelle, die bei jedem Zugriff fehlschlägt.

    Dient dem Nachweis, dass der Loader ohne ``live=True`` nicht ins Netz geht.
    """

    @property
    def label(self) -> str:
        """Bezeichnung der Quelle."""
        return "darf nicht aufgerufen werden"

    def monthly_closes(self, symbol: str) -> pd.Series:
        """Schlägt immer fehl.

        Args:
            symbol: Abgefragtes Symbol.

        Raises:
            AssertionError: Immer.
        """
        msg = f"Es wurde unerwartet ein Live-Abruf für {symbol!r} versucht."
        raise AssertionError(msg)


# --- Universum -------------------------------------------------------------


def test_universum_hat_acht_assetklassen() -> None:
    """Das Universum umfasst die acht festgelegten Assetklassen."""
    assert len(UNIVERSE) == 8
    assert tuple(instrument.key for instrument in UNIVERSE) == ASSET_KEYS


@pytest.mark.parametrize("instrument", UNIVERSE, ids=lambda i: i.key)
def test_isin_ist_formal_gueltig_und_geprueft(instrument: Instrument) -> None:
    """Jede ISIN ist formal gültig und als geprüft ausgewiesen.

    Kennungen werden nicht geraten: Eine ISIN mit dem Status ``UNVERIFIED``
    darf nicht ohne eigene Nachprüfung verwendet werden; dieser Test hält
    fest, dass derzeit keine solche enthalten ist.

    Args:
        instrument: Das geprüfte Instrument.
    """
    assert ISIN_MUSTER.match(instrument.isin), f"{instrument.key}: ISIN formal ungültig"
    assert instrument.isin_status is not IsinStatus.UNVERIFIED


def test_kennungen_symbole_und_isins_sind_eindeutig() -> None:
    """Keine Kennung, kein Symbol und keine ISIN kommt doppelt vor."""
    for feld in ("key", "symbol", "isin"):
        werte = [getattr(instrument, feld) for instrument in UNIVERSE]
        assert len(set(werte)) == len(werte), f"Doppelter Wert im Feld {feld}"


def test_nur_gold_benoetigt_waehrungsumrechnung() -> None:
    """Genau ein Instrument notiert nicht in der Basiswährung."""
    umzurechnen = [i.key for i in UNIVERSE if i.needs_fx_conversion]
    assert umzurechnen == ["GOLD"]
    assert get_instrument("GOLD").currency != BASE_CURRENCY


def test_unbekannte_assetklasse_wirft_fehler() -> None:
    """Eine unbekannte Kennung führt zu ``KeyError`` statt zu ``None`` (Prinzip P5)."""
    with pytest.raises(KeyError, match="Unbekannte Assetklasse"):
        get_instrument("GIBT_ES_NICHT")


# --- Abruf über das Protocol ----------------------------------------------


def test_fremdwaehrung_wird_in_euro_umgerechnet() -> None:
    """Eine USD-Reihe wird über den Wechselkurs korrekt in Euro umgerechnet."""
    universum = (
        Instrument(
            key="USD_ASSET",
            label="Testwert",
            fund_name="Test",
            isin="US0000000001",
            isin_status=IsinStatus.DIRECT,
            symbol="TEST.US",
            exchange="TEST",
            currency="USD",
            wrapper=UNIVERSE[0].wrapper,
            distributing=False,
        ),
    )
    quelle = FakePriceSource(
        {
            "TEST.US": _monatsreihe([100.0, 110.0, 120.0]),
            "EURUSD=X": _monatsreihe([1.0, 1.1, 2.0]),
        }
    )

    kurse = fetch_prices_eur(quelle, universum)

    # 100/1,0 = 100 · 110/1,1 = 100 · 120/2,0 = 60
    assert list(kurse["USD_ASSET"]) == pytest.approx([100.0, 100.0, 60.0])
    assert "EURUSD=X" in quelle.abgefragt


def test_ohne_fremdwaehrung_kein_wechselkursabruf() -> None:
    """Notieren alle Instrumente in Euro, wird keine Wechselkursreihe abgerufen."""
    universum = (
        Instrument(
            key="EUR_ASSET",
            label="Testwert",
            fund_name="Test",
            isin="DE0000000001",
            isin_status=IsinStatus.DIRECT,
            symbol="TEST.DE",
            exchange="TEST",
            currency="EUR",
            wrapper=UNIVERSE[0].wrapper,
            distributing=False,
        ),
    )
    quelle = FakePriceSource({"TEST.DE": _monatsreihe([100.0, 101.0])})

    fetch_prices_eur(quelle, universum)

    assert quelle.abgefragt == ["TEST.DE"]


def test_monatsrenditen_werden_korrekt_gebildet() -> None:
    """Aus Kursen entstehen einfache Renditen; die erste Zeile entfällt."""
    kurse = pd.DataFrame({"A": _monatsreihe([100.0, 110.0, 99.0])})

    renditen = to_monthly_returns(kurse)

    assert len(renditen) == 2
    assert list(renditen["A"]) == pytest.approx([0.10, -0.10])


# --- Lückenerkennung -------------------------------------------------------


def test_luecke_innerhalb_der_reihe_wird_erkannt() -> None:
    """Ein fehlender Monat mitten in der Reihe wird als Befund gemeldet.

    Das ist der Test zu dem Fehler, der die Gold-Reihe beschädigt hatte:
    Ohne Umrechnung in die Börsenzeitzone fehlten Oktober und November jedes
    Jahres.
    """
    reihe = _monatsreihe([1.0, 2.0, 3.0, 4.0])
    mit_luecke = reihe.drop(reihe.index[2])

    befunde = find_internal_gaps(pd.DataFrame({"A": mit_luecke}))

    assert len(befunde) == 1
    assert "fehlende(r) Monat(e)" in befunde[0]


def test_fehlender_monat_am_anfang_ist_kein_befund() -> None:
    """Ein später einsetzendes Instrument gilt nicht als Lücke."""
    lang = _monatsreihe([1.0, 2.0, 3.0, 4.0])
    kurz = lang.copy()
    kurz.iloc[:2] = np.nan

    assert find_internal_gaps(pd.DataFrame({"A": lang, "B": kurz})) == []


# --- Snapshot und Manifest -------------------------------------------------


def test_snapshot_wird_nicht_stillschweigend_ueberschrieben(out_dir: Path) -> None:
    """Ein vorhandener Snapshot wird ohne ``force`` nicht überschrieben.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    quelle = FakePriceSource(
        {i.symbol: _monatsreihe([100.0, 101.0, 102.0]) for i in UNIVERSE}
        | {"EURUSD=X": _monatsreihe([1.0, 1.0, 1.0])}
    )
    build_snapshot(out_dir, quelle, UNIVERSE)

    with pytest.raises(SnapshotError, match="besteht bereits"):
        build_snapshot(out_dir, quelle, UNIVERSE)


def test_snapshot_und_manifest_entstehen_vollstaendig(out_dir: Path) -> None:
    """Der Snapshot erzeugt beide Parquet-Dateien und ein prüfbares Manifest.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    quelle = FakePriceSource(
        {i.symbol: _monatsreihe([100.0, 105.0, 103.0]) for i in UNIVERSE}
        | {"EURUSD=X": _monatsreihe([1.0, 1.0, 1.0])}
    )

    manifest_pfad = build_snapshot(out_dir, quelle, UNIVERSE)
    manifest = json.loads(manifest_pfad.read_text(encoding="utf-8"))

    assert (out_dir / PRICES_FILE).is_file()
    assert (out_dir / RETURNS_FILE).is_file()
    assert {eintrag["filename"] for eintrag in manifest["files"]} == {PRICES_FILE, RETURNS_FILE}
    assert manifest["source"] == quelle.label
    assert manifest["package_versions"]["candor"]
    assert len(manifest["universe"]) == len(UNIVERSE)
    assert manifest["known_limitations"], "Bekannte Einschränkungen fehlen im Manifest"
    assert verify_snapshot(out_dir).ok


def test_manifest_erkennt_veraenderte_datei(out_dir: Path) -> None:
    """Jede Veränderung einer Snapshot-Datei wird über den Hash erkannt.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    quelle = FakePriceSource(
        {i.symbol: _monatsreihe([100.0, 105.0, 103.0]) for i in UNIVERSE}
        | {"EURUSD=X": _monatsreihe([1.0, 1.0, 1.0])}
    )
    build_snapshot(out_dir, quelle, UNIVERSE)

    with (out_dir / RETURNS_FILE).open("ab") as datei:
        datei.write(b"\x00")

    ergebnis = verify_snapshot(out_dir)

    assert not ergebnis.ok
    assert ergebnis.problems
    assert "Hash weicht ab" in ergebnis.problems[0]
    assert ergebnis.render().startswith("FEHLER")


def test_manifest_erkennt_fehlende_datei(out_dir: Path) -> None:
    """Eine gelöschte Snapshot-Datei wird als fehlend gemeldet.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    quelle = FakePriceSource(
        {i.symbol: _monatsreihe([100.0, 105.0, 103.0]) for i in UNIVERSE}
        | {"EURUSD=X": _monatsreihe([1.0, 1.0, 1.0])}
    )
    build_snapshot(out_dir, quelle, UNIVERSE)
    (out_dir / PRICES_FILE).unlink()

    ergebnis = verify_snapshot(out_dir)

    assert not ergebnis.ok
    assert any("Datei fehlt" in problem for problem in ergebnis.problems)


def test_fehlendes_manifest_wirft_fehler(out_dir: Path) -> None:
    """Ohne Manifest bricht die Prüfung mit klarer Meldung ab (Prinzip P5).

    Args:
        out_dir: Temporäres, leeres Verzeichnis.
    """
    with pytest.raises(ManifestError, match="Kein Manifest"):
        verify_snapshot(out_dir)


def test_hash_ist_stabil(out_dir: Path) -> None:
    """Derselbe Inhalt ergibt denselben SHA-256-Hash.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    erste = out_dir / "a.bin"
    zweite = out_dir / "b.bin"
    erste.write_bytes(b"CANDOR")
    zweite.write_bytes(b"CANDOR")

    assert sha256_of(erste) == sha256_of(zweite)


# --- Loader ----------------------------------------------------------------


def test_loader_geht_ohne_live_flag_nicht_ins_netz(out_dir: Path) -> None:
    """Ohne ``live=True`` wird ausschließlich der Snapshot gelesen.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    quelle = FakePriceSource(
        {i.symbol: _monatsreihe([100.0, 105.0, 103.0]) for i in UNIVERSE}
        | {"EURUSD=X": _monatsreihe([1.0, 1.0, 1.0])}
    )
    build_snapshot(out_dir, quelle, UNIVERSE)

    renditen = load_returns(snapshot_dir=out_dir, source=ExplodingPriceSource())

    assert list(renditen.columns) == list(ASSET_KEYS)


def test_fehlender_snapshot_wirft_fehler(out_dir: Path) -> None:
    """Fehlt der Snapshot, gibt es keinen stillen Rückfall auf einen Live-Abruf.

    Args:
        out_dir: Temporäres, leeres Verzeichnis.
    """
    with pytest.raises(SnapshotMissingError, match="Snapshot-Datei fehlt"):
        load_returns(snapshot_dir=out_dir)


# --- Kennzahlen ------------------------------------------------------------


def test_deutsches_dezimalkomma() -> None:
    """Zahlen werden mit deutschem Dezimalkomma statt mit Punkt ausgegeben."""
    assert format_de(1234.5678) == "1234,57"
    assert format_de(0.0742, 2, percent=True) == "7,42 %"
    assert format_de(-0.31) == "-0,31"
    assert format_de(float("nan")) == "–"


def test_maximaler_drawdown_gegen_bekanntes_beispiel() -> None:
    """Der Drawdown wird gegen einen von Hand nachrechenbaren Fall geprüft."""
    # Index: 1,00 → 1,20 → 0,60 → 0,90. Tiefster Stand 0,60 nach Hoch 1,20.
    renditen = pd.Series([0.20, -0.50, 0.50])

    assert max_drawdown(renditen) == pytest.approx(-0.50)
    assert max_drawdown(pd.Series([0.01, 0.02])) == pytest.approx(0.0)
    assert max_drawdown(pd.Series(dtype="float64")) == 0.0


def test_annualisierung_ist_geometrisch() -> None:
    """Zwölf Monate mit je 1 % ergeben 12,68 % p. a., nicht 12 %."""
    renditen = pd.DataFrame({"A": [0.01] * 12})

    kennzahlen = compute_stats(renditen)[0]

    assert kennzahlen.annual_return == pytest.approx(1.01**12 - 1)
    assert kennzahlen.months == 12


def test_sharpe_beruecksichtigt_den_risikofreien_zins() -> None:
    """Ein höherer risikofreier Zins senkt die Sharpe Ratio."""
    renditen = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.03] * 6})

    ohne = compute_stats(renditen, risk_free_rate=0.0)[0]
    mit = compute_stats(renditen, risk_free_rate=0.05)[0]

    assert mit.sharpe < ohne.sharpe


def test_korrelationsmatrix_ist_symmetrisch_mit_einsen(rng: np.random.Generator) -> None:
    """Die Korrelationsmatrix hat eine Einsdiagonale und ist symmetrisch.

    Args:
        rng: Zufallsgenerator mit dem Projekt-Seed (Prinzip P1).
    """
    daten = pd.DataFrame(rng.normal(size=(60, 3)), columns=["A", "B", "C"])

    matrix = correlation_matrix(daten)

    assert np.allclose(np.diag(matrix.to_numpy()), 1.0)
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)


# --- Der ausgelieferte Snapshot -------------------------------------------


def test_ausgelieferter_snapshot_ist_unveraendert() -> None:
    """Der im Repository liegende Snapshot stimmt mit seinem Manifest überein.

    Dieser Test ist der Reproduzierbarkeitsnachweis gegenüber jedem, der die
    Ergebnisse später nachrechnet.
    """
    assert verify_snapshot().ok


def test_ausgelieferter_snapshot_deckt_die_krisenjahre_ab() -> None:
    """Der Snapshot enthält 2008, 2020 und 2022 und weist keine inneren Lücken auf."""
    kurse = load_prices()
    jahre = set(pd.DatetimeIndex(kurse.index).year)

    assert {2008, 2020, 2022} <= jahre
    assert find_internal_gaps(kurse) == []
    assert list(kurse.columns) == list(ASSET_KEYS)


def test_snapshot_herkunft_ist_vollstaendig() -> None:
    """Die Herkunftsangaben für Traces und Berichte sind vollständig belegt."""
    info = snapshot_info()

    assert info["source"]
    assert len(info["sha256_short"]) == 12
    assert re.match(r"^\d{4}-\d{2}$", info["period_start"])
    assert re.match(r"^\d{4}-\d{2}$", info["period_end"])


def test_manifest_weist_die_bekannten_einschraenkungen_aus() -> None:
    """Die gemessenen Datenmängel stehen im Manifest und gehen nicht verloren."""
    manifest = read_manifest(Path(__file__).resolve().parent.parent / "data" / "snapshot")
    kennungen = {eintrag["id"] for eintrag in manifest["known_limitations"]}

    assert {"L-DIST", "L-START"} <= kennungen


# --- Live-Abruf (nur auf Anforderung) --------------------------------------


@pytest.mark.netz
@pytest.mark.skipif(
    os.environ.get("CANDOR_NETZTEST") != "1",
    reason="Live-Abruf nur mit CANDOR_NETZTEST=1",
)
def test_live_abruf_liefert_dieselben_spalten() -> None:
    """Ein Live-Abruf liefert dieselbe Spaltenstruktur wie der Snapshot."""
    live = load_returns(live=True)

    assert list(live.columns) == list(ASSET_KEYS)


def test_snapshot_kopie_bleibt_pruefbar(out_dir: Path) -> None:
    """Eine unveränderte Kopie des Snapshots besteht die Prüfung ebenfalls.

    Belegt, dass die Prüfung am Inhalt hängt und nicht am Ablageort.

    Args:
        out_dir: Temporäres Ausgabeverzeichnis.
    """
    quell_verzeichnis = Path(__file__).resolve().parent.parent / "data" / "snapshot"
    ziel = out_dir / "kopie"
    shutil.copytree(quell_verzeichnis, ziel)

    assert verify_snapshot(ziel).ok
