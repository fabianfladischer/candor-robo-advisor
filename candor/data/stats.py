"""Deskriptive Kennzahlen je Assetklasse und Korrelationsmatrix.

Die Kennzahlen beschreiben den eingefrorenen Datenstand und dienen zweierlei:
als Plausibilitätsprüfung der Datenschicht und als Grundlage der
Methodenauswertung.

Berechnungskonventionen — jede Zahl der Ausgabe ist damit nachrechenbar:

* **Annualisierte Rendite:** geometrisch, ``(∏(1+r))^(12/n) − 1``. Nicht das
  arithmetische Mittel mal zwölf; dieses überschätzt den tatsächlich erzielten
  Vermögensendwert umso stärker, je volatiler die Reihe ist.
* **Volatilität:** Standardabweichung der Monatsrenditen mit ``ddof=1``,
  skaliert mit ``√12``.
* **Sharpe Ratio:** ``(annualisierte Rendite − risikofreier Zins) / Volatilität``.
  Der risikofreie Zins ist voreingestellt **0 %** und damit eine Konvention,
  keine Messung. Über den Zeitraum ab 2008 lagen die Euro-Geldmarktsätze zeitweise
  deutlich darüber und zeitweise im negativen Bereich; die ausgewiesenen Werte
  sind deshalb als Rangordnung zwischen den Assetklassen zu lesen, nicht als
  absolute Risikoprämie.
* **Maximaler Drawdown:** größter relativer Rückgang eines aus den Monatsrenditen
  aufgebauten Indexes gegenüber seinem bisherigen Höchststand. Da die Reihen im
  Januar 2008 beginnen, wird der Höchststand von Oktober 2007 nicht erfasst; die
  Drawdowns der Finanzkrise fallen dadurch **zu niedrig** aus.
* **Schiefe und Wölbung:** Stichprobenschiefe und Exzess-Wölbung nach Fisher
  (Normalverteilung ⇒ 0). Beide belegen, warum ein Block-Bootstrap und
  keine Normalverteilungsannahme verwendet wird.

Aufruf:
    python -m candor.data.stats [--write-evaluation] [--risk-free 0.0]
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd

from candor.data.loader import load_returns, snapshot_info
from candor.data.universe_meta import get_instrument

#: Monate je Jahr — Skalierungsfaktor der Annualisierung.
MONTHS_PER_YEAR: Final = 12

#: Markierungen des erzeugten Abschnitts in der Kennzahlendatei.
BLOCK_START: Final = "<!-- BEGIN:KENNZAHLEN -->"
BLOCK_END: Final = "<!-- END:KENNZAHLEN -->"


def format_de(value: float, decimals: int = 2, *, percent: bool = False) -> str:
    """Formatiert eine Zahl mit deutschem Dezimalkomma.

    Args:
        value: Zu formatierender Wert.
        decimals: Zahl der Nachkommastellen.
        percent: Multipliziert mit 100 und hängt ein Prozentzeichen an.

    Returns:
        Der formatierte Wert, etwa ``"7,42 %"`` oder ``"-0,31"``.
    """
    if not np.isfinite(value):
        return "–"
    zahl = value * 100 if percent else value
    text = f"{zahl:.{decimals}f}".replace(".", ",")
    return f"{text} %" if percent else text


@dataclass(frozen=True, slots=True)
class AssetStats:
    """Kennzahlen einer Assetklasse.

    Attributes:
        key: Kennung der Assetklasse.
        label: Deutsche Bezeichnung.
        months: Zahl der ausgewerteten Monatsrenditen.
        period_start: Erster ausgewerteter Monat (``YYYY-MM``).
        period_end: Letzter ausgewerteter Monat (``YYYY-MM``).
        annual_return: Geometrisch annualisierte Rendite.
        annual_volatility: Annualisierte Standardabweichung.
        sharpe: Sharpe Ratio gegen den übergebenen risikofreien Zins.
        max_drawdown: Größter relativer Rückgang (negativer Wert).
        skewness: Stichprobenschiefe der Monatsrenditen.
        excess_kurtosis: Exzess-Wölbung nach Fisher.
    """

    key: str
    label: str
    months: int
    period_start: str
    period_end: str
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    skewness: float
    excess_kurtosis: float


def max_drawdown(returns: pd.Series) -> float:
    """Berechnet den maximalen Drawdown einer Renditereihe.

    Args:
        returns: Monatsrenditen ohne fehlende Werte.

    Returns:
        Größter relativer Rückgang als negative Zahl. ``0.0`` bei leerer Reihe.
    """
    if returns.empty:
        return 0.0
    index = (1.0 + returns).cumprod()
    hoechststand = index.cummax()
    return float((index / hoechststand - 1.0).min())


def compute_stats(returns: pd.DataFrame, risk_free_rate: float = 0.0) -> list[AssetStats]:
    """Berechnet die Kennzahlen aller Assetklassen.

    Args:
        returns: Renditetabelle mit Monatsenden als Index.
        risk_free_rate: Annualisierter risikofreier Zins für die Sharpe Ratio.

    Returns:
        Kennzahlen in Spaltenreihenfolge der Tabelle.
    """
    ergebnisse: list[AssetStats] = []
    for spalte in returns.columns:
        reihe = returns[spalte].dropna()
        key = str(spalte)
        if reihe.empty:
            ergebnisse.append(
                AssetStats(key, key, 0, "–", "–", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            )
            continue

        n = int(len(reihe))
        gesamtfaktor = cast("float", (1.0 + reihe).prod())
        annualisiert = gesamtfaktor ** (MONTHS_PER_YEAR / n) - 1.0
        volatilitaet = float(reihe.std(ddof=1)) * float(np.sqrt(MONTHS_PER_YEAR))
        sharpe = (annualisiert - risk_free_rate) / volatilitaet if volatilitaet > 0 else np.nan
        index = pd.DatetimeIndex(reihe.index)

        try:
            label = get_instrument(key).label
        except KeyError:
            label = key

        ergebnisse.append(
            AssetStats(
                key=key,
                label=label,
                months=n,
                period_start=index.min().strftime("%Y-%m"),
                period_end=index.max().strftime("%Y-%m"),
                annual_return=annualisiert,
                annual_volatility=volatilitaet,
                sharpe=sharpe,
                max_drawdown=max_drawdown(reihe),
                skewness=cast("float", reihe.skew()),
                excess_kurtosis=cast("float", reihe.kurtosis()),
            )
        )
    return ergebnisse


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Berechnet die Korrelationsmatrix der Monatsrenditen.

    Args:
        returns: Renditetabelle.

    Returns:
        Symmetrische Matrix der Pearson-Korrelationen über den gemeinsamen
        Zeitraum je Paar.
    """
    return returns.corr()


def render_stats_table(stats: list[AssetStats], risk_free_rate: float) -> str:
    """Erzeugt die Markdown-Tabelle der Kennzahlen.

    Args:
        stats: Berechnete Kennzahlen.
        risk_free_rate: Verwendeter risikofreier Zins, für die Spaltenüberschrift.

    Returns:
        Markdown-Tabelle mit deutschem Dezimalkomma.
    """
    kopf = (
        f"| Assetklasse | Monate | Zeitraum | Rendite p. a. | Volatilität p. a. | "
        f"Sharpe (rf = {format_de(risk_free_rate, 1, percent=True)}) | Max. Drawdown | "
        f"Schiefe | Wölbung |"
    )
    trenner = "|---|---:|---|---:|---:|---:|---:|---:|---:|"
    zeilen = [kopf, trenner]
    for eintrag in stats:
        zeilen.append(
            f"| {eintrag.label} (`{eintrag.key}`) "
            f"| {eintrag.months} "
            f"| {eintrag.period_start}–{eintrag.period_end} "
            f"| {format_de(eintrag.annual_return, 2, percent=True)} "
            f"| {format_de(eintrag.annual_volatility, 2, percent=True)} "
            f"| {format_de(eintrag.sharpe)} "
            f"| {format_de(eintrag.max_drawdown, 2, percent=True)} "
            f"| {format_de(eintrag.skewness)} "
            f"| {format_de(eintrag.excess_kurtosis)} |"
        )
    return "\n".join(zeilen)


def render_correlation_table(matrix: pd.DataFrame) -> str:
    """Erzeugt die Markdown-Tabelle der Korrelationsmatrix.

    Args:
        matrix: Korrelationsmatrix.

    Returns:
        Markdown-Tabelle mit deutschem Dezimalkomma.
    """
    spalten = [str(spalte) for spalte in matrix.columns]
    zeilen = ["| | " + " | ".join(f"`{name}`" for name in spalten) + " |"]
    zeilen.append("|---|" + "---:|" * len(spalten))
    for name in spalten:
        werte = " | ".join(format_de(cast("float", matrix.loc[name, spalte])) for spalte in spalten)
        zeilen.append(f"| **`{name}`** | {werte} |")
    return "\n".join(zeilen)


def render_block(returns: pd.DataFrame, risk_free_rate: float, info: dict[str, str]) -> str:
    """Erzeugt den vollständigen Abschnitt für die Kennzahlendatei.

    Args:
        returns: Renditetabelle.
        risk_free_rate: Verwendeter risikofreier Zins.
        info: Herkunftsangaben des Snapshots.

    Returns:
        Markdown-Abschnitt einschließlich der Blockmarkierungen.
    """
    stats = compute_stats(returns, risk_free_rate)
    erzeugt = dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d")
    return "\n".join(
        [
            BLOCK_START,
            "",
            f"_Erzeugt am {erzeugt} durch `make stats`. Nicht von Hand bearbeiten —",
            "der Abschnitt wird bei jedem Lauf vollständig ersetzt._",
            "",
            f"**Datenstand:** {info['period_start']} bis {info['period_end']} · "
            f"{info['rows']} Monatsrenditen · Snapshot-Kurzhash `{info['sha256_short']}` · "
            f"abgerufen {info['fetched_at']}",
            "",
            f"**Quelle:** {info['source']}",
            "",
            "### Kennzahlen je Assetklasse",
            "",
            render_stats_table(stats, risk_free_rate),
            "",
            "> **Einschränkung des Datenstands — bei jeder Interpretation mitzulesen.**",
            "> Die Quelle führt Ausschüttungen erst ab 2013 bzw. 2014. Für die Jahre",
            "> 2008 bis 2013 bereinigt der ausschüttungsbereinigte Kurs deshalb nichts;",
            "> die Renditen dieser Jahre sind reine Kursrenditen. Die ausgewiesene",
            "> Rendite p. a. **unterschätzt** die tatsächliche Gesamtrendite aller",
            "> Positionen außer Gold. Nachgewiesen an `EXHB.DE`: Bereinigungsfaktor über",
            "> 2009 konstant 0,8541, Renditen aus `adjclose` und `close` identisch.",
            "> Zweite Einschränkung: Die Historie beginnt 2008-02, der Markthöchststand",
            "> vom Oktober 2007 liegt davor — die Drawdowns der Finanzkrise fallen",
            "> dadurch zu niedrig aus. Beide Befunde stehen im Manifest unter",
            "> `known_limitations`.",
            "",
            "Rendite geometrisch annualisiert, Volatilität mit `√12` skaliert, Wölbung als",
            "Exzess nach Fisher (Normalverteilung ⇒ 0). Der risikofreie Zins ist eine",
            "gesetzte Konvention, keine Messung — die Sharpe-Werte sind als Rangordnung",
            "zwischen den Assetklassen zu lesen.",
            "",
            "### Korrelationsmatrix der Monatsrenditen",
            "",
            render_correlation_table(correlation_matrix(returns)),
            "",
            BLOCK_END,
        ]
    )


def write_evaluation(block: str, evaluation_path: Path | None = None) -> Path:
    """Schreibt den Kennzahlenabschnitt in die Kennzahlendatei.

    Ein bereits vorhandener Abschnitt zwischen den Blockmarkierungen wird
    ersetzt, sodass wiederholte Läufe die Datei nicht anwachsen lassen.

    Args:
        block: Der einzusetzende Abschnitt.
        evaluation_path: Abweichender Pfad der Zieldatei.

    Returns:
        Pfad der geschriebenen Datei.

    Raises:
        FileNotFoundError: Wenn die Zieldatei nicht existiert.
    """
    ziel = evaluation_path or (Path(__file__).resolve().parents[2] / "kennzahlen.md")
    if not ziel.is_file():
        raise FileNotFoundError(f"Zieldatei fehlt: {ziel}")

    inhalt = ziel.read_text(encoding="utf-8")
    if BLOCK_START in inhalt and BLOCK_END in inhalt:
        vorher = inhalt.split(BLOCK_START)[0]
        nachher = inhalt.split(BLOCK_END, 1)[1]
        neu = f"{vorher}{block}{nachher}"
    else:
        neu = f"{inhalt.rstrip()}\n\n---\n\n## Deskriptive Kennzahlen der Datenschicht\n\n{block}\n"
    ziel.write_text(neu, encoding="utf-8")
    return ziel


def main(argv: list[str] | None = None) -> int:
    """Einstiegspunkt der Kommandozeile.

    Args:
        argv: Argumente ohne Programmnamen.

    Returns:
        ``0`` bei Erfolg, ``1`` wenn der Snapshot fehlt.
    """
    parser = argparse.ArgumentParser(description="Deskriptive Kennzahlen des Snapshots.")
    parser.add_argument(
        "--write-evaluation", action="store_true", help="Ergebnis in die Kennzahlendatei schreiben"
    )
    parser.add_argument(
        "--risk-free", type=float, default=0.0, help="annualisierter risikofreier Zins (Vorgabe 0)"
    )
    args = parser.parse_args(argv)

    try:
        returns = load_returns()
        info = snapshot_info()
    except (FileNotFoundError, RuntimeError) as fehler:
        print(f"FEHLER — {fehler}")
        return 1

    block = render_block(returns, args.risk_free, info)
    if args.write_evaluation:
        ziel = write_evaluation(block)
        print(f"Kennzahlen geschrieben nach {ziel}")
    else:
        print(block)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
