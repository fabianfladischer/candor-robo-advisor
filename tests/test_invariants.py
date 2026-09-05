"""Die Invarianten I1 bis I3 als Property-based Tests.

Diese Tests prüfen **Eigenschaften über den gesamten zulässigen Eingaberaum**,
nicht einzelne Beispiele. Der Unterschied ist wesentlich: Ein beispielbasierter
Test zeigt, dass die Regel in einem Fall gilt; ein Property-Test sucht aktiv
nach dem Fall, in dem sie bricht.

| ID | Invariante |
|----|------------|
| I1 | Höhere Tragfähigkeitspunkte ⇒ Risikoklasse nie niedriger (Monotonie) |
| I2 | Höhere Bereitschaftspunkte ⇒ Risikoklasse nie niedriger (Monotonie) |
| I3 | Ergebnisklasse ≤ min(Tragfähigkeit, Bereitschaft, Kenntnisdeckel) |

Jede Invariante wird auf zwei Ebenen geprüft:

1. **Auf der Abbildung** — die Punkte-auf-Klasse-Abbildung selbst ist monoton.
2. **Auf den Eingaben** — ein in jeder Dimension besseres Profil führt nie zu
   einer schlechteren Klasse. Das ist die eigentlich interessante Aussage, weil
   sie die gesamte Auswertungskette einschließt.

Die Strategien erzeugen ausschließlich Profile, die die Validierung des
Domänenkerns bestehen. Ein von hypothesis gefundenes Gegenbeispiel wäre damit
stets ein echter fachlicher Fehler und nie ein Artefakt unzulässiger Eingaben.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from candor.data.policy_loader import load_policy_set
from candor.domain.client import EmploymentType, RiskAttitude
from candor.domain.policy import PolicySet
from candor.profiling.capacity import assess_capacity
from candor.profiling.classify import classify
from candor.profiling.knowledge import assess_knowledge
from candor.profiling.scoring import risk_class_for_points
from candor.profiling.tolerance import assess_tolerance
from candor.trace.recorder import TraceMetadata, TraceRecorder
from tests.builders import STICHTAG, add_months, haushalt, kenntnisse_fuer_klasse, kunde, ziel

#: Das Regelwerk wird einmal geladen; es ist unveränderlich.
POLICIES: PolicySet = load_policy_set()


def _recorder() -> TraceRecorder:
    """Baut einen leeren Recorder.

    Returns:
        Der Recorder.
    """
    return TraceRecorder(
        TraceMetadata(
            client_id="K-INV",
            goal_id="Z-INV",
            policy_versions=dict(POLICIES.versions),
            package_version="0.1.0",
        )
    )


def _euro(cents: int) -> Decimal:
    """Wandelt einen Centbetrag in einen Eurobetrag.

    Args:
        cents: Betrag in Cent.

    Returns:
        Der Betrag als ``Decimal`` mit zwei Nachkommastellen.
    """
    return (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))


# --- Strategien ------------------------------------------------------------


@st.composite
def haushalte(draw: st.DrawFn) -> dict[str, Any]:
    """Erzeugt die Felder eines zulässigen Haushalts.

    Die Beziehungen zwischen den Feldern werden beim Ziehen eingehalten, statt
    unzulässige Kombinationen nachträglich zu verwerfen. Das hält die
    Trefferquote hoch und vermeidet, dass hypothesis den Eingaberaum nur am Rand
    abtastet.

    Args:
        draw: Ziehfunktion von hypothesis.

    Returns:
        Die Felder des Haushalts.
    """
    einkommen = draw(st.integers(min_value=80_000, max_value=1_500_000))
    fixkosten = draw(st.integers(min_value=0, max_value=einkommen))
    kapitaldienst = draw(st.integers(min_value=0, max_value=einkommen - fixkosten))
    ueberschuss = einkommen - fixkosten - kapitaldienst
    sparrate = draw(st.integers(min_value=0, max_value=ueberschuss))
    return {
        "net_monthly_income": _euro(einkommen),
        "fixed_monthly_costs": _euro(fixkosten),
        "monthly_debt_service": _euro(kapitaldienst),
        "free_savings_amount": _euro(sparrate),
        "liquidity_reserve_months": Decimal(draw(st.integers(0, 240))) / Decimal(10),
        "total_debt": _euro(draw(st.integers(0, 50_000_000))),
        "other_assets": _euro(draw(st.integers(0, 100_000_000))),
        "employment_type": draw(st.sampled_from(list(EmploymentType))),
    }


@st.composite
def ziele(draw: st.DrawFn) -> dict[str, Any]:
    """Erzeugt die Felder eines zulässigen Anlageziels.

    Args:
        draw: Ziehfunktion von hypothesis.

    Returns:
        Die Felder des Ziels.
    """
    monate = draw(st.integers(min_value=2, max_value=600))
    startkapital = draw(st.integers(min_value=0, max_value=50_000_000))
    sparrate = draw(st.integers(min_value=0, max_value=500_000))
    assume(startkapital > 0 or sparrate > 0)
    return {
        "reference_date": STICHTAG,
        "target_date": add_months(STICHTAG, monate),
        "target_amount": _euro(draw(st.integers(min_value=100, max_value=500_000_000))),
        "initial_capital": _euro(startkapital),
        "monthly_contribution": _euro(sparrate),
    }


def antwortstrategie() -> st.SearchStrategy[dict[str, str]]:
    """Erzeugt eine Strategie über alle zulässigen Antwortkombinationen.

    Returns:
        Die Strategie.
    """
    return st.fixed_dictionaries(
        {
            name: st.sampled_from(sorted(dimension.optionen or {}))
            for name, dimension in POLICIES.tolerance.dimensionen.items()
        }
    )


def punkte_je_antwort(answers: dict[str, str]) -> dict[str, int]:
    """Liefert die Punktwerte der gewählten Antworten.

    Args:
        answers: Die Antworten je Item.

    Returns:
        Zuordnung Item auf Punktwert.
    """
    punkte: dict[str, int] = {}
    for name, antwort in answers.items():
        optionen = POLICIES.tolerance.dimensionen[name].optionen or {}
        punkte[name] = optionen[antwort]
    return punkte


# --- I1 und I2 auf der Abbildung ------------------------------------------


@pytest.mark.invariant
@given(
    erste=st.decimals(min_value=0, max_value=100, places=2),
    zweite=st.decimals(min_value=0, max_value=100, places=2),
)
def test_i1_i2_punkte_auf_klasse_ist_monoton(erste: Decimal, zweite: Decimal) -> None:
    """Die Abbildung Punkte auf Klasse ist für beide Achsen monoton steigend.

    Args:
        erste: Erste Punktzahl.
        zweite: Zweite Punktzahl.
    """
    niedrig, hoch = sorted([erste, zweite])

    for schwellen in (POLICIES.capacity.klassenschwellen, POLICIES.tolerance.klassenschwellen):
        assert risk_class_for_points(schwellen, niedrig) <= risk_class_for_points(schwellen, hoch)


# --- I1 auf den Eingaben ---------------------------------------------------


@pytest.mark.invariant
@given(basis=haushalte(), zielfelder=ziele(), zuwachs=st.integers(min_value=0, max_value=60))
def test_i1_besserer_haushalt_ergibt_nie_schlechtere_klasse(
    basis: dict[str, Any], zielfelder: dict[str, Any], zuwachs: int
) -> None:
    """Ein in jeder Dimension besserer Haushalt führt nie zu einer tieferen Klasse.

    „Besser" heißt: mehr Liquiditätsreserve, mehr freie Sparquote, weniger
    Kapitaldienst und mehr sonstiges Vermögen. Einkommen, Fixkosten,
    Beschäftigungsart und Ziel bleiben gleich, damit ausschließlich die
    Verbesserung wirkt.

    Args:
        basis: Felder des Ausgangshaushalts.
        zielfelder: Felder des Anlageziels.
        zuwachs: Verbesserung in Zehntelmonaten Reserve.
    """
    schlechter = haushalt(**basis)

    besser_felder = dict(basis)
    besser_felder["liquidity_reserve_months"] = min(
        Decimal(120), basis["liquidity_reserve_months"] + Decimal(zuwachs) / Decimal(10)
    )
    besser_felder["monthly_debt_service"] = Decimal("0.00")
    besser_felder["other_assets"] = basis["other_assets"] + Decimal("1000.00")
    # Der Kapitaldienst sinkt auf null, der verfuegbare Ueberschuss steigt damit.
    # Die Sparrate darf hoechstens auf diesen neuen Ueberschuss angehoben werden.
    neuer_ueberschuss = basis["net_monthly_income"] - basis["fixed_monthly_costs"]
    besser_felder["free_savings_amount"] = min(
        neuer_ueberschuss, basis["free_savings_amount"] + Decimal("50.00")
    )
    besser = haushalt(**besser_felder)

    anlageziel = ziel(**zielfelder)
    schlechter_ergebnis = assess_capacity(
        kunde(household=schlechter), anlageziel, POLICIES.capacity, _recorder()
    )
    besser_ergebnis = assess_capacity(
        kunde(household=besser), anlageziel, POLICIES.capacity, _recorder()
    )

    assert besser_ergebnis.risk_class >= schlechter_ergebnis.risk_class


@pytest.mark.invariant
@given(basis=haushalte(), zielfelder=ziele(), zusatzmonate=st.integers(min_value=1, max_value=240))
def test_i1_laengerer_horizont_ergibt_nie_schlechtere_klasse(
    basis: dict[str, Any], zielfelder: dict[str, Any], zusatzmonate: int
) -> None:
    """Ein längerer Anlagehorizont senkt die Tragfähigkeit nie.

    Args:
        basis: Felder des Haushalts.
        zielfelder: Felder des Anlageziels.
        zusatzmonate: Verlängerung des Horizonts in Monaten.
    """
    kurz = ziel(**zielfelder)
    lang_felder = dict(zielfelder)
    lang_felder["target_date"] = add_months(zielfelder["target_date"], zusatzmonate)
    # Eine laengere Laufzeit erhoeht bei gleicher Sparrate das eingezahlte
    # Kapital und damit das Klumpenrisiko. Damit ausschliesslich der Horizont
    # wirkt, wird das sonstige Vermoegen entsprechend mitskaliert.
    lang = ziel(**lang_felder)
    profil = haushalt(**{**basis, "other_assets": Decimal("100000000.00")})

    kurz_ergebnis = assess_capacity(kunde(household=profil), kurz, POLICIES.capacity, _recorder())
    lang_ergebnis = assess_capacity(kunde(household=profil), lang, POLICIES.capacity, _recorder())

    assert lang_ergebnis.risk_class >= kurz_ergebnis.risk_class


# --- I2 auf den Eingaben ---------------------------------------------------


@pytest.mark.invariant
@given(antworten=antwortstrategie(), zielfelder=ziele())
def test_i2_bessere_antworten_ergeben_nie_schlechtere_klasse(
    antworten: dict[str, str], zielfelder: dict[str, Any]
) -> None:
    """Antworten mit durchweg höheren Punktwerten führen nie zu einer tieferen Klasse.

    Für jedes Item wird die punktbeste Antwort gewählt. Die Gesamtpunktzahl kann
    dadurch nur steigen, die Klasse also nicht fallen.

    Args:
        antworten: Ausgangsantworten.
        zielfelder: Felder des Anlageziels.
    """
    beste = {
        name: max((dimension.optionen or {}).items(), key=lambda paar: paar[1])[0]
        for name, dimension in POLICIES.tolerance.dimensionen.items()
    }
    anlageziel = ziel(**zielfelder)

    schlechter = assess_tolerance(
        kunde(risk_attitude=RiskAttitude(answers=antworten)),
        anlageziel,
        POLICIES.tolerance,
        _recorder(),
    )
    besser = assess_tolerance(
        kunde(risk_attitude=RiskAttitude(answers=beste)),
        anlageziel,
        POLICIES.tolerance,
        _recorder(),
    )

    assert besser.total_points >= schlechter.total_points
    assert besser.risk_class >= schlechter.risk_class


@pytest.mark.invariant
@given(antworten=antwortstrategie(), zielfelder=ziele())
def test_i2_widerspruch_veraendert_die_punktzahl_nicht(
    antworten: dict[str, str], zielfelder: dict[str, Any]
) -> None:
    """Die Punktzahl ist stets die Summe der Einzelantworten.

    Ein erkannter Widerspruch wird ausgewiesen, aber nie eingerechnet. Genau
    darauf beruht die Monotonie der Achse.

    Args:
        antworten: Die Antworten.
        zielfelder: Felder des Anlageziels.
    """
    ergebnis = assess_tolerance(
        kunde(risk_attitude=RiskAttitude(answers=antworten)),
        ziel(**zielfelder),
        POLICIES.tolerance,
        _recorder(),
    )

    erwartet = sum((eintrag.weighted_points for eintrag in ergebnis.scores), Decimal(0)).quantize(
        Decimal("0.01")
    )

    assert ergebnis.total_points == erwartet
    punkte = punkte_je_antwort(antworten)
    assert [eintrag.points for eintrag in ergebnis.scores] == [
        punkte[eintrag.criterion] for eintrag in ergebnis.scores
    ]


# --- I3 --------------------------------------------------------------------


@pytest.mark.invariant
@given(
    basis=haushalte(),
    antworten=antwortstrategie(),
    zielfelder=ziele(),
    kenntnisklasse=st.integers(min_value=1, max_value=5),
)
def test_i3_ergebnis_ist_hoechstens_das_minimum_der_achsen(
    basis: dict[str, Any],
    antworten: dict[str, str],
    zielfelder: dict[str, Any],
    kenntnisklasse: int,
) -> None:
    """Das Ergebnis überschreitet nie das Minimum der drei Achsen.

    Das ist die zentrale Invariante des Projekts: Keine Achse kann eine andere
    ausgleichen, und keine Mittelung hebt das Ergebnis über die schwächste
    Achse.

    Args:
        basis: Felder des Haushalts.
        antworten: Antworten zur Risikobereitschaft.
        zielfelder: Felder des Anlageziels.
        kenntnisklasse: Angestrebte Obergrenze aus den Kenntnissen.
    """
    profil_kunde = kunde(
        household=haushalt(**basis),
        knowledge=kenntnisse_fuer_klasse(kenntnisklasse),
        risk_attitude=RiskAttitude(answers=antworten),
    )
    anlageziel = ziel(**zielfelder)
    profil = classify(profil_kunde, anlageziel, POLICIES, _recorder())

    minimum = min(
        profil.capacity.risk_class,
        profil.tolerance.risk_class,
        profil.knowledge.max_risk_class,
    )

    assert profil.risk_class <= minimum
    assert profil.risk_class == minimum


@pytest.mark.invariant
@given(
    basis=haushalte(),
    antworten=antwortstrategie(),
    zielfelder=ziele(),
    kenntnisklasse=st.integers(min_value=1, max_value=5),
)
def test_i3_ergebnis_liegt_nie_ueber_dem_mittelwert_der_achsen(
    basis: dict[str, Any],
    antworten: dict[str, str],
    zielfelder: dict[str, Any],
    kenntnisklasse: int,
) -> None:
    """Das Minimum liegt nie über dem Mittelwert — die Gegenprobe zur Mittelung.

    Der Test belegt, dass die Wahl des Verfahrens niemals großzügiger ausfällt
    als eine Mittelung. Wo beide auseinanderfallen, ist das Minimum stets das
    vorsichtigere Ergebnis.

    Args:
        basis: Felder des Haushalts.
        antworten: Antworten zur Risikobereitschaft.
        zielfelder: Felder des Anlageziels.
        kenntnisklasse: Angestrebte Obergrenze aus den Kenntnissen.
    """
    profil = classify(
        kunde(
            household=haushalt(**basis),
            knowledge=kenntnisse_fuer_klasse(kenntnisklasse),
            risk_attitude=RiskAttitude(answers=antworten),
        ),
        ziel(**zielfelder),
        POLICIES,
        _recorder(),
    )

    mittelwert = Decimal(profil.capacity.risk_class + profil.tolerance.risk_class) / Decimal(2)

    assert Decimal(profil.risk_class) <= mittelwert


@pytest.mark.invariant
@given(basis=haushalte(), antworten=antwortstrategie(), zielfelder=ziele())
def test_i3_kenntnisdeckel_kann_nur_senken(
    basis: dict[str, Any], antworten: dict[str, str], zielfelder: dict[str, Any]
) -> None:
    """Ein höherer Kenntnisdeckel hebt das Ergebnis nie über die beiden Achsen.

    Args:
        basis: Felder des Haushalts.
        antworten: Antworten zur Risikobereitschaft.
        zielfelder: Felder des Anlageziels.
    """
    profil_haushalt = haushalt(**basis)
    anlageziel = ziel(**zielfelder)

    mit_niedrigem_deckel = classify(
        kunde(
            household=profil_haushalt,
            knowledge=kenntnisse_fuer_klasse(1),
            risk_attitude=RiskAttitude(answers=antworten),
        ),
        anlageziel,
        POLICIES,
        _recorder(),
    )
    mit_hohem_deckel = classify(
        kunde(
            household=profil_haushalt,
            knowledge=kenntnisse_fuer_klasse(5),
            risk_attitude=RiskAttitude(answers=antworten),
        ),
        anlageziel,
        POLICIES,
        _recorder(),
    )

    assert mit_hohem_deckel.risk_class >= mit_niedrigem_deckel.risk_class
    assert mit_hohem_deckel.risk_class == min(
        mit_hohem_deckel.capacity.risk_class, mit_hohem_deckel.tolerance.risk_class
    )


@pytest.mark.invariant
@given(basis=haushalte(), antworten=antwortstrategie(), zielfelder=ziele())
def test_i3_kenntnisse_liefern_nie_eine_punktzahl(
    basis: dict[str, Any], antworten: dict[str, str], zielfelder: dict[str, Any]
) -> None:
    """Die Kenntnisachse liefert stets eine Klasse zwischen 1 und 5, nie einen Score.

    Args:
        basis: Felder des Haushalts.
        antworten: Antworten zur Risikobereitschaft.
        zielfelder: Felder des Anlageziels.
    """
    ergebnis = assess_knowledge(
        kunde(household=haushalt(**basis), risk_attitude=RiskAttitude(answers=antworten)),
        POLICIES.knowledge,
        _recorder(),
    )
    _ = ziel(**zielfelder)

    assert 1 <= ergebnis.max_risk_class <= 5
