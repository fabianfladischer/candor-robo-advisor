"""Textbausteine der erzeugten Dokumente, gebunden an Regel-Kennungen.

Das Prinzip
-----------
Kein Satz in einer Geeignetheitserklärung entsteht frei. Jeder Baustein hängt an
der Kennung genau der Regel, die den zugehörigen Entscheidungsschritt im Trace
aufgezeichnet hat. Wer den Text liest, kann ihn über die Kennung im Trace
nachschlagen; wer den Trace liest, findet zu jedem Schritt den Satz, der ihn im
Dokument beschreibt.

Das ist die Umsetzung von Prinzip P2 auf der Textebene: **Keine Ausgabe ohne
benannte Regel.** Eine frei formulierte Erklärung wäre nicht nachvollziehbar und
im Streitfall nicht belegbar — und sie könnte etwas behaupten, was das System
nicht gerechnet hat.

Aufbau eines Bausteins
----------------------
Jeder Baustein ist eine Vorlage mit benannten Feldern. Die Felder werden
ausschließlich mit Werten aus dem Trace oder den Ergebnisobjekten gefüllt, nie
mit freiem Text. Fehlt ein Feld, bricht die Formatierung ab — das ist gewollt:
Eine Erklärung mit einer Lücke ist schlechter als keine (Prinzip P5).

Sprache
-------
Die Bausteine sind bewusst nüchtern und kurz gehalten. Sie beschreiben, was
gerechnet wurde, und bewerten es nicht. Die Einordnung ist Aufgabe des Beraters
im Gespräch, nicht des erzeugten Dokuments.
"""

from __future__ import annotations

from typing import Final

#: Vorlagen je Regel-Kennung. Die Kennungen entsprechen denen der ``Rule``-
#: Objekte in ``candor/profiling/``, ``candor/goals/``, ``candor/engine/``.
TEMPLATES: Final[dict[str, str]] = {
    # --- Risikotragfähigkeit ----------------------------------------------
    "CAP-CLASS": (
        "Die Risikotragfähigkeit wurde anhand von sechs Kriterien erhoben und mit "
        "{punkte} von 100 Punkten bewertet. Nach den Klassenschwellen des Regelwerks "
        "entspricht diese Punktzahl Risikoklasse {klasse_aus_punkten}; maßgeblich für die "
        "Tragfähigkeit ist Risikoklasse {klasse}. Eine Abweichung zwischen beiden Werten "
        "beruht auf einem Ausschlusskriterium, das die Klasse unabhängig von der Punktzahl "
        "setzt. "
        "Die Tragfähigkeit beschreibt, welches Risiko die wirtschaftlichen Verhältnisse "
        "objektiv zulassen, unabhängig davon, welches Risiko gewünscht wird."
    ),
    "CAP-KO-LIQ": (
        "Die Liquiditätsreserve beträgt {reserve} Monatsausgaben und liegt damit unter der "
        "geforderten Mindestreserve von {mindest} Monatsausgaben. Dieses Kriterium wirkt als "
        "Ausschlusskriterium: Die Risikoklasse wird unabhängig von allen übrigen Kriterien "
        "auf 1 gesetzt. Vor jeder Anlageentscheidung ist zunächst die Rücklage aufzubauen."
    ),
    # --- Risikobereitschaft -----------------------------------------------
    "TOL-CLASS": (
        "Die Risikobereitschaft wurde über fünf Fragen erhoben und mit {punkte} von 100 "
        "Punkten bewertet. Daraus ergibt sich Risikoklasse {klasse}. Die Bereitschaft "
        "beschreibt, welches Risiko subjektiv getragen werden will."
    ),
    "TOL-SCEN": (
        "Zur Erhebung der Verlusttoleranz wurden konkrete Eurobeträge vorgelegt und keine "
        "Prozentangaben. Bezugsgröße ist das über die Laufzeit eingezahlte Kapital von "
        "{bezugsbetrag}; als tragbar bezeichnet wurde ein zwischenzeitlicher Rückgang von "
        "{betrag}. Prozentangaben werden erfahrungsgemäß systematisch anders bewertet als "
        "der gleich hohe Betrag; die Betragsdarstellung begrenzt diesen Effekt."
    ),
    "TOL-CONFLICT": (
        "Bei der Erhebung der Risikobereitschaft wurden widersprüchliche Angaben "
        "festgestellt: {befund} Der Widerspruch wurde nicht rechnerisch geglättet, sondern "
        "im Entscheidungspfad vermerkt. Er ist im Beratungsgespräch aufzuklären."
    ),
    # --- Kenntnisse und Erfahrungen ---------------------------------------
    "KNW-CAP": (
        "Kenntnisse und Erfahrungen wurden nach Instrumentengattungen erhoben. Sie ergeben "
        "keinen eigenen Punktwert, sondern eine Obergrenze der zulässigen Produktkomplexität: "
        "Risikoklasse {deckel}. Rechtsgrundlage ist Art. 55 DelVO (EU) 2017/565."
    ),
    # --- Zusammenführung --------------------------------------------------
    "CLS-MIN": (
        "Aus dem Profil ergibt sich die Risikoklasse als Minimum aus Risikotragfähigkeit "
        "({tragfaehigkeit}) und Risikobereitschaft ({bereitschaft}), zusätzlich begrenzt durch "
        "die Kenntnisse und Erfahrungen ({deckel}): Risikoklasse {klasse}, bindend ist die "
        "Achse {achse}. Maßgeblich für die Anlageentscheidung ist Risikoklasse {endklasse}; "
        "eine Abweichung von der Profilklasse beruht auf der zusätzlichen Deckelung durch "
        "die verbleibende Restlaufzeit (Glide-Path). "
        "Die Achsen werden ausdrücklich nicht gemittelt: Eine "
        "Mittelung würde eine fehlende Voraussetzung durch eine andere ausgleichen, die sie "
        "nicht ersetzen kann."
    ),
    "CLS-DIV": (
        "Tragfähigkeit und Bereitschaft liegen {abstand} Stufen auseinander. Der Abstand wird "
        "ausgewiesen, weil er auf einen Klärungsbedarf hinweist: Wer mehr Risiko tragen könnte, "
        "als er tragen will, trifft eine andere Entscheidung als jemand, der mehr Risiko will, "
        "als er tragen kann."
    ),
    # --- Glide-Path --------------------------------------------------------
    "GLD-CAP": (
        "Wegen der verbleibenden Laufzeit von {restlaufzeit} Monaten bis zum Zieltermin wird "
        "die zulässige Risikoklasse zusätzlich auf {deckel} begrenzt. Mit abnehmender "
        "Restlaufzeit sinkt die Fähigkeit, einen zwischenzeitlichen Rückgang auszusitzen."
    ),
    # --- Anlageziel --------------------------------------------------------
    #
    # Hinweis: Für die Regel GOAL-TPL (Vorbelegung aus dem Zielkatalog) gibt es
    # bewusst KEINEN Baustein. Sie wird beim Anlegen des Ziels aufgezeichnet,
    # also vor Beginn der Entscheidungskette und damit ausserhalb des Trace, den
    # das Dokument belegt. Ein Baustein dafuer wuerde eine Herkunft behaupten,
    # die das Dokument nicht nachweisen kann. Der Zieltyp steht stattdessen als
    # erhobene Angabe in der Zieltabelle.
    "GOAL-RET": (
        "Um den Zielbetrag von {zielbetrag} in {monate} Monaten zu erreichen, ist bei einem "
        "Startkapital von {startkapital} und einer monatlichen Sparrate von {sparrate} eine "
        "Rendite von {nominal} jährlich erforderlich. Bei einer angenommenen Inflation von "
        "{inflation} entspricht das real {real} jährlich."
    ),
    # --- Zielabgleich und Simulation ---------------------------------------
    "REC-SIM": (
        "Die Zielerreichung wurde über {pfade} simulierte Verläufe geprüft. Verwendet wurde ein "
        "zirkulärer Block-Bootstrap mit einer Blocklänge von zwölf Monaten: Er zieht "
        "zusammenhängende historische Zeiträume statt einzelner Monate und erhält damit "
        "Autokorrelation, Volatilitätscluster und die Abhängigkeit zwischen den Assetklassen. "
        "Kosten und jährliches Rebalancing sind berücksichtigt."
    ),
    "REC-THR": (
        "Für dieses Ziel wird eine Mindest-Erfolgswahrscheinlichkeit von {gefordert} verlangt. "
        "Der Wert ergibt sich aus Zieltyp und Priorität und ist im Regelwerk hinterlegt."
    ),
    "REC-OUT": (
        "Die simulierte Erfolgswahrscheinlichkeit beträgt {erreicht} gegenüber geforderten "
        "{gefordert}. Ergebnis des Abgleichs: {ausgang}."
    ),
    "REC-CONFLICT": (
        "Die geforderte Erfolgswahrscheinlichkeit wird unter der zulässigen Risikoklasse nicht "
        "erreicht. Es ergeht **keine Anlageempfehlung**. Rechtsgrundlage: Art. 54 Abs. 10 "
        "DelVO (EU) 2017/565 — ein Geschäft, das für den Kunden nicht geeignet ist, darf nicht "
        "empfohlen werden."
    ),
    "REC-LEVER": (
        "Stellschraube {kennung} — {bezeichnung}: gegenwärtig {ist}, erforderlich {soll}. "
        "{bewertung}"
    ),
    # --- Portfolio ---------------------------------------------------------
    "ADV-PF": (
        "Das empfohlene Portfolio wurde durch eine restringierte Optimierung nach dem "
        "Erwartungswert-Varianz-Ansatz bestimmt. Die Bandbreiten der Risikoklasse {klasse} sind "
        "dabei zwingende Nebenbedingungen; Leerverkäufe sind ausgeschlossen. Die erwarteten "
        "Renditen und die Kovarianz wurden geschrumpft geschätzt, um die bekannte "
        "Empfindlichkeit des Verfahrens gegenüber Schätzfehlern zu begrenzen."
    ),
    "ADV-ESG": (
        "Nachhaltigkeitspräferenzen wurden nach Art. 2 Nr. 7 DelVO (EU) 2017/565 in der Fassung "
        "der DelVO (EU) 2021/1253 erhoben. Ergebnis: {befund}"
    ),
    "ADV-STOP": (
        "Da keine Empfehlung ergeht, wurde kein Portfolio berechnet. Die Entscheidung darüber, "
        "**ob** empfohlen werden darf, geht der Entscheidung darüber, **was** empfohlen wird, "
        "voraus."
    ),
}

#: Bausteine ohne eigene Regel — sie beschreiben das Dokument selbst und nicht
#: einen Rechenschritt. Sie tragen deshalb keine Regel-Kennung.
STATIC: Final[dict[str, str]] = {
    "zweck_geeignetheit": (
        "Diese Erklärung dokumentiert, wie die ausgesprochene Anlageempfehlung zustande "
        "gekommen ist und weshalb sie zu den erhobenen Verhältnissen, Zielen und Kenntnissen "
        "passt. Rechtsgrundlage ist Art. 54 Abs. 12 DelVO (EU) 2017/565 in Verbindung mit "
        "§ 64 Abs. 4 WpHG."
    ),
    "zweck_konflikt": (
        "Dieses Dokument tritt an die Stelle einer Geeignetheitserklärung. Es dokumentiert, "
        "weshalb für das erhobene Anlageziel unter der zulässigen Risikoklasse **keine** "
        "Empfehlung ausgesprochen werden kann, und beziffert, was sich ändern müsste."
    ),
    "risiko_allgemein": (
        "Wertpapieranlagen unterliegen Kursschwankungen. Der Wert der Anlage kann fallen; "
        "eingesetztes Kapital kann teilweise oder vollständig verloren gehen. Aus der "
        "historischen Wertentwicklung lässt sich die künftige nicht ableiten."
    ),
    "risiko_simulation": (
        "Die ausgewiesene Erfolgswahrscheinlichkeit ist das Ergebnis einer Simulation auf "
        "Grundlage historischer Monatsrenditen. Das Verfahren kann nur Marktverläufe erzeugen, "
        "die in ähnlicher Form bereits vorgekommen sind; ein Verlauf, der schlechter ist als "
        "alles historisch Beobachtete, ist darin nicht enthalten. Die Angabe ist eine "
        "Modellrechnung und keine Zusage."
    ),
    "risiko_annahmen": (
        "Die Kostenannahmen sind Größenordnungen und nicht die tatsächlichen Kostensätze der "
        "genannten Instrumente. Steuern sind nicht berücksichtigt. Beides verändert das "
        "Ergebnis und ist vor einer Anlageentscheidung zu konkretisieren."
    ),
    "reproduktion": (
        "Alle Angaben dieses Dokuments sind reproduzierbar: Gleiches Profil, gleicher Seed und "
        "gleicher Regelstand führen zu einem zeichengleichen Dokument. Der Inhaltshash des "
        "Entscheidungspfads ist unten ausgewiesen."
    ),
}


def render(rule_id: str, **felder: object) -> str:
    """Füllt den Baustein einer Regel.

    Args:
        rule_id: Kennung der Regel.
        **felder: Werte der benannten Felder.

    Returns:
        Der gefüllte Baustein.

    Raises:
        KeyError: Wenn die Regel keinen Baustein hat oder ein Feld fehlt. Beides
            ist ein Fehler und wird nicht durch einen Platzhalter überdeckt
            (Prinzip P5).
    """
    if rule_id not in TEMPLATES:
        msg = (
            f"Für die Regel {rule_id!r} ist kein Textbaustein hinterlegt. Jeder im Dokument "
            "beschriebene Schritt muss an eine benannte Regel gebunden sein (Prinzip P2)."
        )
        raise KeyError(msg)
    try:
        return TEMPLATES[rule_id].format(**felder)
    except KeyError as fehler:
        msg = (
            f"Im Baustein der Regel {rule_id!r} fehlt der Wert für {fehler.args[0]!r}. "
            "Eine Erklärung mit einer Lücke wird nicht erzeugt (Prinzip P5)."
        )
        raise KeyError(msg) from fehler


def static(key: str) -> str:
    """Liefert einen Baustein ohne Regelbezug.

    Args:
        key: Kennung des Bausteins.

    Returns:
        Der Text.

    Raises:
        KeyError: Wenn der Baustein nicht hinterlegt ist.
    """
    if key not in STATIC:
        msg = f"Kein statischer Textbaustein unter {key!r}."
        raise KeyError(msg)
    return STATIC[key]


def known_rule_ids() -> frozenset[str]:
    """Liefert alle Regel-Kennungen mit hinterlegtem Baustein.

    Returns:
        Die Kennungen.
    """
    return frozenset(TEMPLATES)
