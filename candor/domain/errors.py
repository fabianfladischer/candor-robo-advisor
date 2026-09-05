"""Fehlertypen des Domänenkerns.

Alle Fehler tragen neben der Meldung das betroffene Feld und — soweit
einschlägig — die **Rechtsgrundlage**, aus der sich die Pflichtangabe ergibt.
Das ist kein Beiwerk: Eine Anlageempfehlung auf Basis fehlender Pflichtangaben
ist ein Regulierungsverstoß, kein Komfortproblem (Prinzip P5). Wer den Fehler
liest, soll ohne Blick in den Code erkennen, *warum* die Angabe verlangt wird.
"""

from __future__ import annotations


class DomainError(Exception):
    """Oberklasse aller fachlichen Fehler des Domänenkerns."""


class IncompleteProfileError(DomainError):
    """Eine Pflichtangabe fehlt oder ist unzulässig.

    Attributes:
        field: Pfad des betroffenen Feldes, etwa ``household.net_monthly_income``.
        reason: Beschreibung des Mangels im Klartext.
        legal_basis: Rechtsgrundlage der Pflichtangabe, sofern einschlägig.
    """

    def __init__(self, field: str, reason: str, legal_basis: str | None = None) -> None:
        """Erzeugt den Fehler.

        Args:
            field: Pfad des betroffenen Feldes.
            reason: Beschreibung des Mangels.
            legal_basis: Rechtsgrundlage, sofern einschlägig.
        """
        self.field = field
        self.reason = reason
        self.legal_basis = legal_basis
        text = f"Pflichtangabe '{field}' fehlt oder ist unzulässig: {reason}"
        if legal_basis:
            text = f"{text} — Rechtsgrundlage: {legal_basis}"
        super().__init__(text)


class InconsistentProfileError(DomainError):
    """Die Angaben sind für sich zulässig, widersprechen sich aber gegenseitig.

    Beispiel: eine freie Sparrate, die höher liegt als Einkommen abzüglich
    Fixkosten und Schuldendienst. Eine solche Eingabe ist nicht unvollständig,
    aber unbrauchbar, und wird deshalb ebenso hart abgewiesen.

    Attributes:
        fields: Beteiligte Felder.
        reason: Beschreibung des Widerspruchs.
    """

    def __init__(self, fields: tuple[str, ...], reason: str) -> None:
        """Erzeugt den Fehler.

        Args:
            fields: Beteiligte Felder.
            reason: Beschreibung des Widerspruchs.
        """
        self.fields = fields
        self.reason = reason
        super().__init__(f"Widersprüchliche Angaben in {', '.join(fields)}: {reason}")


class PolicyError(DomainError):
    """Das Regelwerk ist fehlerhaft, unvollständig oder unbekannt versioniert."""
