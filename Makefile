# CANDOR — Automatisierung
#
# `make verify` ist die Qualitätsschranke: Linter, strikte Typprüfung und die
# vollständige Testsuite. Sie muss vor jeder Änderung grün sein.

VENV      := .venv
PY        := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
RUFF      := $(VENV)/bin/ruff
MYPY      := $(VENV)/bin/mypy
PYTEST    := $(VENV)/bin/pytest
STREAMLIT := $(VENV)/bin/streamlit

# Basis-Seed. Gleicher Seed, gleiches Ergebnis.
SEED := 20260726

.DEFAULT_GOAL := help

.PHONY: help setup check-venv lint format typecheck test coverage verify \
        verify-lock verify-snapshot app clean

help: ## Diese Übersicht anzeigen
	@echo "CANDOR — verfügbare Ziele:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""

# --- Umgebung --------------------------------------------------------------

setup: ## Virtuelle Umgebung anlegen und Projekt installieren
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "Fertig. Aktivieren mit:  source $(VENV)/bin/activate"

check-venv:
	@test -x $(PY) || { echo "Keine virtuelle Umgebung. Zuerst 'make setup'."; exit 1; }

# --- Qualitätsschranke -----------------------------------------------------

lint: check-venv ## Regelverstöße und Formatierung prüfen
	$(RUFF) check .
	$(RUFF) format --check .

format: check-venv ## Formatierung anwenden
	$(RUFF) format .
	$(RUFF) check --fix .

typecheck: check-venv ## Strikte Typprüfung
	$(MYPY) candor/

test: check-venv ## Vollständige Testsuite
	$(PYTEST) -q

coverage: check-venv ## Testsuite mit Abdeckungsbericht
	$(PYTEST) -q --cov=candor --cov-report=term-missing

verify: lint typecheck test ## Linter, Typprüfung und Tests
	@echo ""
	@echo "Qualitätsschranke bestanden."

verify-lock: ## Kaltstart: frische Umgebung allein aus requirements.lock
	@echo "Baue eine Wegwerf-Umgebung ausschließlich aus requirements.lock ..."
	@rm -rf .lockprobe
	@python3 -m venv .lockprobe
	@.lockprobe/bin/pip install --quiet --upgrade pip setuptools wheel
	@.lockprobe/bin/pip install --quiet -r requirements.lock
	@.lockprobe/bin/pip install --quiet --no-deps -e .
	@.lockprobe/bin/python -m pytest --collect-only -q > /dev/null \
	  || { echo "FEHLGESCHLAGEN — requirements.lock ist unvollständig."; \
	       .lockprobe/bin/python -m pytest --collect-only -q 2>&1 | tail -20; \
	       rm -rf .lockprobe; exit 1; }
	@rm -rf .lockprobe
	@echo "Kaltstart-Probe bestanden."

verify-snapshot: check-venv ## Datenstand gegen das SHA-256-Manifest prüfen
	$(PY) -m candor.data.manifest --verify

# --- Anwendung -------------------------------------------------------------

app: check-venv ## Demonstrationsoberfläche starten
	$(STREAMLIT) run app/streamlit_app.py

# --- Aufräumen -------------------------------------------------------------

clean: ## Caches und Build-Artefakte entfernen
	find . -path ./$(VENV) -prune -o -name '__pycache__' -type d -print0 | xargs -0 rm -rf
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis .coverage htmlcov build dist *.egg-info
	@echo "Aufgeräumt."
