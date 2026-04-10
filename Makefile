PYTHON ?= python3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: help venv install setup env run-all score-emotions paper check clean

help:
	@echo "Targets:"
	@echo "  make setup      - create venv, install package, create .env"
	@echo "  make run-all    - run preregistered stage scaffold"
	@echo "  make score-emotions - run threaded OpenRouter emotion semantic scoring"
	@echo "  make paper      - compile paper/report/main.tex with tectonic"
	@echo "  make check      - compile Python sources to verify syntax"
	@echo "  make clean      - remove caches and build artifacts"

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e .

env:
	@if [ ! -f .env ]; then cp .env.example .env; fi

setup: install env

run-all:
	$(PY) -m cbias run-all

score-emotions:
	$(PY) -m cbias score-emotions --max-workers 50

paper:
	cd paper/report && tectonic --reruns 4 main.tex

check:
	$(PY) -m compileall src

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache
