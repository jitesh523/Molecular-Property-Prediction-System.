# Molecular Property Prediction — Developer Makefile
# Usage: `make <target>`

PYTHON ?= python
PIP    ?= pip
APP    ?= molprop.serving.api:app
HOST   ?= 0.0.0.0
PORT   ?= 8000

.PHONY: help install dev lint format test cov api docker-build docker-run clean

help:
	@echo "Targets:"
	@echo "  install      Install the package (editable mode)"
	@echo "  dev          Install dev dependencies (ruff, pytest, bandit)"
	@echo "  lint         Run ruff check + format check"
	@echo "  format       Auto-format code with ruff"
	@echo "  test         Run test suite"
	@echo "  cov          Run tests with coverage report"
	@echo "  api          Start FastAPI dev server with auto-reload"
	@echo "  docker-build Build the inference Docker image"
	@echo "  docker-run   Run the inference container on :$(PORT)"
	@echo "  clean        Remove caches and build artifacts"

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e .
	$(PIP) install pytest pytest-cov ruff bandit httpx

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest tests/ -q

cov:
	pytest tests/ --cov=molprop --cov-report=term-missing --cov-report=xml -q

api:
	uvicorn $(APP) --host $(HOST) --port $(PORT) --reload

docker-build:
	docker build -t molprop-api:latest .

docker-run:
	docker run --rm -p $(PORT):8000 molprop-api:latest

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
