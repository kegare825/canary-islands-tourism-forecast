.PHONY: help install test notebooks app demo-screens reports metrics docker-build docker-run clean

VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Targets: install test notebooks app demo-screens reports metrics docker-build docker-run"

install:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

notebooks:
	$(PYTHON) scripts/execute_and_report.py

app:
	$(VENV)/bin/streamlit run app/streamlit_app.py

demo-screens:
	$(PYTHON) scripts/generate_demo_screenshots.py

reports: demo-screens
	$(PYTHON) scripts/ine_contrast.py
	$(PYTHON) scripts/feature_importance.py

metrics:
	$(PYTHON) scripts/compute_backtest_summary.py

docker-build:
	docker build -t canary-islands-tourism-forecast -f docker/Dockerfile .

docker-run:
	docker run --rm -p 8501:8501 -v $(PWD)/models:/app/models:ro -v $(PWD)/data/processed:/app/data/processed:ro canary-islands-tourism-forecast

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
