.PHONY: data features train segment explain all lint format test api dashboard clean

PYTHON := .venv/Scripts/python
PYTEST := .venv/Scripts/pytest
RUFF := .venv/Scripts/ruff
BLACK := .venv/Scripts/black
UVICORN := .venv/Scripts/uvicorn
STREAMLIT := .venv/Scripts/streamlit

data:
    $(PYTHON) -m src.pipeline --stage data

features:
    $(PYTHON) -m src.pipeline --stage features

train:
    $(PYTHON) -m src.pipeline --stage train

segment:
    $(PYTHON) -m src.pipeline --stage segment

explain:
    $(PYTHON) -m src.pipeline --stage explain

all:
    $(PYTHON) -m src.pipeline --stage all

lint:
    $(RUFF) check src api frontend tests
    $(BLACK) --check src api frontend tests

format:
    $(BLACK) src api frontend tests
    $(RUFF) check --fix src api frontend tests

test:
    $(PYTEST) --cov=src --cov=api

api:
    $(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
    $(STREAMLIT) run frontend/dashboard.py --server.port 8501

clean:
    rm -rf __pycache__ .pytest_cache .ruff_cache logs/*.log
