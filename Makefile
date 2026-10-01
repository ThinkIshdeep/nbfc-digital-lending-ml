.PHONY: install dev test lint format check train verify quality predict api app docker

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest -q

lint:
	ruff check src scripts tests

format:
	ruff format src scripts tests

check: lint test verify

quality: lint test verify

train:
	python scripts/train.py

verify:
	python scripts/verify_repo.py

predict:
	python scripts/predict.py --disbursement 100 --loan-count 10000 --avg-ticket 10000 --period-index 3

api:
	uvicorn nbfc_ml.api:app --host 0.0.0.0 --port 8000

app: api

docker:
	docker build -t nbfc-digital-lending-ml .
