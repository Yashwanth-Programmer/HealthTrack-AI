.PHONY: install run test lint cov benchmark docker

install:
	pip install -r requirements-dev.txt ruff

run:
	python -m healthtrack

test:
	pytest

cov:
	pytest --cov=healthtrack --cov-report=term-missing

lint:
	ruff check .

benchmark:
	python -m healthtrack.benchmark --sweep

docker:
	docker compose up --build
