.PHONY: help install run test docker-up docker-down pull-model ingest

help:
	@echo "Targets:"
	@echo "  install      Install dependencies"
	@echo "  run          Run FastAPI locally (port 8001)"
	@echo "  test         Run pytest"
	@echo "  docker-up    Start API + Ollama via docker-compose"
	@echo "  docker-down  Stop docker-compose"
	@echo "  pull-model   Pull default Ollama model inside container"
	@echo "  ingest       Call /ingest on the running API"

install:
	python -m pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

test:
	pytest -q

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

pull-model:
	docker-compose exec ollama ollama pull llama3.2:3b

ingest:
	curl -X POST "http://localhost:8001/ingest"
