.PHONY: install lint format typecheck test eval serve run-demo docker-build up down

install:
	uv venv && uv pip install -e ".[dev,eval,openai,anthropic,postgres]"

lint:
	ruff check src tests && ruff format --check src tests

format:
	ruff check --fix src tests && ruff format src tests

typecheck:
	mypy

test:
	pytest --cov=loopguard --cov-report=term-missing

eval:
	loopguard eval --out reports

serve:
	loopguard serve --reload

run-demo:
	loopguard run "What's the weather in Chennai right now?"

docker-build:
	docker build -t loopguard-agent:local .

up:
	docker compose up --build -d

down:
	docker compose down
