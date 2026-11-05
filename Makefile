.PHONY: setup lint test train serve
setup:
	python -m pip install -e ".[dev,serve]"
lint:
	python -m ruff check .
	python -m ruff format --check .
test:
	python -m pytest
train:
	python scripts/train.py --restart
serve:
	python scripts/serve.py
