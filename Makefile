.PHONY: setup lint test train-smoke serve
setup:
	python -m pip install -e ".[dev,serve]"
lint:
	python -m ruff check .
	python -m ruff format --check .
test:
	python -m pytest
train-smoke:
	python scripts/prepare_data.py --input examples/demo.txt --output data/demo
	python scripts/train.py --config configs/training/demo.yaml --model configs/model/demo-moe.yaml --max-steps 20
serve:
	python -m uvicorn moelab.serving.api:app --host 127.0.0.1 --port 8000
