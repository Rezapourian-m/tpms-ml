.PHONY: setup fmt lint clean

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

clean:
	rm -rf __pycache__ .pytest_cache .venv models/*

# Examples
train-forward-a:
	python -m src.cli.train_forward --data data/dataset_clean.csv --task A --model-out models/forward_A.keras --scaler-out models/forward_A_scaler.joblib
