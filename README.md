# Traffic Automated ML Pipeline

This project demonstrates an automated ML workflow for transportation-related traffic prediction (using the UCI Bike Sharing "day.csv" dataset as a traffic proxy).

Quick start

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

2. Initialize DVC (first time only) and run the pipeline with one command:

```bash
# Initialize dvc (only first time)
dvc init

# Run the full pipeline
python run_pipeline.py
```

What the pipeline does
- `src/data_load.py` — downloads the public dataset to `data/raw/`.
- `src/preprocess.py` — basic cleaning and normalization to `data/processed/`.
- `src/features.py` — feature preparation and train/test split to `data/features/`.
- `src/train.py` — trains a RandomForest and saves the model to `models/` and logs params to MLflow.
- `src/evaluate.py` — evaluates the model on the test set and logs metrics to MLflow and `metrics/metrics.json`.

Configuration
- All configurable values are in `params.yaml` (dataset URL, model hyperparameters, MLflow settings).

Notes
- Do not commit `data/`, `models/`, or `mlruns/` to Git — they are excluded via `.gitignore`.
- To reproduce only certain stages, use `dvc repro <stage>`.
