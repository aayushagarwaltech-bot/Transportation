import yaml
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import mlflow


def main():
    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / "params.yaml") as f:
        params = yaml.safe_load(f)

    mlflow_cfg = params.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "experiment"))

    test_path = repo_root / "data/features/test.csv"
    df = pd.read_csv(test_path)
    target = "cnt"
    X = df.drop(columns=[target])
    y = df[target]

    model_path = repo_root / "models/model.pkl"
    model = joblib.load(model_path)

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = float(mse ** 0.5)
    r2 = r2_score(y, preds)

    out_dir = repo_root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"rmse": float(rmse), "r2": float(r2)}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with mlflow.start_run():
        mlflow.log_metrics(metrics)

    print(f"Evaluation metrics: {metrics}")


if __name__ == '__main__':
    main()
