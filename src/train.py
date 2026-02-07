import yaml
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import mlflow


def main():
    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / "params.yaml") as f:
        params = yaml.safe_load(f)

    mlflow_cfg = params.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "experiment"))

    train_path = repo_root / "data/features/train.csv"
    df = pd.read_csv(train_path)
    target = "cnt"
    X = df.drop(columns=[target])
    y = df[target]

    model_params = params.get("model", {})
    model = RandomForestRegressor(
        n_estimators=model_params.get("n_estimators", 100),
        random_state=model_params.get("random_state", 42),
    )

    with mlflow.start_run():
        mlflow.log_params(model_params)
        model.fit(X, y)

        models_dir = repo_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))
        print(f"Saved model to {model_path}")


if __name__ == '__main__':
    main()
