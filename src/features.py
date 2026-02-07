import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / "params.yaml") as f:
        params = yaml.safe_load(f)

    in_path = repo_root / "data/processed/preprocessed.csv"
    df = pd.read_csv(in_path)

    drop_cols = params.get("features", {}).get("drop_columns", [])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    target_col = "cnt"
    if target_col not in df.columns:
        raise RuntimeError(f"Target column '{target_col}' not found in data")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    test_size = params.get("preprocessing", {}).get("test_size", 0.2)
    random_state = params.get("preprocessing", {}).get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    out_dir = repo_root / "data/features"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = X_train.copy()
    train_df[target_col] = y_train
    test_df = X_test.copy()
    test_df[target_col] = y_test

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print(f"Saved features to {out_dir}")


if __name__ == '__main__':
    main()
