import yaml
from pathlib import Path
import pandas as pd


def main():
    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / "params.yaml") as f:
        params = yaml.safe_load(f)

    raw_path = repo_root / params["dataset"]["raw_path"]
    out_dir = repo_root / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preprocessed.csv"

    print(f"Reading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    # Basic cleaning: drop any fully empty rows and reset index
    df = df.dropna(how="all").reset_index(drop=True)

    # Example: convert categorical to numeric where useful (weekday, season)
    # Pandas read already provides numbers for these columns in this dataset

    df.to_csv(out_path, index=False)
    print(f"Wrote preprocessed data to {out_path}")


if __name__ == '__main__':
    main()
