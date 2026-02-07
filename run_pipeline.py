import subprocess
import sys
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[0]
    # Run DVC repro to execute the pipeline in order
    try:
        subprocess.check_call([sys.executable, "-m", "dvc", "repro"], cwd=repo)
    except Exception as e:
        print("Failed to run 'dvc repro'. Ensure DVC is installed and initialized in the repo.")
        raise


if __name__ == '__main__':
    main()
