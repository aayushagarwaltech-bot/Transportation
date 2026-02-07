import os
import yaml
from pathlib import Path
import requests
import zipfile
import io


def main():
    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / "params.yaml") as f:
        params = yaml.safe_load(f)

    url = params["dataset"]["url"]
    out_path = repo_root / params["dataset"]["raw_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Raw data already exists at {out_path}")
        return

    print(f"Downloading dataset from {url} to {out_path}")
    r = requests.get(url)
    if r.status_code == 404:
        # Try common zip filename fallback in same directory
        base = url.rsplit('/', 1)[0]
        zip_url = f"{base}/Bike-Sharing-Dataset.zip"
        print(f"Primary URL 404; trying {zip_url}")
        r = requests.get(zip_url)
        r.raise_for_status()

    r.raise_for_status()

    content_type = r.headers.get('Content-Type', '')
    if 'zip' in content_type or url.lower().endswith('.zip'):
        z = zipfile.ZipFile(io.BytesIO(r.content))
        # find day.csv inside zip
        members = [n for n in z.namelist() if n.lower().endswith('day.csv')]
        if not members:
            raise RuntimeError('day.csv not found inside downloaded zip')
        data = z.read(members[0])
        out_path.write_bytes(data)
        print(f"Extracted {members[0]} to {out_path}")
    else:
        out_path.write_bytes(r.content)
        print("Download complete")


if __name__ == '__main__':
    main()
