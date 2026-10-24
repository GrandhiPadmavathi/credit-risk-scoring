"""Download the UCI 'default of credit card clients' dataset (CC BY 4.0) into data/raw/."""

import io
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from creditrisk.config import SETTINGS  # noqa: E402
from creditrisk.data import URL  # noqa: E402


def main() -> None:
    target = SETTINGS.paths.raw_xls
    if target.exists():
        print(f"Already present: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".xls"))
        target.write_bytes(zf.read(name))
    print(f"Saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
