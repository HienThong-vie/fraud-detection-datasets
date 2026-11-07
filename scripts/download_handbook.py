from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import certifi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training.json"
DESTINATION = PROJECT_ROOT / "data" / "raw"


def dates_between(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "fraud-handbook-ml/0.1"})
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60, context=ssl_context) as response:
                partial.write_bytes(response.read())
            partial.replace(destination)
            return
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download official Handbook daily pickle files")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--limit-days", type=int)
    args = parser.parse_args(argv)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    start = date.fromisoformat(args.start_date or config["download_start_date"])
    end = date.fromisoformat(args.end_date or config["download_end_date"])
    commit = config["source_commit"]
    base_url = (
        "https://raw.githubusercontent.com/Fraud-Detection-Handbook/"
        f"simulated-data-raw/{commit}/data"
    )
    requested_dates = list(dates_between(start, end))
    if args.limit_days is not None:
        requested_dates = requested_dates[: args.limit_days]

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for index, day in enumerate(requested_dates, start=1):
        destination = DESTINATION / f"{day.isoformat()}.pkl"
        if destination.exists():
            print(f"[{index}/{len(requested_dates)}] exists: {destination.name}")
            continue
        print(f"[{index}/{len(requested_dates)}] downloading {destination.name}")
        download(f"{base_url}/{destination.name}", destination)
    print(f"Dataset directory: {DESTINATION}")


if __name__ == "__main__":
    main()
