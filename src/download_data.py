"""
download_data.py
----------------
Programmatically download all datasets used in this project.

Sources:
  - Voteview (congress members + votes + rollcalls)
  - TidyTuesday mirror of UN General Assembly votes (from dgrtwo/unvotes)
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── dataset registry ─────────────────────────────────────────────────────────
DATASETS = {
    # Congress / Voteview
    "HSall_members.csv": (
        "https://voteview.com/static/data/out/members/HSall_members.csv",
        "Congress members with DW-NOMINATE ideology scores"
    ),
    "HSall_votes.csv": (
        "https://voteview.com/static/data/out/votes/HSall_votes.csv",
        "Individual cast-code votes for every roll call"
    ),
    "HSall_rollcalls.csv": (
        "https://voteview.com/static/data/out/rollcalls/HSall_rollcalls.csv",
        "Roll call metadata (date, description, yea/nay counts)"
    ),
    # UN General Assembly — TidyTuesday 2021-03-23 mirror
    "un_votes.csv": (
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
        "master/data/2021/2021-03-23/unvotes.csv",
        "Country-level yes/no/abstain votes on every UNGA resolution"
    ),
    "un_roll_calls.csv": (
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
        "master/data/2021/2021-03-23/roll_calls.csv",
        "UNGA roll call metadata"
    ),
    "un_roll_call_issues.csv": (
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
        "master/data/2021/2021-03-23/issues.csv",
        "Issue-area tags for UNGA resolutions"
    ),
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream-download *url* to *dest*, showing progress."""
    print(f"  Downloading {dest.name} …", end=" ", flush=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                downloaded += len(chunk)
        mb = downloaded / 1e6
        print(f"done ({mb:.1f} MB)")


def download_all(force: bool = False) -> None:
    """Download every dataset.  Skip files that already exist unless *force*."""
    print(f"Data directory: {DATA_DIR}\n")
    for filename, (url, description) in DATASETS.items():
        dest = DATA_DIR / filename
        if dest.exists() and not force:
            size_mb = dest.stat().st_size / 1e6
            print(f"  {filename} already present ({size_mb:.1f} MB) — skip")
            continue
        try:
            _download_file(url, dest)
        except Exception as exc:
            print(f"  ERROR downloading {filename}: {exc}")
    print("\nAll downloads complete.")


def verify_downloads() -> bool:
    """Quick sanity-check: confirm each file is a parseable CSV."""
    print("\nVerifying downloads …")
    ok = True
    for filename in DATASETS:
        dest = DATA_DIR / filename
        if not dest.exists():
            print(f"  MISSING: {filename}")
            ok = False
            continue
        try:
            df = pd.read_csv(dest, nrows=5)
            print(f"  {filename}: {list(df.columns)}")
        except Exception as exc:
            print(f"  CORRUPT {filename}: {exc}")
            ok = False
    return ok


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    force = "--force" in sys.argv
    download_all(force=force)
    verify_downloads()
