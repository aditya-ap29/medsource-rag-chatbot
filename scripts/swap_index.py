"""
scripts/swap_index.py
---------------------
Safely swap a new versioned index into the active position.

Steps:
  1. Validate that the new index directory exists and contains
     the required files.
  2. Backup the current active index to data/vector_store_backup/
     (overwriting any previous backup).
  3. Copy the new index files into data/vector_store/.
  4. Write a swap_log.json record.

Rollback:
  python scripts/swap_index.py --rollback
  Restores data/vector_store/ from data/vector_store_backup/.

The application must be restarted (or the Streamlit cache cleared)
after swapping.

Usage:
    # Activate new index:
    python scripts/swap_index.py [--new-index data/vector_store_v1]

    # Roll back to previous index:
    python scripts/swap_index.py --rollback
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ACTIVE_DIR  = ROOT / "data" / "vector_store"
BACKUP_DIR  = ROOT / "data" / "vector_store_backup"
DEFAULT_NEW = ROOT / "data" / "vector_store_v1"
SWAP_LOG    = ROOT / "data" / "swap_log.json"

REQUIRED_FILES = ["faiss.index", "documents.pkl", "config.pkl"]


def _check_required_files(index_dir: Path) -> list[str]:
    return [f for f in REQUIRED_FILES if not (index_dir / f).exists()]


def _write_log(record: dict) -> None:
    log = []
    if SWAP_LOG.exists():
        try:
            with open(SWAP_LOG) as fh:
                log = json.load(fh)
        except Exception:
            pass
    log.append(record)
    with open(SWAP_LOG, "w") as fh:
        json.dump(log, fh, indent=2)


def swap_index(new_dir: Path) -> None:
    """Swap new_dir into the active position with backup."""
    missing = _check_required_files(new_dir)
    if missing:
        print(f"ERROR: new index at {new_dir} is missing: {missing}")
        sys.exit(1)

    # Backup current active index
    if ACTIVE_DIR.exists():
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(ACTIVE_DIR, BACKUP_DIR)
        print(f"Backed up active index to {BACKUP_DIR}")
    else:
        print(f"No active index at {ACTIVE_DIR} (first-time activation).")

    # Copy new index files into active directory
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    for fname in new_dir.iterdir():
        shutil.copy2(new_dir / fname.name, ACTIVE_DIR / fname.name)
    print(f"Copied {new_dir} -> {ACTIVE_DIR}")

    # Verify the copy
    missing_after = _check_required_files(ACTIVE_DIR)
    if missing_after:
        print(f"ERROR: active index is missing files after swap: {missing_after}")
        print("Rolling back automatically...")
        rollback()
        sys.exit(1)

    record = {
        "action":     "swap",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "new_source": str(new_dir),
        "backup_at":  str(BACKUP_DIR),
    }
    _write_log(record)
    print(f"Swap complete. Log written to {SWAP_LOG}")
    print("Restart the Streamlit app (or clear st.cache_resource) to load the new index.")


def rollback() -> None:
    """Restore the backup index to the active position."""
    if not BACKUP_DIR.exists():
        print(f"ERROR: No backup found at {BACKUP_DIR}")
        sys.exit(1)

    missing = _check_required_files(BACKUP_DIR)
    if missing:
        print(f"ERROR: backup at {BACKUP_DIR} is incomplete: {missing}")
        sys.exit(1)

    if ACTIVE_DIR.exists():
        shutil.rmtree(ACTIVE_DIR)
    shutil.copytree(BACKUP_DIR, ACTIVE_DIR)
    print(f"Rolled back: {BACKUP_DIR} -> {ACTIVE_DIR}")

    record = {
        "action":    "rollback",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "restored_from": str(BACKUP_DIR),
    }
    _write_log(record)
    print(f"Rollback complete. Log written to {SWAP_LOG}")
    print("Restart the Streamlit app (or clear st.cache_resource) to reload.")


def main():
    parser = argparse.ArgumentParser(description="Swap or roll back the active FAISS index.")
    parser.add_argument(
        "--new-index",
        default=str(DEFAULT_NEW),
        help="Path to the new versioned index directory (default: data/vector_store_v1/).",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Roll back to the previous backup instead of swapping.",
    )
    args = parser.parse_args()

    if args.rollback:
        rollback()
    else:
        swap_index(Path(args.new_index))


if __name__ == "__main__":
    main()
