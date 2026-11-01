"""
scripts/clean_corpus.py
-----------------------
Read-only corpus cleaner: reads every JSON file under data/processed/,
applies deterministic cleaning rules, deduplicates by doc_id and by
content hash, and writes a single consolidated clean corpus to
data/processed/clean_corpus_v1.json.

The original files are NEVER modified or deleted.

Cleaning rules applied (in order):
  1. HTML tag removal via regex (catches MedlinePlus span/li/p/ul tags).
  2. Whitespace normalisation: collapse runs of spaces/tabs; strip leading/
     trailing whitespace; collapse 3+ newlines to 2.
  3. Drop records with empty content after cleaning (content must be
     >= MIN_CONTENT_CHARS characters after stripping).
  4. Remove control characters (ASCII 0-8, 11-12, 14-31, 127).
  5. Deduplicate: keep the first occurrence of each (doc_id, source_file)
     pair, then among those keep the first occurrence of each content_hash.

Outputs:
  data/processed/clean_corpus_v1.json   – cleaned, deduplicated records
  data/processed/clean_corpus_v1_report.json – per-file cleaning stats

Run from project root:
    python scripts/clean_corpus.py [--dry-run]
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Constants ────────────────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_FILE   = PROCESSED_DIR / "clean_corpus_v1.json"
REPORT_FILE   = PROCESSED_DIR / "clean_corpus_v1_report.json"

# Minimum content length (characters) after all cleaning.
MIN_CONTENT_CHARS = 80

# Source files to include in the clean corpus, in priority order.
# pubmed_corpus.json supersedes pubmed_bootstrap_corpus.json for overlapping PMIDs.
# sample_corpus.json duplicates local_corpus.json — excluded.
SOURCE_FILES = [
    "pubmed_corpus.json",
    "pubmed_bootstrap_corpus.json",
    "medlineplus_corpus.json",
    "openfda_corpus.json",
    "clinical_trials_corpus.json",
    "local_corpus.json",          # includes patient_education
    "patient_education_corpus.json",
]

# ── Cleaning helpers ─────────────────────────────────────────────────────────

_HTML_TAG_RE   = re.compile(r"<[^>]+>")
_MULTI_NL_RE   = re.compile(r"\n{3,}")
_MULTI_WS_RE   = re.compile(r"[ \t]+")
_CTRL_CHAR_RE  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Ampersand HTML entities that survive tag removal
_HTML_ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);")
_ENTITY_MAP     = {"&amp;": "&", "&lt;": "<", "&gt;": ">",
                   "&quot;": '"', "&apos;": "'", "&nbsp;": " "}


def _decode_entity(m: re.Match) -> str:
    raw = m.group(0)
    return _ENTITY_MAP.get(raw, " ")


def clean_text(text: str) -> str:
    """Apply all cleaning steps to a single text string."""
    if not text:
        return ""
    # 1. Remove HTML tags
    text = _HTML_TAG_RE.sub(" ", text)
    # 2. Decode common HTML entities
    text = _HTML_ENTITY_RE.sub(_decode_entity, text)
    # 3. Remove control characters
    text = _CTRL_CHAR_RE.sub("", text)
    # 4. Collapse whitespace runs (preserve single newlines for readability)
    text = _MULTI_WS_RE.sub(" ", text)
    # 5. Collapse 3+ newlines to 2
    text = _MULTI_NL_RE.sub("\n\n", text)
    # 6. Strip
    return text.strip()


def content_hash(text: str) -> str:
    """MD5 of the cleaned content (used for exact-duplicate detection)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a new record dict with cleaned content and a content_hash field.
    Returns None if the record should be dropped."""
    raw_content = record.get("content") or ""
    cleaned     = clean_text(raw_content)
    if len(cleaned) < MIN_CONTENT_CHARS:
        return None

    out = dict(record)
    out["content"]      = cleaned
    out["content_hash"] = content_hash(cleaned)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def build_clean_corpus(dry_run: bool = False) -> None:
    print("=" * 65)
    print("MedSource corpus cleaner")
    print(f"  Source dir : {PROCESSED_DIR}")
    print(f"  Output     : {OUTPUT_FILE}")
    print(f"  Dry run    : {dry_run}")
    print("=" * 65)

    seen_doc_ids:     set[str] = set()
    seen_hashes:      set[str] = set()
    clean_records:    list[dict] = []
    report:           dict[str, dict] = {}

    total_in = total_dropped_short = total_dropped_dup_id = 0
    total_dropped_dup_hash = total_kept = 0

    for filename in SOURCE_FILES:
        src_path = PROCESSED_DIR / filename
        if not src_path.exists():
            print(f"  [SKIP] {filename} — file not found")
            report[filename] = {"status": "not_found"}
            continue

        with open(src_path, encoding="utf-8") as fh:
            try:
                raw_records = json.load(fh)
            except json.JSONDecodeError as exc:
                print(f"  [ERROR] {filename}: JSON parse error — {exc}")
                report[filename] = {"status": "json_error", "error": str(exc)}
                continue

        n_in = len(raw_records)
        n_dropped_short = n_dropped_dup_id = n_dropped_dup_hash = n_kept = 0

        for rec in raw_records:
            cleaned = clean_record(rec)

            if cleaned is None:
                n_dropped_short += 1
                continue

            doc_id = cleaned.get("doc_id", "")
            c_hash = cleaned["content_hash"]

            if doc_id and doc_id in seen_doc_ids:
                n_dropped_dup_id += 1
                continue

            if c_hash in seen_hashes:
                n_dropped_dup_hash += 1
                continue

            # Accept record
            if doc_id:
                seen_doc_ids.add(doc_id)
            seen_hashes.add(c_hash)
            clean_records.append(cleaned)
            n_kept += 1

        report[filename] = {
            "status": "ok",
            "total_in": n_in,
            "dropped_short_content": n_dropped_short,
            "dropped_duplicate_doc_id": n_dropped_dup_id,
            "dropped_duplicate_content": n_dropped_dup_hash,
            "kept": n_kept,
        }
        total_in             += n_in
        total_dropped_short  += n_dropped_short
        total_dropped_dup_id += n_dropped_dup_id
        total_dropped_dup_hash += n_dropped_dup_hash
        total_kept           += n_kept

        print(
            f"  {filename:<42} in={n_in:5d}  kept={n_kept:5d}"
            f"  short={n_dropped_short}  dup_id={n_dropped_dup_id}"
            f"  dup_content={n_dropped_dup_hash}"
        )

    # Verify no HTML survived
    html_residual = sum(
        1 for r in clean_records if re.search(r"<[a-zA-Z][^>]*>", r["content"])
    )

    print()
    print("=" * 65)
    print(f"  Total input records  : {total_in:,}")
    print(f"  Dropped — too short  : {total_dropped_short:,}")
    print(f"  Dropped — dup doc_id : {total_dropped_dup_id:,}")
    print(f"  Dropped — dup content: {total_dropped_dup_hash:,}")
    print(f"  Kept (clean corpus)  : {total_kept:,}")
    print(f"  HTML residual check  : {html_residual} records still contain tags")
    print("=" * 65)

    if html_residual:
        print("  WARNING: HTML residual > 0 — check clean_text() logic")

    report["_summary"] = {
        "total_in": total_in,
        "dropped_short": total_dropped_short,
        "dropped_dup_doc_id": total_dropped_dup_id,
        "dropped_dup_content": total_dropped_dup_hash,
        "kept": total_kept,
        "html_residual": html_residual,
        "output_file": str(OUTPUT_FILE),
    }

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(clean_records, fh, indent=2, ensure_ascii=False)
    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nWrote {total_kept:,} clean records to {OUTPUT_FILE}")
    print(f"Wrote cleaning report  to {REPORT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and deduplicate the MedSource corpus.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse only; do not write output files.")
    args = parser.parse_args()
    build_clean_corpus(dry_run=args.dry_run)
