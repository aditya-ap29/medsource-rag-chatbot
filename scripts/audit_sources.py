"""
Audit where the documents in the vector store come from, and (optionally) remove
documents whose source label cannot be verified.

Why this exists
---------------
A citation is only worth something if the cited organisation actually wrote the
text. The sample documents that shipped with the starter code carried labels such
as "Mayo Clinic Medical Reference" (with a mayoclinic.org link) even though the
text was written by the template's author. Users would see those labels as
citations. This tool finds every such document.

What counts as verified
-----------------------
Only the source labels in ALLOWED_SOURCES are kept:
  - "PubMed"                    : abstracts fetched from NCBI, with PMID links
  - "Patient Education Summary" : the curated summaries written for this project
                                  (AI-drafted, not clinically reviewed)
Anything else is reported as NOT VERIFIED.

Usage (from the project root, venv active)
------------------------------------------
    python scripts/audit_sources.py            # preview only, changes nothing
    python scripts/audit_sources.py --apply    # backs up, then removes them

--apply:
  1. copies the store to <store>_backup_before_source_cleanup (once)
  2. removes the unverified chunks and rebuilds the FAISS index from the vectors
     already stored (no re-embedding, no API calls)
  3. moves any data/raw_documents/*.json file whose documents are ALL unverified
     into data/raw_documents_removed/, so a later `--source local` run cannot
     bring them back. Files that mix verified and unverified documents are only
     reported, never edited.
Restart the API / Streamlit app afterwards so they load the cleaned store.
"""
import argparse
import json
import pickle
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_SOURCES = {"PubMed", "Patient Education Summary"}
SKIP_DIRS = {"venv", ".venv", ".git", "__pycache__", "node_modules"}
TEXT_SUFFIXES = {".py", ".json", ".md", ".yml", ".yaml", ".txt", ".sh", ".ps1"}


def summarize(docs):
    """Return {source: {'chunks': n, 'doc_ids': set()}} in first-seen order."""
    by_source = OrderedDict()
    for d in docs:
        s = d.get("source") or "(no source label)"
        entry = by_source.setdefault(s, {"chunks": 0, "doc_ids": set()})
        entry["chunks"] += 1
        entry["doc_ids"].add(d.get("doc_id"))
    return by_source


def find_label_uses(root, terms, limit=40):
    """Find files in the project (not data/venv/.git) that contain any of the terms."""
    hits = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue  # this tool's own docstring mentions the labels it hunts for
        rel = path.relative_to(root)
        parts = set(rel.parts)
        if parts & SKIP_DIRS:
            continue
        if rel.parts[:2] in (("data", "vector_store"), ("data", "processed")) or \
                (len(rel.parts) > 1 and rel.parts[0] == "data" and rel.parts[1].startswith("vector_store")):
            continue
        try:
            for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if any(t.lower() in line.lower() for t in terms):
                    hits.append(f"  {rel}:{n}: {line.strip()[:110]}")
                    if len(hits) >= limit:
                        return hits
        except OSError:
            continue
    return hits


def raw_json_files_fully_unverified(raw_dir):
    """data/raw_documents/*.json files whose documents are ALL from unverified sources."""
    fully, mixed = [], []
    if not raw_dir.exists():
        return fully, mixed
    for f in sorted(raw_dir.glob("*.json")):
        try:
            items = json.loads(f.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(items, list) or not items or not all(isinstance(i, dict) for i in items):
            continue
        flags = [(i.get("source") not in ALLOWED_SOURCES) for i in items]
        if all(flags):
            fully.append(f)
        elif any(flags):
            mixed.append(f)
    return fully, mixed


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit and clean unverified document sources")
    ap.add_argument("--store", default="data/vector_store", help="vector store directory")
    ap.add_argument("--apply", action="store_true", help="actually remove unverified documents")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    store = Path(args.store)
    if not store.is_absolute():
        store = Path.cwd() / store
    docs_path = store / "documents.pkl"
    if not docs_path.exists():
        print(f"No vector store found at {store}")
        return 1

    docs = pickle.loads(docs_path.read_bytes())
    by_source = summarize(docs)
    all_ids = {d.get("doc_id") for d in docs}
    print(f"Vector store: {store}  ({len(docs)} chunks, {len(all_ids)} documents)\n")
    print("Sources in the store:")
    bad_sources = []
    for s, e in by_source.items():
        ok = s in ALLOWED_SOURCES
        if not ok:
            bad_sources.append(s)
        print(f"  {s[:44]:<44} {e['chunks']:>7} chunks {len(e['doc_ids']):>6} docs   "
              f"{'ok' if ok else 'NOT VERIFIED'}")

    if not bad_sources:
        print("\nEvery document comes from a verified source. Nothing to do.")
        return 0

    bad_docs = OrderedDict()
    for d in docs:
        if d.get("source") not in ALLOWED_SOURCES:
            bad_docs.setdefault(d.get("doc_id"), d)
    print(f"\nUnverified documents ({len(bad_docs)}):")
    for d in bad_docs.values():
        print(f"  - [{d.get('source')}] {str(d.get('title'))[:70]}  ({d.get('url')})")

    terms = set(bad_sources)
    for d in bad_docs.values():
        host = urlparse(d.get("url") or "").netloc.lower().removeprefix("www.")
        if host:
            terms.add(host)
    print("\nWhere these labels appear in project files (these also need fixing at the source):")
    hits = find_label_uses(root, terms)
    print("\n".join(hits) if hits else "  (none found)")

    raw_dir = root / "data" / "raw_documents"
    fully, mixed = raw_json_files_fully_unverified(raw_dir)
    if fully:
        print("\nRaw files that contain only unverified documents "
              "(would be moved to data/raw_documents_removed/):")
        for f in fully:
            print(f"  {f.relative_to(root)}")
    if mixed:
        print("\nRaw files mixing verified and unverified documents (NOT touched, edit by hand):")
        for f in mixed:
            print(f"  {f.relative_to(root)}")

    keep = [i for i, d in enumerate(docs) if d.get("source") in ALLOWED_SOURCES]
    removed_chunks = len(docs) - len(keep)
    if not args.apply:
        print(f"\nPREVIEW ONLY. --apply would remove {removed_chunks} chunks "
              f"({len(bad_docs)} documents) and keep {len(keep)}.")
        print("Nothing was changed. Re-run with --apply to proceed.")
        return 0

    import faiss
    index = faiss.read_index(str(store / "faiss.index"))
    if index.ntotal != len(docs):
        print(f"ERROR: index has {index.ntotal} vectors but documents.pkl has {len(docs)} chunks. "
              f"Refusing to modify a store that is already out of sync.")
        return 1

    backup = store.parent / (store.name + "_backup_before_source_cleanup")
    if backup.exists():
        print(f"\nBackup already exists at {backup} (kept as is).")
    else:
        shutil.copytree(store, backup)
        print(f"\nBackup written to {backup}")

    vectors = index.reconstruct_n(0, index.ntotal)
    new_index = faiss.IndexFlatIP(index.d)
    new_index.add(vectors[keep])
    kept_docs = [docs[i] for i in keep]

    faiss.write_index(new_index, str(store / "faiss.index"))
    (store / "documents.pkl").write_bytes(pickle.dumps(kept_docs))
    cfg_path = store / "config.pkl"
    if cfg_path.exists():
        cfg = pickle.loads(cfg_path.read_bytes())
        if isinstance(cfg, dict):
            cfg["num_documents"] = len(kept_docs)
            cfg_path.write_bytes(pickle.dumps(cfg))

    # Self-check: the index and the document list must still line up.
    check = faiss.read_index(str(store / "faiss.index"))
    assert check.ntotal == len(kept_docs), "index/document count mismatch after rewrite"
    if kept_docs:
        _, ids = check.search(vectors[keep][:1], 1)
        assert int(ids[0][0]) == 0, "alignment self-check failed"

    if fully:
        dest = raw_dir.parent / "raw_documents_removed"
        dest.mkdir(exist_ok=True)
        for f in fully:
            shutil.move(str(f), str(dest / f.name))
            print(f"Moved {f.name} -> data/raw_documents_removed/")

    print(f"\nDone. Removed {removed_chunks} chunks ({len(bad_docs)} documents). "
          f"Store now has {len(kept_docs)} chunks.")
    print("Restart the API / Streamlit app so they load the cleaned store.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

