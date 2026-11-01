"""
Expand the MedSource corpus with three under-represented topic areas:
  1. Paediatrics      — MedlinePlus paediatric health topics
  2. Drug interactions — openFDA drug label interactions sections
  3. Mental health    — NIMH (nimh.nih.gov) condition fact pages via web scrape

Usage (from project root, venv active):
    python scripts/expand_corpus.py [--dry-run] [--skip-paeds] [--skip-interactions] [--skip-mental-health]

Flags:
  --dry-run            Print what would be fetched / added without modifying the store.
  --skip-paeds         Skip MedlinePlus paediatric fetch.
  --skip-interactions  Skip openFDA drug interactions fetch.
  --skip-mental-health Skip NIMH mental health fetch.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import get_settings
from src.data_ingestion import MedicalDocument, MedlinePlusLoader, OpenFDADrugLoader
from src.vector_store import VectorStore

settings = get_settings()

# ── Source lists ─────────────────────────────────────────────────────────────

# MedlinePlus — paediatric and adolescent topics not well covered by adult corpus
PAEDS_QUERIES = [
    "childhood asthma pediatric",
    "pediatric type 1 diabetes children",
    "childhood obesity children BMI",
    "attention deficit hyperactivity disorder ADHD children",
    "childhood vaccines immunization schedule",
    "pediatric fever management children",
    "growth hormone deficiency children",
    "sickle cell disease children",
    "autism spectrum disorder children",
    "pediatric epilepsy seizure children",
    "childhood leukemia ALL children",
    "juvenile idiopathic arthritis children",
    "neonatal jaundice newborn",
    "pediatric respiratory syncytial virus RSV",
    "cystic fibrosis children lung",
]

# openFDA drug labels — interaction-heavy drugs not yet in the corpus
INTERACTION_DRUGS = [
    "warfarin",
    "methotrexate",
    "lithium",
    "phenytoin",
    "amoxicillin",
    "ciprofloxacin",
    "atorvastatin",
    "omeprazole",
    "sertraline",
    "prednisone",
    "furosemide",
    "lisinopril",
    "levothyroxine",
    "clopidogrel",
    "fluoxetine",
]

# NIMH mental health fact-sheet URLs (static, verified 2025)
NIMH_PAGES = {
    "Depression":                "https://www.nimh.nih.gov/health/topics/depression",
    "Anxiety Disorders":         "https://www.nimh.nih.gov/health/topics/anxiety-disorders",
    "Bipolar Disorder":          "https://www.nimh.nih.gov/health/topics/bipolar-disorder",
    "Schizophrenia":             "https://www.nimh.nih.gov/health/topics/schizophrenia",
    "PTSD":                      "https://www.nimh.nih.gov/health/topics/post-traumatic-stress-disorder-ptsd",
    "OCD":                       "https://www.nimh.nih.gov/health/topics/obsessive-compulsive-disorder-ocd",
    "ADHD":                      "https://www.nimh.nih.gov/health/topics/attention-deficit-hyperactivity-disorder-adhd",
    "Eating Disorders":          "https://www.nimh.nih.gov/health/topics/eating-disorders",
    "Suicide Prevention":        "https://www.nimh.nih.gov/health/topics/suicide-prevention",
    "Autism Spectrum Disorder":  "https://www.nimh.nih.gov/health/topics/autism-spectrum-disorder-asd",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_store() -> VectorStore:
    vs = VectorStore(
        embedding_model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_batch_size=settings.embedding_batch_size,
    )
    store_path = Path(settings.vector_store_path)
    if store_path.exists():
        vs.load(str(store_path))
        print(f"Store loaded: {len(vs.documents)} chunks")
    else:
        print("No existing store found — starting fresh.")
    return vs


def existing_doc_ids(vs: VectorStore) -> set:
    return {d["doc_id"] for d in vs.documents}


def deduplicate(docs: list, existing: set) -> list:
    seen = set(existing)
    out = []
    for d in docs:
        if d.doc_id in seen:
            continue
        seen.add(d.doc_id)
        out.append(d)
    return out


def scrape_nimh(title: str, url: str) -> MedicalDocument | None:
    """Scrape a single NIMH fact page and return a MedicalDocument."""
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=15, headers={"User-Agent": "MedSourceBot/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        # NIMH pages put main content in <article> or <main>
        main = soup.find("article") or soup.find("main") or soup
        text = " ".join(main.get_text(separator=" ").split())
        if len(text) < 200:
            print(f"  ⚠️  {title}: page too short ({len(text)} chars), skipping.")
            return None
        return MedicalDocument(
            content=text[:6000],  # cap at ~1500 tokens
            title=f"NIMH: {title}",
            source="NIMH Mental Health",
            year="2024",
            url=url,
            metadata={"category": "mental health", "scrape_source": "nimh.nih.gov"},
        )
    except Exception as e:
        print(f"  ⚠️  {title}: scrape failed — {e}")
        return None


# ── Fetch functions ───────────────────────────────────────────────────────────

def fetch_paediatrics(dry_run: bool) -> list:
    print("\n── Paediatrics (MedlinePlus) ─────────────────────────────────")
    loader = MedlinePlusLoader()
    docs = []
    for q in PAEDS_QUERIES:
        try:
            results = loader.search(q, max_results=3)
            docs.extend(results)
            print(f"  {q!r:50s} → {len(results)} docs")
        except Exception as e:
            print(f"  {q!r}: error — {e}")
        time.sleep(0.3)
    print(f"  Total paediatric docs fetched: {len(docs)}")
    return docs


def fetch_interactions(dry_run: bool) -> list:
    print("\n── Drug interactions (openFDA) ───────────────────────────────")
    loader = OpenFDADrugLoader()
    docs = []
    for drug in INTERACTION_DRUGS:
        try:
            results = loader.search_drug(drug, max_results=2)
            docs.extend(results)
            print(f"  {drug:30s} → {len(results)} label sections")
        except Exception as e:
            print(f"  {drug}: error — {e}")
        time.sleep(0.4)
    print(f"  Total drug-label docs fetched: {len(docs)}")
    return docs


def fetch_mental_health(dry_run: bool) -> list:
    print("\n── Mental health (NIMH scrape) ───────────────────────────────")
    docs = []
    for title, url in NIMH_PAGES.items():
        doc = scrape_nimh(title, url)
        if doc:
            docs.append(doc)
            print(f"  ✓ {title}")
        time.sleep(0.5)
    print(f"  Total NIMH docs fetched: {len(docs)}")
    return docs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Expand MedSource corpus with paediatrics, drug interactions, and mental health")
    ap.add_argument("--dry-run",            action="store_true")
    ap.add_argument("--skip-paeds",         action="store_true")
    ap.add_argument("--skip-interactions",  action="store_true")
    ap.add_argument("--skip-mental-health", action="store_true")
    args = ap.parse_args()

    vs = load_store()
    existing = existing_doc_ids(vs)
    all_new: list[MedicalDocument] = []

    if not args.skip_paeds:
        docs = fetch_paediatrics(args.dry_run)
        new = deduplicate(docs, existing | {d.doc_id for d in all_new})
        print(f"  New after dedup: {len(new)}")
        all_new.extend(new)

    if not args.skip_interactions:
        docs = fetch_interactions(args.dry_run)
        new = deduplicate(docs, existing | {d.doc_id for d in all_new})
        print(f"  New after dedup: {len(new)}")
        all_new.extend(new)

    if not args.skip_mental_health:
        docs = fetch_mental_health(args.dry_run)
        new = deduplicate(docs, existing | {d.doc_id for d in all_new})
        print(f"  New after dedup: {len(new)}")
        all_new.extend(new)

    print(f"\nTotal new documents to add: {len(all_new)}")

    if args.dry_run:
        print("DRY RUN — nothing written to disk.")
        return

    if not all_new:
        print("Nothing new to add.")
        return

    added = vs.add_documents(all_new)
    vs.save(settings.vector_store_path)
    stats = vs.get_stats()
    print(f"\n✅ Done.  Added {added} chunks.  Store now: {stats['total_chunks']} chunks / {stats['unique_documents']} docs.")
    print("Restart the API / Streamlit app to load the updated store.")


if __name__ == "__main__":
    main()
