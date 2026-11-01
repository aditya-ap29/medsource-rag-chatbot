"""
pubmed_append.py  — fetch PubMed, save corpus JSON, then embed + save store.
Processes in small batches so each can complete within a reasonable time.
Usage:
    python scripts/pubmed_append.py          # runs all 82 queries, 8 per query
    python scripts/pubmed_append.py --start 0 --end 40   # first half
    python scripts/pubmed_append.py --start 40 --end 82  # second half
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from src.config import get_settings
from src.data_ingestion import MedicalDocument, PubMedLoader, MedicalCorpusBuilder
from src.vector_store import VectorStore

QUERIES = [
    "COVID-19 SARS-CoV-2 clinical management treatment outcomes",
    "influenza seasonal flu treatment antiviral oseltamivir",
    "pneumonia community-acquired antibiotics treatment guidelines",
    "tuberculosis diagnosis treatment first-line regimen",
    "malaria Plasmodium diagnosis artemisinin treatment",
    "dengue fever clinical presentation management",
    "HIV AIDS antiretroviral therapy ART outcomes",
    "sepsis early recognition management bundle",
    "urinary tract infection antibiotics treatment women",
    "Clostridioides difficile C diff infection treatment",
    "hypertension treatment guidelines antihypertensive medications",
    "heart failure reduced ejection fraction treatment",
    "acute myocardial infarction STEMI PCI reperfusion",
    "atrial fibrillation anticoagulation stroke prevention",
    "ischemic stroke thrombolysis tPA outcomes",
    "coronary artery disease prevention statin",
    "deep vein thrombosis DVT pulmonary embolism anticoagulation",
    "type 2 diabetes mellitus glycemic control metformin GLP-1",
    "type 1 diabetes insulin therapy continuous glucose monitoring",
    "hypothyroidism levothyroxine treatment TSH",
    "hyperthyroidism Graves disease antithyroid methimazole",
    "obesity management bariatric semaglutide lifestyle",
    "polycystic ovary syndrome PCOS insulin resistance treatment",
    "metabolic syndrome prevention cardiovascular risk",
    "asthma inhaled corticosteroid bronchodilator management",
    "COPD chronic obstructive pulmonary disease exacerbation treatment",
    "sleep apnea OSA CPAP treatment outcomes",
    "pulmonary fibrosis IPF nintedanib pirfenidone",
    "gastroesophageal reflux GERD proton pump inhibitor",
    "irritable bowel syndrome IBS low FODMAP treatment",
    "inflammatory bowel disease Crohn ulcerative colitis biologics",
    "celiac disease gluten-free diet management",
    "peptic ulcer H pylori eradication bismuth",
    "liver cirrhosis hepatic encephalopathy management",
    "colorectal cancer screening colonoscopy prevention",
    "migraine prophylaxis triptans acute treatment",
    "epilepsy seizure antiepileptic valproate levetiracetam",
    "major depressive disorder SSRIs antidepressant psychotherapy",
    "generalized anxiety disorder cognitive behavioral therapy SSRI",
    "ADHD attention deficit methylphenidate behavioral therapy",
    "Alzheimer disease dementia cholinesterase inhibitor",
    "Parkinson disease levodopa dopaminergic therapy",
    "stroke rehabilitation outcomes recovery",
    "multiple sclerosis disease modifying therapy interferon",
    "rheumatoid arthritis methotrexate biologic DMARD",
    "osteoarthritis knee pain management analgesics",
    "osteoporosis bisphosphonate fracture prevention DEXA",
    "gout uric acid allopurinol febuxostat management",
    "systemic lupus erythematosus SLE hydroxychloroquine",
    "ankylosing spondylitis TNF inhibitor management",
    "chronic kidney disease CKD progression RAAS inhibitor",
    "acute kidney injury AKI prevention treatment",
    "nephrolithiasis kidney stone lithotripsy management",
    "nephrotic syndrome proteinuria immunosuppression",
    "iron deficiency anemia oral intravenous iron supplementation",
    "vitamin B12 deficiency cobalamin pernicious anemia",
    "sickle cell disease hydroxyurea management",
    "atopic dermatitis eczema dupilumab topical corticosteroid",
    "psoriasis biologic anti-TNF IL-17 treatment",
    "acne vulgaris retinoid benzoyl peroxide isotretinoin",
    "melanoma immunotherapy checkpoint inhibitor",
    "breast cancer adjuvant chemotherapy hormone therapy",
    "lung cancer NSCLC immunotherapy checkpoint inhibitor",
    "prostate cancer androgen deprivation therapy",
    "cervical cancer HPV vaccination screening",
    "childhood asthma inhaler spacer management",
    "pediatric fever management antipyretics",
    "pregnancy hypertension preeclampsia management",
    "gestational diabetes insulin management",
    "menopause hormone replacement therapy HRT",
    "endometriosis treatment hormonal surgical",
    "vitamin D deficiency supplementation guidelines",
    "dyslipidemia statin lipid lowering treatment",
    "smoking cessation pharmacotherapy varenicline",
    "anaphylaxis epinephrine emergency management",
    "cardiac arrest CPR resuscitation outcomes",
    "traumatic brain injury management outcomes",
]

CORPUS_FILE = Path("data/processed/pubmed_bootstrap_corpus.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(QUERIES))
    parser.add_argument("--per-query", type=int, default=10)
    parser.add_argument("--embed-only", action="store_true",
                        help="Skip fetching; only embed+save existing corpus JSON")
    args = parser.parse_args()

    settings = get_settings()
    import os
    email = os.getenv("PUBMED_EMAIL", "")
    if not email and not args.embed_only:
        print("ERROR: PUBMED_EMAIL not set in .env")
        sys.exit(1)

    # ── Load store ────────────────────────────────────────────────────────────
    vs = VectorStore(
        embedding_model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_batch_size=settings.embedding_batch_size,
    )
    store_path = Path(settings.vector_store_path)
    if store_path.exists() and (store_path / "faiss.index").exists():
        vs.load(str(store_path))
        print(f"Store loaded: {len(vs.documents)} existing chunks.")
    existing_ids = {d["doc_id"] for d in vs.documents}

    # ── Load existing corpus accumulator ─────────────────────────────────────
    if CORPUS_FILE.exists():
        saved = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))
        print(f"Resuming corpus: {len(saved)} docs already saved to {CORPUS_FILE}")
        for item in saved:
            existing_ids.add(item["doc_id"])
    else:
        saved = []

    if not args.embed_only:
        # ── Fetch PubMed ──────────────────────────────────────────────────────
        loader = PubMedLoader(email=email)
        subset = QUERIES[args.start:args.end]
        print(f"\nFetching queries [{args.start}:{args.end}] ({len(subset)} queries, {args.per_query} per query)...")

        new_items = []
        for q in subset:
            print(f"  {q[:70]}")
            pmids = loader.search_pubmed(q, max_results=args.per_query)
            if pmids:
                docs = loader.fetch_abstracts(pmids)
                for d in docs:
                    if d.doc_id not in existing_ids:
                        existing_ids.add(d.doc_id)
                        new_items.append(d.to_dict())

        print(f"\nFetched {len(new_items)} new docs. Saving corpus JSON...")
        saved.extend(new_items)
        CORPUS_FILE.write_text(
            json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Corpus saved: {len(saved)} total docs in {CORPUS_FILE}")

    # ── Embed all saved-but-not-yet-indexed docs ──────────────────────────────
    in_store_ids = {d["doc_id"] for d in vs.documents}
    to_embed = [item for item in saved if item["doc_id"] not in in_store_ids]
    print(f"\n{len(to_embed)} docs to embed and index.")

    if to_embed:
        med_docs = [
            MedicalDocument(
                content=item["content"],
                title=item["title"],
                source=item["source"],
                doc_id=item["doc_id"],
                year=item.get("year"),
                url=item.get("url"),
                metadata=item.get("metadata", {}),
            )
            for item in to_embed
        ]
        added = vs.add_documents(med_docs)
        vs.save(str(store_path))
        stats = vs.get_stats()
        print(f"\nSaved.  Total chunks: {stats['total_chunks']}  |  Unique docs: {stats['unique_documents']}")
    else:
        print("All corpus docs already in store.")


if __name__ == "__main__":
    main()
