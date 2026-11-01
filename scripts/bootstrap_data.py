"""
bootstrap_data.py — One-shot data pipeline for MedSource.

Runs every data source in order, deduplicates across sources,
builds (or rebuilds) the vector store with BAAI/bge-base-en-v1.5,
and prints a final summary.

Usage (from project root, venv active):
    python scripts/bootstrap_data.py [--fresh] [--skip-pubmed] [--skip-medlineplus]
                                     [--skip-openfda] [--skip-trials]

Flags:
  --fresh           Wipe the existing vector store and start from scratch.
                    Without this flag, new docs are appended to the existing store.
  --skip-pubmed     Skip all PubMed fetching (useful if quota is exhausted).
  --skip-medlineplus  Skip MedlinePlus fetching.
  --skip-openfda    Skip openFDA drug label fetching.
  --skip-trials     Skip ClinicalTrials.gov fetching.
"""
import argparse
import json
import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import get_settings
from src.data_ingestion import (
    MedicalDocument,
    MedicalCorpusBuilder,
    PubMedLoader,
    MedlinePlusLoader,
    OpenFDADrugLoader,
    ClinicalTrialsLoader,
)
from src.vector_store import VectorStore

settings = get_settings()

# ──────────────────────────────────────────────────────────────────────────────
# Curated source lists
# ──────────────────────────────────────────────────────────────────────────────

PUBMED_QUERIES = [
    # Infectious disease
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
    # Cardiovascular
    "hypertension treatment guidelines antihypertensive medications",
    "heart failure reduced ejection fraction treatment",
    "acute myocardial infarction STEMI PCI reperfusion",
    "atrial fibrillation anticoagulation stroke prevention",
    "ischemic stroke thrombolysis tPA outcomes",
    "coronary artery disease prevention statin",
    "deep vein thrombosis DVT pulmonary embolism anticoagulation",
    # Endocrine / Metabolic
    "type 2 diabetes mellitus glycemic control metformin GLP-1",
    "type 1 diabetes insulin therapy continuous glucose monitoring",
    "hypothyroidism levothyroxine treatment TSH",
    "hyperthyroidism Graves disease antithyroid methimazole",
    "obesity management bariatric semaglutide lifestyle",
    "polycystic ovary syndrome PCOS insulin resistance treatment",
    "metabolic syndrome prevention cardiovascular risk",
    # Respiratory
    "asthma inhaled corticosteroid bronchodilator management",
    "COPD chronic obstructive pulmonary disease exacerbation treatment",
    "sleep apnea OSA CPAP treatment outcomes",
    "pulmonary fibrosis IPF nintedanib pirfenidone",
    # Gastrointestinal
    "gastroesophageal reflux GERD proton pump inhibitor",
    "irritable bowel syndrome IBS low FODMAP treatment",
    "inflammatory bowel disease Crohn ulcerative colitis biologics",
    "celiac disease gluten-free diet management",
    "peptic ulcer H pylori eradication bismuth",
    "liver cirrhosis hepatic encephalopathy management",
    "colorectal cancer screening colonoscopy prevention",
    # Neurological / Mental health
    "migraine prophylaxis triptans acute treatment",
    "epilepsy seizure antiepileptic valproate levetiracetam",
    "major depressive disorder SSRIs antidepressant psychotherapy",
    "generalized anxiety disorder cognitive behavioral therapy SSRI",
    "ADHD attention deficit methylphenidate behavioral therapy",
    "Alzheimer disease dementia cholinesterase inhibitor",
    "Parkinson disease levodopa dopaminergic therapy",
    "stroke rehabilitation outcomes recovery",
    "multiple sclerosis disease modifying therapy interferon",
    # Musculoskeletal
    "rheumatoid arthritis methotrexate biologic DMARD",
    "osteoarthritis knee pain management analgesics",
    "osteoporosis bisphosphonate fracture prevention DEXA",
    "gout uric acid allopurinol febuxostat management",
    "systemic lupus erythematosus SLE hydroxychloroquine",
    "ankylosing spondylitis TNF inhibitor management",
    # Renal
    "chronic kidney disease CKD progression RAAS inhibitor",
    "acute kidney injury AKI prevention treatment",
    "nephrolithiasis kidney stone lithotripsy management",
    "nephrotic syndrome proteinuria immunosuppression",
    # Hematological
    "iron deficiency anemia oral intravenous iron supplementation",
    "vitamin B12 deficiency cobalamin pernicious anemia",
    "sickle cell disease hydroxyurea management",
    "deep vein thrombosis anticoagulation heparin warfarin",
    # Dermatological
    "atopic dermatitis eczema dupilumab topical corticosteroid",
    "psoriasis biologic anti-TNF IL-17 treatment",
    "acne vulgaris retinoid benzoyl peroxide isotretinoin",
    "melanoma immunotherapy checkpoint inhibitor",
    # Cancer
    "breast cancer adjuvant chemotherapy hormone therapy",
    "lung cancer NSCLC immunotherapy checkpoint inhibitor",
    "prostate cancer androgen deprivation therapy",
    "cervical cancer HPV vaccination screening",
    # Pediatrics
    "childhood asthma inhaler spacer management",
    "pediatric fever management antipyretics",
    "ADHD children stimulant behavioral therapy outcomes",
    "childhood obesity prevention lifestyle",
    # Women's health
    "pregnancy hypertension preeclampsia management",
    "gestational diabetes insulin management",
    "menopause hormone replacement therapy HRT",
    "endometriosis treatment hormonal surgical",
    # Nutrition / Preventive
    "vitamin D deficiency supplementation guidelines",
    "obesity prevention diet exercise intervention",
    "dyslipidemia statin lipid lowering treatment",
    "smoking cessation pharmacotherapy varenicline",
    # Emergency / Critical care
    "anaphylaxis epinephrine emergency management",
    "cardiac arrest CPR resuscitation outcomes",
    "traumatic brain injury management outcomes",
]

MEDLINEPLUS_QUERIES = [
    "diabetes", "hypertension high blood pressure", "anemia",
    "asthma", "depression", "anxiety", "arthritis", "thyroid disorders",
    "kidney disease", "stroke", "heart attack", "heart failure",
    "COPD", "pneumonia", "influenza flu", "COVID-19",
    "cancer screening", "obesity", "cholesterol", "osteoporosis",
    "migraine headache", "epilepsy seizures", "Parkinson disease",
    "Alzheimer dementia", "multiple sclerosis", "HIV AIDS",
    "tuberculosis", "malaria", "hepatitis", "sexually transmitted infections",
    "urinary tract infection", "kidney stones", "gallstones",
    "GERD acid reflux", "irritable bowel syndrome", "celiac disease",
    "eczema atopic dermatitis", "psoriasis", "acne",
    "vitamin D deficiency", "iron deficiency", "vitamin B12",
    "sleep apnea insomnia", "eating disorders", "PTSD",
    "schizophrenia", "bipolar disorder", "autism spectrum disorder",
]

FDA_DRUGS = [
    # Diabetes
    "metformin", "insulin glargine", "semaglutide", "empagliflozin",
    "sitagliptin", "glipizide",
    # Cardiovascular
    "lisinopril", "amlodipine", "atorvastatin", "metoprolol",
    "losartan", "warfarin", "apixaban", "clopidogrel",
    "furosemide", "spironolactone",
    # Respiratory
    "albuterol", "fluticasone", "montelukast", "tiotropium",
    # GI
    "omeprazole", "pantoprazole", "ondansetron",
    # Antibiotics
    "amoxicillin", "azithromycin", "ciprofloxacin",
    "doxycycline", "metronidazole",
    # Neuro / Mental health
    "sertraline", "escitalopram", "fluoxetine",
    "alprazolam", "zolpidem", "methylphenidate",
    "valproate", "levetiracetam", "gabapentin",
    # Pain / Inflammation
    "ibuprofen", "naproxen", "acetaminophen",
    "prednisone", "celecoxib",
    # Thyroid
    "levothyroxine", "methimazole",
    # Other
    "alendronate", "allopurinol", "hydroxychloroquine",
    "methotrexate",
]

CLINICAL_TRIAL_QUERIES = [
    "type 2 diabetes GLP-1 semaglutide",
    "hypertension antihypertensive",
    "asthma biologic dupilumab",
    "major depressive disorder antidepressant",
    "rheumatoid arthritis biologic",
    "heart failure HFrEF",
    "atrial fibrillation anticoagulation",
    "COPD tiotropium bronchodilator",
    "chronic kidney disease ACE inhibitor",
    "obesity lifestyle bariatric",
    "Alzheimer dementia treatment",
    "Parkinson disease levodopa",
    "breast cancer adjuvant",
    "lung cancer immunotherapy",
    "HIV antiretroviral",
    "osteoporosis bisphosphonate",
    "migraine preventive treatment",
    "inflammatory bowel disease biologic",
    "epilepsy antiepileptic",
    "stroke rehabilitation",
]

# ──────────────────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


def load_patient_education_summaries() -> list:
    """Load the curated patient-education JSON from raw_documents/."""
    path = Path("data/raw_documents/patient_education_summaries.json")
    if not path.exists():
        print(f"  [!]  Not found: {path}")
        return []
    items = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for item in items:
        if isinstance(item, dict) and item.get("content"):
            docs.append(MedicalDocument(
                content=item["content"],
                title=item.get("title", "Patient Education Summary"),
                source=item.get("source", "Patient Education Summary"),
                year=item.get("year"),
                url=item.get("url"),
                metadata=item.get("metadata", {}),
            ))
    print(f"  [OK] Loaded {len(docs)} patient-education documents")
    return docs


def fetch_pubmed(queries: list, email: str, max_per_query: int, existing_ids: set) -> list:
    loader = PubMedLoader(email=email)
    all_docs, skipped = [], 0
    for q in queries:
        print(f"  [?>] PubMed: {q[:70]}")
        pmids = loader.search_pubmed(q, max_results=max_per_query)
        if pmids:
            docs = loader.fetch_abstracts(pmids)
            for d in docs:
                if d.doc_id in existing_ids:
                    skipped += 1
                else:
                    existing_ids.add(d.doc_id)
                    all_docs.append(d)
            print(f"     {len(docs)} fetched, {len(docs)-skipped if skipped else len(docs)} new")
    print(f"  [OK] PubMed total: {len(all_docs)} new docs ({skipped} duplicates skipped)")
    return all_docs


def fetch_medlineplus(queries: list, existing_ids: set) -> list:
    loader = MedlinePlusLoader()
    all_docs, skipped = [], 0
    for q in queries:
        docs = loader.search(q, max_results=8)
        for d in docs:
            if d.doc_id in existing_ids:
                skipped += 1
            else:
                existing_ids.add(d.doc_id)
                all_docs.append(d)
    print(f"  [OK] MedlinePlus: {len(all_docs)} new docs ({skipped} duplicates skipped)")
    return all_docs


def fetch_openfda(drugs: list, existing_ids: set) -> list:
    loader = OpenFDADrugLoader()
    all_docs, skipped = [], 0
    for drug in drugs:
        docs = loader.search_drug(drug, max_results=2)
        for d in docs:
            if d.doc_id in existing_ids:
                skipped += 1
            else:
                existing_ids.add(d.doc_id)
                all_docs.append(d)
    print(f"  [OK] openFDA: {len(all_docs)} new docs ({skipped} duplicates skipped)")
    return all_docs


def fetch_clinical_trials(queries: list, existing_ids: set) -> list:
    loader = ClinicalTrialsLoader()
    all_docs, skipped = [], 0
    for q in queries:
        docs = loader.search(q, max_results=8, status="COMPLETED", min_enrollment=50)
        for d in docs:
            if d.doc_id in existing_ids:
                skipped += 1
            else:
                existing_ids.add(d.doc_id)
                all_docs.append(d)
    print(f"  [OK] ClinicalTrials: {len(all_docs)} new docs ({skipped} duplicates skipped)")
    return all_docs


def main():
    parser = argparse.ArgumentParser(description="MedSource data bootstrap")
    parser.add_argument("--fresh", action="store_true",
                        help="Wipe existing vector store and rebuild from scratch")
    parser.add_argument("--skip-pubmed", action="store_true")
    parser.add_argument("--skip-medlineplus", action="store_true")
    parser.add_argument("--skip-openfda", action="store_true")
    parser.add_argument("--skip-trials", action="store_true")
    parser.add_argument("--pubmed-per-query", type=int, default=20,
                        help="Max PubMed results per query (default 20)")
    args = parser.parse_args()

    pubmed_email = os.getenv("PUBMED_EMAIL", "")
    if not pubmed_email and not args.skip_pubmed:
        print("[!]  PUBMED_EMAIL not set in .env — skipping PubMed ingestion.")
        args.skip_pubmed = True

    print_section("MedSource Data Bootstrap")
    print(f"  Embedding model : {settings.embedding_model}")
    print(f"  Chunk size      : {settings.chunk_size} tokens")
    print(f"  Chunk overlap   : {settings.chunk_overlap} tokens")
    print(f"  Vector store    : {settings.vector_store_path}")
    print(f"  Mode            : {'FRESH REBUILD' if args.fresh else 'APPEND'}")

    # ── Initialise vector store ──────────────────────────────────────────────
    vs = VectorStore(
        embedding_model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_batch_size=settings.embedding_batch_size,
    )
    corpus_builder = MedicalCorpusBuilder()
    store_path = Path(settings.vector_store_path)

    if not args.fresh and store_path.exists() and (store_path / "faiss.index").exists():
        print(f"\n  [DIR] Loading existing store to append...")
        vs.load(settings.vector_store_path)
        print(f"     Starting from {len(vs.documents)} existing chunks.")
    else:
        print(f"\n  [DIR] Starting fresh (empty store).")

    # Track IDs already in the store to avoid re-adding duplicates
    existing_ids = {d["doc_id"] for d in vs.documents}
    total_added = 0

    # ── 1. Patient-education summaries ──────────────────────────────────────
    print_section("1 / 5  Patient Education Summaries")
    pe_docs = load_patient_education_summaries()
    new_pe = [d for d in pe_docs if d.doc_id not in existing_ids]
    if new_pe:
        for d in new_pe:
            existing_ids.add(d.doc_id)
        added = vs.add_documents(new_pe)
        corpus_builder.save_corpus(new_pe, "patient_education_corpus.json")
        total_added += added
        print(f"  [+] {added} chunks added from patient-education summaries")
    else:
        print("  ✓  All patient-education summaries already in store — skipped")

    # ── 2. MedlinePlus (NLM/NIH) ────────────────────────────────────────────
    print_section("2 / 5  MedlinePlus (NLM / NIH)")
    if args.skip_medlineplus:
        print("  Skipped (--skip-medlineplus)")
    else:
        ml_docs = fetch_medlineplus(MEDLINEPLUS_QUERIES, existing_ids)
        if ml_docs:
            added = vs.add_documents(ml_docs)
            corpus_builder.save_corpus(ml_docs, "medlineplus_corpus.json")
            total_added += added
            print(f"  [+] {added} chunks added from MedlinePlus")
        else:
            print("  No new MedlinePlus documents.")

    # ── 3. openFDA Drug Labels ───────────────────────────────────────────────
    print_section("3 / 5  FDA Drug Labels (openFDA)")
    if args.skip_openfda:
        print("  Skipped (--skip-openfda)")
    else:
        fda_docs = fetch_openfda(FDA_DRUGS, existing_ids)
        if fda_docs:
            added = vs.add_documents(fda_docs)
            corpus_builder.save_corpus(fda_docs, "openfda_corpus.json")
            total_added += added
            print(f"  [+] {added} chunks added from openFDA")
        else:
            print("  No new openFDA documents.")

    # ── 4. ClinicalTrials.gov ────────────────────────────────────────────────
    print_section("4 / 5  ClinicalTrials.gov")
    if args.skip_trials:
        print("  Skipped (--skip-trials)")
    else:
        ct_docs = fetch_clinical_trials(CLINICAL_TRIAL_QUERIES, existing_ids)
        if ct_docs:
            added = vs.add_documents(ct_docs)
            corpus_builder.save_corpus(ct_docs, "clinical_trials_corpus.json")
            total_added += added
            print(f"  [+] {added} chunks added from ClinicalTrials.gov")
        else:
            print("  No new ClinicalTrials.gov documents.")

    # ── 5. PubMed ────────────────────────────────────────────────────────────
    print_section("5 / 5  PubMed (NCBI)")
    if args.skip_pubmed:
        print("  Skipped (--skip-pubmed)")
    else:
        pm_docs = fetch_pubmed(PUBMED_QUERIES, pubmed_email, args.pubmed_per_query, existing_ids)
        if pm_docs:
            added = vs.add_documents(pm_docs)
            corpus_builder.save_corpus(pm_docs, "pubmed_bootstrap_corpus.json")
            total_added += added
            print(f"  [+] {added} chunks added from PubMed")
        else:
            print("  No new PubMed documents.")

    # ── Save ─────────────────────────────────────────────────────────────────
    print_section("Saving vector store")
    if total_added > 0:
        vs.save(settings.vector_store_path)
        stats = vs.get_stats()
        print(f"\n  [OK] Done!")
        print(f"     Total chunks        : {stats['total_chunks']}")
        print(f"     Unique documents    : {stats['unique_documents']}")
        print(f"     Embedding model     : {stats['embedding_model']}")
        print(f"     Dimension           : {stats['dimension']}")
        print(f"     BGE prefix          : {'on' if stats['use_bge_prefix'] else 'off'}")
        print(f"     Avg chunk length    : {stats['avg_chunk_chars']} chars")
        print(f"     New chunks added    : {total_added}")
    else:
        print("  No new documents added — vector store unchanged.")

    print_section("Bootstrap complete")


if __name__ == "__main__":
    main()
