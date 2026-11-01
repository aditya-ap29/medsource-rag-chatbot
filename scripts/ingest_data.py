"""
Script to ingest medical documents and build vector store
"""
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion import (
    MedicalCorpusBuilder,
    PubMedLoader,
    MedlinePlusLoader,
    OpenFDADrugLoader,
    ClinicalTrialsLoader,
)
from src.vector_store import VectorStore
from src.config import get_settings

# Load environment variables
load_dotenv()


def ingest_local_documents(corpus_builder: MedicalCorpusBuilder, vector_store: VectorStore):
    """Ingest documents from local directory"""
    print("\n" + "="*60)
    print("ðŸ“‚ Loading local documents...")
    print("="*60)
    
    documents = corpus_builder.load_local_documents()
    
    if not documents:
        print("âš ï¸  No documents found in ./data/raw_documents/")
        print("   Please add PDF, DOCX, TXT, or JSON files to this directory")
        return 0
    
    print(f"âœ… Loaded {len(documents)} documents")
    
    # Save corpus
    corpus_builder.save_corpus(documents, "local_corpus.json")
    
    # Add to vector store
    print("\nðŸ“Š Adding documents to vector store...")
    vector_store.add_documents(documents)
    
    return len(documents)


def ingest_pubmed_abstracts(
    corpus_builder: MedicalCorpusBuilder,
    vector_store: VectorStore,
    queries: list,
    max_results_per_query: int = 50
):
    """Ingest abstracts from PubMed"""
    print("\n" + "="*60)
    print("ðŸ”¬ Fetching PubMed abstracts...")
    print("="*60)
    
    email = os.getenv("PUBMED_EMAIL", "your_email@example.com")
    pubmed_loader = PubMedLoader(email=email)

    # Skip articles already present in the vector store (relevant when
    # appending across multiple batch runs with overlapping topics).
    existing_doc_ids = {doc["doc_id"] for doc in vector_store.documents}

    all_documents = []
    skipped_duplicates = 0

    for query in queries:
        print(f"\nðŸ” Searching: {query}")
        pmids = pubmed_loader.search_pubmed(query, max_results=max_results_per_query)
        print(f"   Found {len(pmids)} articles")

        if pmids:
            print(f"   Fetching abstracts...")
            documents = pubmed_loader.fetch_abstracts(pmids)
            new_documents = []
            for doc in documents:
                if doc.doc_id in existing_doc_ids:
                    skipped_duplicates += 1
                else:
                    existing_doc_ids.add(doc.doc_id)
                    new_documents.append(doc)
            print(f"   Retrieved {len(documents)} abstracts ({len(new_documents)} new, {len(documents) - len(new_documents)} already in store)")
            all_documents.extend(new_documents)

    if not all_documents:
        print(f"âš ï¸  No new PubMed abstracts retrieved ({skipped_duplicates} duplicates skipped)")
        return 0

    print(f"\nâœ… Total new PubMed documents: {len(all_documents)} ({skipped_duplicates} duplicates skipped)")
    
    # Save corpus
    corpus_builder.save_corpus(all_documents, "pubmed_corpus.json")
    
    # Add to vector store
    print("\nðŸ“Š Adding PubMed documents to vector store...")
    vector_store.add_documents(all_documents)
    
    return len(all_documents)


def create_sample_medical_data():
    """Create sample medical documents for testing"""
    print("\n" + "="*60)
    print("ðŸ“ Creating sample medical documents...")
    print("="*60)
    
    from src.data_ingestion import MedicalDocument
    
    sample_docs = [
        MedicalDocument(
            content="Vitamin D deficiency is a common condition that can lead to various health problems. "
                   "Common symptoms include fatigue, bone pain, muscle weakness, and mood changes such as depression. "
                   "In children, severe deficiency can cause rickets, leading to soft bones and skeletal deformities. "
                   "In adults, it can lead to osteomalacia, causing bone pain and muscle weakness. "
                   "Risk factors include limited sun exposure, dark skin, older age, obesity, and certain medical conditions. "
                   "Treatment typically involves vitamin D supplementation and increased sun exposure.",
            title="Vitamin D Deficiency: Symptoms and Treatment",
            source="Harrison's Internal Medicine",
            year="2018",
            url="https://www.ncbi.nlm.nih.gov/books/NBK532266/",
            metadata={"category": "nutrition", "severity": "common"}
        ),
        MedicalDocument(
            content="Anemia is a condition in which you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues. "
                   "Common symptoms include fatigue, weakness, pale or yellowish skin, irregular heartbeats, shortness of breath, "
                   "dizziness or lightheadedness, chest pain, cold hands and feet, and headaches. "
                   "There are many types of anemia, each with different causes. Iron deficiency anemia is the most common type worldwide. "
                   "Treatment depends on the type and cause but may include dietary changes, supplements, or medications.",
            title="Anemia: Types, Symptoms, and Treatment Options",
            source="Mayo Clinic Medical Reference",
            year="2022",
            url="https://www.mayoclinic.org/diseases-conditions/anemia",
            metadata={"category": "hematology", "severity": "common"}
        ),
        MedicalDocument(
            content="Diabetes mellitus is a metabolic disease that causes high blood sugar. "
                   "Type 1 diabetes is an autoimmune condition where the body doesn't produce insulin. "
                   "Type 2 diabetes occurs when the body becomes resistant to insulin or doesn't produce enough. "
                   "Common symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, "
                   "fatigue, blurred vision, slow-healing sores, and frequent infections. "
                   "Management includes blood sugar monitoring, insulin or medication, healthy eating, and regular exercise.",
            title="Diabetes Mellitus: Overview and Management",
            source="World Health Organization Guidelines",
            year="2021",
            url="https://www.who.int/health-topics/diabetes",
            metadata={"category": "endocrinology", "severity": "chronic"}
        ),
        MedicalDocument(
            content="Hypertension, or high blood pressure, is a common condition in which the force of blood against artery walls is too high. "
                   "It often has no symptoms but can lead to serious health problems including heart disease, stroke, and kidney disease. "
                   "Normal blood pressure is less than 120/80 mmHg. Hypertension is diagnosed when readings consistently exceed 130/80 mmHg. "
                   "Risk factors include age, family history, obesity, sedentary lifestyle, high salt intake, and excessive alcohol consumption. "
                   "Treatment includes lifestyle modifications and antihypertensive medications as prescribed by healthcare providers.",
            title="Hypertension: Definition, Risk Factors, and Management",
            source="American Heart Association Guidelines",
            year="2020",
            url="https://www.heart.org/en/health-topics/high-blood-pressure",
            metadata={"category": "cardiology", "severity": "common"}
        ),
        MedicalDocument(
            content="Migraine is a neurological condition characterized by intense, debilitating headaches. "
                   "Symptoms include throbbing pain, usually on one side of the head, nausea, vomiting, and extreme sensitivity to light and sound. "
                   "Some people experience aura before migraines, which can include visual disturbances, tingling sensations, or difficulty speaking. "
                   "Triggers vary but may include stress, certain foods, hormonal changes, and sleep pattern changes. "
                   "Treatment options include pain-relieving medications, preventive medications, and lifestyle modifications.",
            title="Migraine Headaches: Symptoms, Triggers, and Treatment",
            source="NIH National Institute of Neurological Disorders",
            year="2023",
            url="https://www.ninds.nih.gov/migraine",
            metadata={"category": "neurology", "severity": "moderate"}
        ),
        MedicalDocument(
            content="Asthma is a chronic respiratory condition that inflames and narrows the airways. "
                   "Common symptoms include shortness of breath, chest tightness, wheezing, and coughing, especially at night or early morning. "
                   "Asthma attacks can be triggered by allergens, cold air, exercise, stress, or respiratory infections. "
                   "The condition varies from person to person in terms of severity and frequency of symptoms. "
                   "Management includes avoiding triggers, using inhaled corticosteroids for long-term control, and rescue inhalers for quick relief. "
                   "Regular monitoring with a peak flow meter can help track lung function.",
            title="Asthma: Pathophysiology and Clinical Management",
            source="Journal of Respiratory Medicine",
            year="2022",
            url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9876543/",
            metadata={"category": "pulmonology", "severity": "chronic"}
        )
    ]
    
    # Save sample documents
    data_dir = Path("./data/raw_documents")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    sample_file = data_dir / "sample_medical_data.json"
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump([doc.to_dict() for doc in sample_docs], f, indent=2, ensure_ascii=False)
    
    print(f"âœ… Created {len(sample_docs)} sample documents")
    print(f"   Saved to: {sample_file}")
    
    return sample_docs


# ---------------------------------------------------------------------------
# Curated PubMed query presets
# ---------------------------------------------------------------------------

# A broad, high-coverage set of PubMed queries used by --source pubmed-broad.
# Covers the most common conditions people ask medical chatbots about.
BROAD_PUBMED_QUERIES = [
    # Infectious diseases
    "influenza symptoms treatment adults",
    "COVID-19 SARS-CoV-2 clinical management",
    "malaria diagnosis treatment antimalarials",
    "dengue fever clinical presentation management",
    "tuberculosis diagnosis treatment first-line",
    "HIV antiretroviral therapy outcomes",
    "urinary tract infection antibiotics treatment",
    "sepsis early recognition management",
    # Cardiovascular
    "hypertension treatment guidelines antihypertensive",
    "heart failure diagnosis management",
    "acute myocardial infarction percutaneous coronary intervention",
    "atrial fibrillation anticoagulation management",
    "stroke ischemic thrombolysis outcomes",
    # Endocrine / Metabolic
    "type 2 diabetes mellitus glycemic control metformin",
    "type 1 diabetes insulin therapy",
    "hypothyroidism levothyroxine treatment",
    "hyperthyroidism Graves disease treatment",
    "obesity metabolic syndrome GLP-1 semaglutide",
    "polycystic ovary syndrome PCOS management",
    # Respiratory
    "asthma inhaled corticosteroid bronchodilator",
    "COPD chronic obstructive pulmonary disease exacerbation",
    "pneumonia community-acquired antibiotics",
    "pulmonary embolism anticoagulation diagnosis",
    # Gastrointestinal
    "gastroesophageal reflux disease GERD proton pump inhibitor",
    "irritable bowel syndrome low FODMAP treatment",
    "inflammatory bowel disease Crohn colitis biologics",
    "celiac disease gluten-free diet",
    "peptic ulcer H pylori eradication",
    "liver cirrhosis complications management",
    # Neurological / Mental health
    "migraine prophylaxis acute treatment triptans",
    "epilepsy seizure antiepileptic drug",
    "major depressive disorder antidepressant psychotherapy",
    "generalized anxiety disorder cognitive behavioral therapy",
    "ADHD attention deficit hyperactivity disorder stimulant methylphenidate",
    "Alzheimer disease dementia management",
    "Parkinson disease dopaminergic treatment",
    # Musculoskeletal
    "rheumatoid arthritis DMARD biologic treatment",
    "osteoarthritis knee pain management",
    "osteoporosis bisphosphonate fracture prevention",
    "gout uric acid allopurinol management",
    # Renal
    "chronic kidney disease progression management",
    "nephrolithiasis kidney stone treatment prevention",
    # Hematological
    "iron deficiency anemia treatment supplementation",
    "vitamin B12 deficiency peripheral neuropathy",
    # Dermatological
    "atopic dermatitis eczema treatment biologics",
    "psoriasis biologic treatment outcomes",
    # Nutritional
    "vitamin D deficiency supplementation",
    "malnutrition undernutrition children management",
]


def ingest_medlineplus(
    corpus_builder: MedicalCorpusBuilder,
    vector_store,
    queries: list,
) -> int:
    """Fetch and ingest MedlinePlus NLM health-topic articles."""
    print("\n" + "=" * 60)
    print("📖 Fetching MedlinePlus (NLM/NIH) articles...")
    print("=" * 60)

    existing_doc_ids = {doc["doc_id"] for doc in vector_store.documents}
    loader = MedlinePlusLoader()
    all_docs = []
    skipped = 0

    for query in queries:
        print(f"\n🔍 Searching MedlinePlus: {query}")
        docs = loader.search(query, max_results=5)
        new_docs = [d for d in docs if d.doc_id not in existing_doc_ids]
        skipped += len(docs) - len(new_docs)
        for d in new_docs:
            existing_doc_ids.add(d.doc_id)
        print(f"   {len(new_docs)} new article(s)")
        all_docs.extend(new_docs)

    if not all_docs:
        print(f"⚠️  No new MedlinePlus articles retrieved ({skipped} duplicates skipped).")
        return 0

    print(f"\n✅ Total new MedlinePlus documents: {len(all_docs)}")
    corpus_builder.save_corpus(all_docs, "medlineplus_corpus.json")
    print("\n📊 Adding MedlinePlus documents to vector store...")
    vector_store.add_documents(all_docs)
    return len(all_docs)


def ingest_openfda(
    corpus_builder: MedicalCorpusBuilder,
    vector_store,
    drug_names: list,
    max_per_drug: int = 3,
    api_key: str = "",
) -> int:
    """Fetch and ingest FDA drug labels from openFDA."""
    print("\n" + "=" * 60)
    print("💊 Fetching FDA drug labels (openFDA)...")
    print("=" * 60)

    existing_doc_ids = {doc["doc_id"] for doc in vector_store.documents}
    loader = OpenFDADrugLoader(api_key=api_key or None)
    all_docs = []
    skipped = 0

    for drug in drug_names:
        print(f"\n🔍 Fetching label: {drug}")
        docs = loader.search_drug(drug, max_results=max_per_drug)
        new_docs = [d for d in docs if d.doc_id not in existing_doc_ids]
        skipped += len(docs) - len(new_docs)
        for d in new_docs:
            existing_doc_ids.add(d.doc_id)
        print(f"   {len(new_docs)} new section(s)")
        all_docs.extend(new_docs)

    if not all_docs:
        print(f"⚠️  No new FDA drug label sections retrieved ({skipped} duplicates skipped).")
        return 0

    print(f"\n✅ Total new FDA drug label documents: {len(all_docs)}")
    corpus_builder.save_corpus(all_docs, "openfda_corpus.json")
    print("\n📊 Adding FDA drug label documents to vector store...")
    vector_store.add_documents(all_docs)
    return len(all_docs)


def ingest_clinical_trials(
    corpus_builder: MedicalCorpusBuilder,
    vector_store,
    queries: list,
    max_per_query: int = 5,
    status: str = "COMPLETED",
) -> int:
    """Fetch and ingest completed clinical trial summaries from ClinicalTrials.gov."""
    print("\n" + "=" * 60)
    print("🔬 Fetching clinical trial summaries (ClinicalTrials.gov)...")
    print("=" * 60)

    existing_doc_ids = {doc["doc_id"] for doc in vector_store.documents}
    loader = ClinicalTrialsLoader()
    all_docs = []
    skipped = 0

    for query in queries:
        print(f"\n🔍 Searching trials: {query}")
        docs = loader.search(query, max_results=max_per_query, status=status)
        new_docs = [d for d in docs if d.doc_id not in existing_doc_ids]
        skipped += len(docs) - len(new_docs)
        for d in new_docs:
            existing_doc_ids.add(d.doc_id)
        print(f"   {len(new_docs)} new trial(s)")
        all_docs.extend(new_docs)

    if not all_docs:
        print(f"⚠️  No new clinical trials retrieved ({skipped} duplicates skipped).")
        return 0

    print(f"\n✅ Total new clinical trial documents: {len(all_docs)}")
    corpus_builder.save_corpus(all_docs, "clinical_trials_corpus.json")
    print("\n📊 Adding clinical trial documents to vector store...")
    vector_store.add_documents(all_docs)
    return len(all_docs)


def main():
    parser = argparse.ArgumentParser(description="Ingest medical documents and build vector store")
    parser.add_argument(
        "--source",
        choices=["local", "pubmed", "pubmed-broad", "sample", "medlineplus",
                 "openfda", "clinical-trials", "all"],
        default="all",
        help=(
            "Data source to ingest:\n"
            "  local          — files in data/raw_documents/\n"
            "  pubmed         — PubMed queries (use --pubmed-queries)\n"
            "  pubmed-broad   — broad preset of 55 PubMed queries covering common conditions\n"
            "  sample         — built-in 6-document sample set\n"
            "  medlineplus    — MedlinePlus NLM/NIH health topics (free, no key)\n"
            "  openfda        — FDA drug labels (free, no key; use --drugs)\n"
            "  clinical-trials— ClinicalTrials.gov completed study summaries\n"
            "  all            — local + sample + medlineplus (good default bootstrap)"
        )
    )
    parser.add_argument(
        "--pubmed-queries",
        nargs="+",
        default=[
            "vitamin deficiency symptoms treatment",
            "common medical conditions management",
            "chronic disease treatment outcomes",
        ],
        help="PubMed search queries (used with --source pubmed)"
    )
    parser.add_argument(
        "--max-pubmed-results",
        type=int,
        default=50,
        help="Maximum results per PubMed query"
    )
    parser.add_argument(
        "--drugs",
        nargs="+",
        default=[
            "metformin", "lisinopril", "atorvastatin", "amlodipine",
            "levothyroxine", "omeprazole", "amoxicillin", "ibuprofen",
            "sertraline", "albuterol",
        ],
        help="Drug names for --source openfda"
    )
    parser.add_argument(
        "--clinical-trial-queries",
        nargs="+",
        default=[
            "type 2 diabetes", "hypertension", "asthma", "depression",
            "rheumatoid arthritis", "heart failure",
        ],
        help="Condition queries for --source clinical-trials"
    )
    parser.add_argument(
        "--medlineplus-queries",
        nargs="+",
        default=[
            "diabetes", "hypertension", "anemia", "asthma", "depression",
            "anxiety", "arthritis", "thyroid", "kidney disease", "stroke",
        ],
        help="Search queries for --source medlineplus"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any existing vector store and start empty, instead of appending to it"
    )

    args = parser.parse_args()

    # Load settings
    settings = get_settings()

    # Initialize components
    corpus_builder = MedicalCorpusBuilder()
    vector_store = VectorStore(
        embedding_model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )

    # If a vector store already exists on disk, load it and ADD to it instead
    # of starting from empty. This lets ingestion be run in multiple
    # smaller batches (safer at scale) without losing previously ingested data.
    # Use --fresh to explicitly discard any existing store and start empty.
    existing_store_path = Path(settings.vector_store_path)
    if not args.fresh and existing_store_path.exists() and (existing_store_path / "faiss.index").exists():
        print(f"\n📂 Existing vector store found at {settings.vector_store_path} — loading to append...")
        vector_store.load(settings.vector_store_path)
        print(f"   Starting from {len(vector_store.documents)} existing chunks.\n")
    else:
        print("\n📂 No existing vector store found (or --fresh was passed) — starting empty.\n")

    total_docs = 0

    # Create sample data if requested
    if args.source in ["sample", "all"]:
        sample_docs = create_sample_medical_data()

        # Add sample documents to vector store
        print("\n📊 Adding sample documents to vector store...")
        vector_store.add_documents(sample_docs)
        total_docs += len(sample_docs)

        # Save corpus
        corpus_builder.save_corpus(sample_docs, "sample_corpus.json")

    # Ingest from local files
    if args.source in ["local", "all"]:
        total_docs += ingest_local_documents(corpus_builder, vector_store)

    # Ingest from PubMed (custom queries)
    if args.source == "pubmed":
        total_docs += ingest_pubmed_abstracts(
            corpus_builder,
            vector_store,
            args.pubmed_queries,
            args.max_pubmed_results
        )

    # Ingest from PubMed (broad preset — 55 queries)
    if args.source == "pubmed-broad":
        total_docs += ingest_pubmed_abstracts(
            corpus_builder,
            vector_store,
            BROAD_PUBMED_QUERIES,
            max_results_per_query=20,
        )

    # Ingest from MedlinePlus
    if args.source in ["medlineplus", "all"]:
        total_docs += ingest_medlineplus(
            corpus_builder, vector_store, args.medlineplus_queries
        )

    # Ingest from openFDA drug labels
    if args.source == "openfda":
        total_docs += ingest_openfda(
            corpus_builder, vector_store, args.drugs
        )

    # Ingest from ClinicalTrials.gov
    if args.source == "clinical-trials":
        total_docs += ingest_clinical_trials(
            corpus_builder, vector_store, args.clinical_trial_queries
        )

    # Save vector store
    if total_docs > 0:
        print("\n" + "="*60)
        print("ðŸ’¾ Saving vector store...")
        print("="*60)
        vector_store.save(settings.vector_store_path)
        
        stats = vector_store.get_stats()
        print(f"\nâœ… Vector store created successfully!")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Unique documents: {stats['unique_documents']}")
        print(f"   Saved to: {settings.vector_store_path}")
    else:
        print("\nâš ï¸  No documents ingested. Vector store not created.")
    
    print("\n" + "="*60)
    print("ðŸŽ‰ Ingestion complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Set your GEMINI_API_KEY in .env file (get a free key at aistudio.google.com/apikey)")
    print("2. Run the API server: python -m src.api")
    print("3. Test the endpoint: curl -X POST http://localhost:8000/query \\")
    print("   -H 'Content-Type: application/json' \\")
    print("   -d '{\"question\": \"What are the symptoms of anemia?\"}'")


if __name__ == "__main__":
    main()

