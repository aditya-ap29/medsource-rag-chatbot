"""
Minimal data ingestion - creates sample data without downloading large models
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion import MedicalDocument

def create_sample_medical_data():
    """Create sample medical documents"""
    print("📝 Creating sample medical documents...")
    
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
    
    # Save to JSON file
    data_dir = Path("./data/raw_documents")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = data_dir / "sample_medical_data.json"
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump([doc.to_dict() for doc in sample_docs], f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {len(sample_docs)} sample documents")
    print(f"   Saved to: {sample_file}")
    
    # Also save processed corpus
    processed_dir = Path("./data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    corpus_file = processed_dir / "corpus.json"
    with open(corpus_file, 'w', encoding='utf-8') as f:
        json.dump([doc.to_dict() for doc in sample_docs], f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved corpus to: {corpus_file}")
    
    return sample_docs


if __name__ == "__main__":
    print("\n" + "="*60)
    print("📚 MedSource - Minimal Data Setup")
    print("="*60 + "\n")
    
    try:
        docs = create_sample_medical_data()
        
        print("\n" + "="*60)
        print("✅ Sample data created successfully!")
        print("="*60)
        print("\nNote: Vector store will be created automatically when you start the API")
        print("The embedding model will download when first needed (only once)")
        print("\nNext step:")
        print("  python -m src.api")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
