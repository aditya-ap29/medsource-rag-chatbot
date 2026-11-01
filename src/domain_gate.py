"""
src/domain_gate.py
------------------
Lightweight medical-domain relevance gate for MedSource.

Runs *before* answer generation to decide whether a query is:
  1. Clearly medical  → proceed to answer
  2. Emergency medical → proceed to answer (urgency preserved)
  3. Out-of-scope     → return polite redirect without a medical answer

Design rationale
----------------
BGE cosine similarity alone is not a reliable domain gate.  The 0.30
retrieval threshold was calibrated on *medical* queries: it ensures that a
medical question with poor lexical overlap still retrieves something useful.
But any English sentence — "What is the capital of France?" — will obtain
BGE scores in the 0.35–0.55 range against a large medical corpus because
BGE embeds semantic similarity across all topics, not just medicine.

This gate uses a *combination* of signals:

  A. Hard keyword guard — a compact curated set of anatomy, disease,
     drug, and procedure terms.  Broad enough to catch unusual phrasing
     ("I feel lightheaded after my statin") without being an exhaustive
     medical dictionary.  Emergency keywords are a strict superset of the
     medical keywords so emergencies always pass.

  B. Hard OOP guard — a compact curated set of explicit non-medical
     domain signals (currency, stock, weather, geography, sports…).
     If any OOP signal matches AND no medical signal matches, reject.

  C. Retrieval evidence gate — even if the query looks non-medical by
     keyword, if multiple retrieved chunks score ≥ RETRIEVAL_STRONG_THRESHOLD
     (0.62) it is likely a legitimate but unusually worded medical question.
     A single chunk at 0.62 is not enough — we require at least 2 to avoid
     false positives on sentences like "What time does the pharmacy open?".

  D. OOP retrieval ceiling — if the query matched NO medical keywords and
     ALL retrieved chunks score below OOP_SCORE_CEILING (0.55), treat as
     out-of-scope regardless of the retrieval gate above.

Emergency bypass
----------------
Queries containing emergency keywords (chest pain, overdose, stroke …)
always pass the gate.  The gate must never suppress an emergency response.

Tuning
------
All thresholds are module-level constants and can be overridden from
`config.py` if needed.  The keyword sets are deliberately conservative:
it is better to pass a borderline query to the LLM (which will self-limit
via the SYSTEM_PROMPT) than to silently block a genuine medical question.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Minimum BGE score for a single chunk to count as "strong" retrieval evidence.
RETRIEVAL_STRONG_THRESHOLD = 0.62

# Minimum number of *strong* chunks required to override the keyword guard.
RETRIEVAL_STRONG_MIN_CHUNKS = 2

# If NO medical keywords match AND the top chunk score is below this value,
# the query is very likely out-of-scope.
OOP_SCORE_CEILING = 0.55

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

# Emergency keywords — any match → gate passes unconditionally.
# Keep this list broad: a false negative here is more dangerous than a false positive.
_EMERGENCY_TERMS: frozenset[str] = frozenset({
    "911", "ambulance", "emergency",
    "heart attack", "myocardial infarction", "mi ",
    "chest pain", "chest pressure", "crushing chest",
    "can't breathe", "cannot breathe", "difficulty breathing", "stop breathing",
    "stroke", "face drooping", "arm weakness", "speech difficulty", "fast acronym",
    "overdose", "opioid overdose", "drug overdose",
    "unconscious", "unresponsive", "not breathing",
    "anaphylaxis", "anaphylactic", "epipen", "epinephrine",
    "seizure", "convulsion", "status epilepticus",
    "bleeding", "haemorrhage", "hemorrhage",
    "choking", "heimlich",
    "suicide", "suicidal", "self-harm", "self harm",
    "sepsis", "septic shock",
    "code blue", "cpr", "defibrillator", "aed",
})

# Medical keywords — a match means the query is medical.
# This list purposely covers common layman phrasings, not just clinical terms.
_MEDICAL_TERMS: frozenset[str] = frozenset({
    # Generic medical terms (catch-all for named diseases not listed below)
    "disease", "disorder", "syndrome", "condition", "illness", "ailment",
    "neurodegenerat", "neuropath",
    # Body systems / anatomy
    "heart", "cardiac", "cardiovascular", "coronary", "artery", "vein",
    "lung", "pulmonary", "respiratory", "airway", "bronch",
    "brain", "neurolog", "neuron", "cerebr", "spinal", "nerve",
    "kidney", "renal", "nephr", "liver", "hepat", "gallbladder",
    "stomach", "gastro", "intestin", "bowel", "colon", "rectal",
    "pancrea", "thyroid", "adrenal", "pituitary",
    "bone", "joint", "muscle", "tendon", "ligament", "fracture",
    "skin", "dermat", "rash", "lesion", "wound",
    "blood", "hematol", "anaemia", "anemia", "platelet", "clot", "thromb",
    "immune", "lymph", "spleen", "antibod",
    "eye", "ophthalmol", "retina", "vision",
    "ear", "hearing", "vestibular",
    "uterus", "ovary", "cervix", "menstrual", "pregnancy", "obstetric",
    "prostate", "testicular", "urolog",
    # Diseases / conditions
    "diabetes", "diabetic", "hyperglycemia", "insulin",
    "hypertension", "blood pressure", "hypotension",
    "cancer", "tumor", "tumour", "malignant", "benign", "oncol",
    "infection", "bacterial", "viral", "fungal", "parasite",
    "fever", "temperature", "hyperthermia", "hypothermia",
    "asthma", "copd", "emphysema", "bronchitis",
    "pneumonia", "tuberculosis", "tb ", " tb",
    "hiv", "aids", "antiretroviral",
    "hepatitis", "cirrhosis", "fibrosis",
    "alzheimer", "dementia", "parkinson",
    "multiple sclerosis", "sclerosis", "amyotrophic",
    "huntington", "huntington's",
    "depression", "anxiety", "mental health", "psychiatric",
    "schizophrenia", "bipolar", "psychosis",
    "arthritis", "osteoporosis", "gout", "lupus",
    "eczema", "psoriasis", "dermatitis",
    "migraine", "headache", "vertigo",
    "epilepsy", "seizures",
    "ibs", "crohn", "colitis", "celiac",
    "thyroid", "hypothyroid", "hyperthyroid",
    "anaphylaxis", "allergy", "allergic",
    "obesity", "overweight", "bmi",
    "cholesterol", "lipid", "statin",
    "osteoarthritis", "rheumatoid",
    "appendicitis", "gallstone", "kidney stone",
    "copd", "fibrosis",
    "pcos", "endometriosis",
    "sepsis", "bacteremia",
    "vitamin", "deficiency", "malnutrition",
    "dehydration",
    "measles", "chickenpox", "mumps", "rubella",
    "dengue", "malaria", "zika",
    "covid", "coronavirus", "sars",
    "influenza", "flu",
    "syphilis", "gonorrhea", "chlamydia", "herpes", "hpv",
    "lyme disease", "typhoid", "cholera", "ebola",
    "pneumococcal", "meningitis", "encephalitis",
    "cancer", "carcinoma", "sarcoma", "leukemia", "lymphoma",
    "insomnia", "sleep disorder", "apnea",
    "autism", "adhd", "dyslexia",
    # Medications / treatments
    "medication", "medicine", "drug", "dose", "dosage", "prescription",
    "antibiotic", "antiviral", "antifungal", "antimalarial",
    "vaccine", "vaccination", "immunize",
    "surgery", "surgical", "operation", "procedure",
    "chemotherapy", "radiotherapy", "radiation therapy",
    "dialysis", "transplant",
    "aspirin", "ibuprofen", "paracetamol", "acetaminophen",
    "metformin", "metoprolol", "lisinopril", "atorvastatin",
    "antidepressant", "ssri", "snri",
    "opioid", "morphine", "naloxone", "narcan",
    "corticosteroid", "steroid", "prednisone",
    "inhaler", "bronchodilator", "salbutamol", "albuterol",
    "contraceptive", "contraception",
    # Diagnostics / procedures
    "test", "diagnosis", "diagnose", "lab", "laboratory",
    "blood test", "urine test", "biopsy",
    "mri", "ct scan", "x-ray", "xray", "ultrasound", "ecg", "ekg",
    "colonoscopy", "endoscopy",
    "hba1c", "creatinine", "hemoglobin", "haemoglobin",
    "tsh", "t3", "t4", "thyroid stimulating",
    # Symptoms / patient language
    "symptom", "pain", "ache", "hurt", "sore",
    "fatigue", "tired", "exhausted", "weakness",
    "nausea", "vomit", "diarrhea", "diarrhoea", "constipation",
    "swelling", "edema", "oedema",
    "cough", "sneeze", "runny nose", "congestion",
    "itch", "burning", "tingling", "numbness",
    "breathless", "shortness of breath",
    "dizziness", "lightheaded", "fainting",
    "blurry vision", "double vision",
    "palpitation", "irregular heartbeat",
    "weight loss", "weight gain",
    "appetite loss", "loss of appetite",
    "night sweats", "chills",
    "jaundice", "yellow skin",
    # Healthcare system terms
    "doctor", "physician", "nurse", "pharmacist", "specialist",
    "hospital", "clinic", "emergency room", "er visit", "icu", "ward",
    "patient", "treatment", "therapy", "management",
    "health", "medical", "clinical", "healthcare",
    "side effect", "adverse effect", "contraindication",
    "prognosis", "prevention", "screening",
    "chronic", "acute", "benign", "malignant",
    "first aid", "first-aid",
})

# OOP (out-of-scope) signals — a match here (with no medical match) → reject.
# Deliberately narrow: only topics that are clearly never medical questions.
_OOP_TERMS: frozenset[str] = frozenset({
    # Geography / travel
    "capital of", "country", "continent", "ocean", "river", "mountain",
    "city", "town", "population of", "timezone", "currency of",
    "flag of", "language of", "president of", "prime minister",
    "tourist", "travel to", "visa for", "passport",
    # Finance / economics
    "stock", "share price", "invest", "cryptocurrency", "bitcoin",
    "forex", "exchange rate", "inflation rate", "gdp",
    "mortgage", "loan interest", "bank account",
    "tax rate", "income tax", "property tax",
    # Weather / environment (non-medical)
    "weather today", "weather like", "weather in", "weather forecast",
    "forecast", "temperature in", "temperature today",
    "will it rain", "is it hot", "is it cold",
    "what is the weather", "how is the weather",
    # Sports / entertainment
    "sports score", "match result", "game result", "cricket score",
    "football score", "basketball score", "nba score", "nfl score", "ipl score",
    "celebrity", "singer", "band concert", "music album",
    "movie review", "film review", "box office",
    # Tech / non-health computing
    "programming language", "python syntax", "javascript",
    "software license", "github", "source code",
    # Poetry / creative
    "write me a poem", "write a poem", "write a story", "tell me a joke",
    "compose a song", "creative writing",
    # Food / recipes (non-nutritional)
    "recipe for", "how to cook", "cooking instructions",
    "ingredient list",
    # History / politics
    "world war", "historical event", "election result",
    "political party", "legislation",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

OUT_OF_SCOPE_RESPONSE = (
    "I'm MedSource, a medical information assistant. "
    "I can only answer questions about health, medical conditions, "
    "symptoms, treatments, and related topics. "
    "Your question appears to be outside that scope. "
    "Please ask a medical question and I'll be happy to help."
    "\n\n"
    "⚠️ This information is for educational purposes only and is not a "
    "substitute for professional medical advice."
)


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for matching."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _matches_any(text_lower: str, terms: frozenset) -> bool:
    return any(term in text_lower for term in terms)


def is_emergency(question: str) -> bool:
    """Return True if the question contains emergency keywords.

    This check is intentionally permissive: it is better to treat a
    borderline question as an emergency than to suppress urgent guidance.
    """
    q = _normalise(question)
    return _matches_any(q, _EMERGENCY_TERMS)


def classify_domain(
    question: str,
    retrieved_docs: List[Tuple[Dict[str, Any], float]],
) -> Tuple[str, str]:
    """Classify a query as 'medical', 'emergency', or 'out_of_scope'.

    Parameters
    ----------
    question : str
        The user's raw question text.
    retrieved_docs : list of (doc_dict, score) tuples
        The top-k documents already retrieved from the vector store.
        Scores are raw BGE cosine values (floats in ~[0, 1]).

    Returns
    -------
    (label, reason) where label is one of:
        'emergency'     — always proceed with urgent guidance
        'medical'       — proceed with normal answer generation
        'out_of_scope'  — return OUT_OF_SCOPE_RESPONSE, skip generation

    The reason string is a human-readable explanation for logging.
    """
    q = _normalise(question)

    # ----- Emergency bypass (always wins) ---------------------------------
    if _matches_any(q, _EMERGENCY_TERMS):
        return "emergency", "emergency keyword matched"

    # ----- Medical keyword match ------------------------------------------
    has_medical_kw = _matches_any(q, _MEDICAL_TERMS)

    # ----- OOP keyword match ----------------------------------------------
    has_oop_kw = _matches_any(q, _OOP_TERMS)

    # ----- Retrieval evidence ---------------------------------------------
    scores = [s for _, s in retrieved_docs] if retrieved_docs else []
    strong_count = sum(1 for s in scores if s >= RETRIEVAL_STRONG_THRESHOLD)
    max_score = max(scores) if scores else 0.0

    # Clear medical query: keyword match and no explicit OOP signals
    if has_medical_kw and not has_oop_kw:
        return "medical", "medical keyword matched"

    # Strong retrieval evidence overrides weak keyword ambiguity ONLY when
    # there are multiple strong-scoring chunks (avoids false positives from
    # coincidental partial matches like "What time does the pharmacy open?").
    if strong_count >= RETRIEVAL_STRONG_MIN_CHUNKS and not has_oop_kw:
        return "medical", f"retrieval evidence: {strong_count} chunks ≥ {RETRIEVAL_STRONG_THRESHOLD}"

    # Explicit OOP + no medical keywords → out of scope
    if has_oop_kw and not has_medical_kw:
        return "out_of_scope", "OOP keyword matched, no medical keyword"

    # No clear signal either way: use retrieval ceiling
    # If max score is below the OOP ceiling (scores are within the
    # "random background" range for BGE), treat as out-of-scope.
    if not has_medical_kw and max_score < OOP_SCORE_CEILING:
        return "out_of_scope", f"no medical keyword and max_score={max_score:.3f} < {OOP_SCORE_CEILING}"

    # Default: trust the medical pipeline (allow through)
    return "medical", "default pass (no strong OOP signal)"
