# 🚀 Quick Start Guide - MedSource

This guide will get you up and running with MedSource in under 10 minutes!

## 📋 Prerequisites

- Python 3.8+ installed
- Git installed
- OpenAI API key (or plan to use open-source LLMs)

## ⚡ Installation (5 minutes)

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/MedSource-rag-chatbot.git
cd MedSource-rag-chatbot

# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API key
nano .env  # or use any text editor
```

**Option A: Using OpenAI (Recommended for hackathon)**
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
LLM_PROVIDER=openai
```

**Option B: Using Mistral (Open Source, Free)**
```bash
LLM_PROVIDER=mistral
# No API key needed, runs locally!
```

**Option C: Using Qwen (Open Source, Free)**
```bash
LLM_PROVIDER=qwen
# No API key needed, runs locally!
```

### Step 3: Ingest Sample Data

```bash
# This creates sample medical documents and builds the vector store
python scripts/ingest_data.py --source sample

# Expected output:
# ✅ Created 6 sample documents
# ✅ Vector store created successfully!
# Total chunks: 124
```

### Step 4: Start the API Server

```bash
python -m src.api

# Expected output:
# 🚀 Starting MedSource Medical RAG Chatbot
# 📚 Loading vector store...
# ✅ Vector store loaded successfully
#    - Total chunks: 124
#    - Unique documents: 6
# 🤖 Initializing RAG pipeline...
# ✅ RAG pipeline initialized successfully
# 🎯 API Server ready at http://0.0.0.0:8000
```

Server is now running! Keep this terminal open.

## 🧪 Test It (3 minutes)

### Option 1: Using curl (Terminal)

Open a **new terminal** and run:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the symptoms of vitamin D deficiency?",
    "top_k": 5,
    "include_sources": true
  }'
```

### Option 2: Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "What are the symptoms of anemia?",
        "top_k": 5,
        "include_sources": True
    }
)

print(response.json()["answer"])
```

### Option 3: Using the Test Script

```bash
python scripts/test_api.py
```

### Option 4: Browser (Interactive Docs)

Open your browser and go to:
```
http://localhost:8000/docs
```

This opens Swagger UI where you can test all endpoints interactively!

## 📊 Expected Response

You should see something like:

```json
{
  "question": "What are the symptoms of vitamin D deficiency?",
  "answer": "Common symptoms include fatigue, bone pain, muscle weakness, and mood changes [DOC_1]. Prolonged deficiency can lead to rickets in children or osteomalacia in adults [DOC_1]...",
  "sources": [
    {
      "doc_id": "DOC_1",
      "title": "Vitamin D Deficiency: Symptoms and Treatment",
      "year": "2018",
      "relevance_score": 0.89,
      "excerpt": "Vitamin D deficiency is a common condition..."
    }
  ],
  "confidence": 0.89,
  "retrieval_time_ms": 45.2,
  "generation_time_ms": 1823.5,
  "total_time_ms": 1868.7
}
```

## 🎯 Next Steps

### Add More Medical Data

**Option 1: Add your own PDF/DOCX files**
```bash
# Place medical documents in this folder:
mkdir -p data/raw_documents
# Copy your PDFs, DOCX, TXT files there

# Re-run ingestion
python scripts/ingest_data.py --source local
```

**Option 2: Fetch from PubMed**
```bash
python scripts/ingest_data.py --source pubmed \
  --pubmed-queries "diabetes treatment" "hypertension management" "cancer therapy"
```

**Option 3: Use all sources**
```bash
python scripts/ingest_data.py --source all \
  --pubmed-queries "common medical conditions" "preventive medicine"
```

### Run Evaluation

```bash
# Test query
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "What are the symptoms of anemia?",
      "How is diabetes managed?"
    ]
  }'
```

### Deploy to Production

See [DEPLOYMENT.md](deployment/DEPLOYMENT.md) for detailed deployment instructions.

Quick options:
- **Render**: `git push` and it auto-deploys
- **Railway**: `railway up`
- **Docker**: `docker-compose up`

## 🆘 Troubleshooting

### "Vector store not found"
```bash
# Run ingestion first
python scripts/ingest_data.py --source sample
```

### "OpenAI API key not set"
```bash
# Add to .env file
echo "OPENAI_API_KEY=sk-your-key" >> .env

# Or use open-source model
echo "LLM_PROVIDER=mistral" >> .env
```

### "Port 8000 already in use"
```bash
# Change port in .env
echo "API_PORT=8001" >> .env
```

### "Out of memory" (with Mistral/Qwen)
```bash
# Use OpenAI instead (cloud-based)
echo "LLM_PROVIDER=openai" >> .env
```

### Import errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## 📞 Need Help?

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/MedSource-rag-chatbot/issues)

## 🎉 You're Ready!

You now have a fully functional medical RAG chatbot running locally!

For hackathon submission, you'll need to:
1. ✅ Deploy the API (Render/Railway)
2. ✅ Get the public endpoint URL
3. ✅ Run RAGAS evaluation
4. ✅ Submit: API URL + Documentation

Good luck! 🚀
