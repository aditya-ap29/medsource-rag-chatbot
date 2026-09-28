# 🔧 Troubleshooting Guide

Common issues and solutions for MedSource RAG Chatbot.

---

## 🚨 Installation Issues

### Issue: `pip install` fails

**Error**: `Could not find a version that satisfies the requirement...`

**Solution**:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# If specific package fails, install it separately
pip install faiss-cpu
```

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Make sure you're in the project root
cd MedSource-rag-chatbot

# Run from root directory
python -m src.api

# NOT: python src/api.py
```

---

## 🔐 API Key Issues

### Issue: `OpenAI API key not found`

**Error**: `openai.error.AuthenticationError`

**Solution**:
```bash
# Check .env file exists
ls -la .env

# Add API key
echo "OPENAI_API_KEY=sk-your-actual-key" >> .env

# Verify it's loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"
```

### Issue: `Invalid API key`

**Solution**:
1. Get new key from: https://platform.openai.com/api-keys
2. Make sure no spaces or quotes in .env:
   ```bash
   # WRONG:
   OPENAI_API_KEY="sk-..."
   OPENAI_API_KEY= sk-...
   
   # CORRECT:
   OPENAI_API_KEY=sk-...
   ```

### Issue: `Rate limit exceeded`

**Solution**:
```bash
# Switch to open-source model
echo "LLM_PROVIDER=mistral" >> .env

# Or add delay between requests
# (handled automatically in code)
```

---

## 📦 Vector Store Issues

### Issue: `Vector store not found`

**Error**: `FileNotFoundError: Vector store directory not found`

**Solution**:
```bash
# Run data ingestion first
python scripts/ingest_data.py --source sample

# Verify it was created
ls -la data/vector_store/
```

### Issue: `No documents in vector store`

**Solution**:
```bash
# Check if files exist
ls -la data/vector_store/

# If empty, re-ingest
rm -rf data/vector_store
python scripts/ingest_data.py --source sample
```

### Issue: `FAISS index corrupted`

**Solution**:
```bash
# Delete and rebuild
rm -rf data/vector_store/
python scripts/ingest_data.py --source sample
```

---

## 🤖 LLM Issues

### Issue: `Out of memory` (with Mistral/Qwen)

**Error**: `torch.cuda.OutOfMemoryError` or system freeze

**Solution**:
```bash
# Option 1: Use OpenAI (cloud-based)
echo "LLM_PROVIDER=openai" >> .env
echo "OPENAI_API_KEY=sk-your-key" >> .env

# Option 2: Use smaller model
# Edit src/rag_pipeline.py, line 51:
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# Change to:
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Option 3: Increase system memory
# Close other applications
```

### Issue: `Model download stuck`

**Solution**:
```bash
# Models download from HuggingFace
# Check internet connection

# Or pre-download:
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"

# Check disk space (models are 3-15GB)
df -h
```

### Issue: `Slow generation` (30+ seconds)

**Solution**:
```bash
# Use GPU if available
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Or switch to OpenAI
echo "LLM_PROVIDER=openai" >> .env

# Or reduce max_tokens in .env
echo "MAX_TOKENS=500" >> .env
```

---

## 🌐 API Issues

### Issue: `Connection refused`

**Error**: `requests.exceptions.ConnectionError`

**Solution**:
```bash
# Check if server is running
curl http://localhost:8000/health

# If not, start it
python -m src.api

# Check correct port
curl http://localhost:8000/health
# NOT: http://localhost:80/health
```

### Issue: `Port 8000 already in use`

**Solution**:
```bash
# Option 1: Use different port
echo "API_PORT=8001" >> .env
python -m src.api

# Option 2: Kill process using port 8000
# macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: `502 Bad Gateway` (on deployment)

**Solution**:
```bash
# Check logs
railway logs
# or
render logs

# Common causes:
# 1. Out of memory -> Use OpenAI instead of local LLM
# 2. Startup timeout -> Increase timeout in render.yaml
# 3. Vector store not created -> Run ingestion in build step
```

---

## 📄 Data Ingestion Issues

### Issue: `PubMed returns no results`

**Solution**:
```bash
# Check internet connection
ping eutils.ncbi.nlm.nih.gov

# Try broader query
python scripts/ingest_data.py --source pubmed --pubmed-queries "medicine"

# Set email in .env (required by NCBI)
echo "PUBMED_EMAIL=your@email.com" >> .env
```

### Issue: `PDF parsing fails`

**Error**: `PdfReadError` or empty text

**Solution**:
```bash
# Install additional dependencies
pip install PyPDF2 --upgrade

# If still fails, try:
pip uninstall PyPDF2
pip install pypdf

# Or use text/DOCX format instead
```

### Issue: `No local documents found`

**Solution**:
```bash
# Create directory
mkdir -p data/raw_documents

# Add sample files
cp /path/to/medical.pdf data/raw_documents/

# Or use sample data
python scripts/ingest_data.py --source sample
```

---

## 🧪 Testing Issues

### Issue: `Test script fails`

**Solution**:
```bash
# Make sure server is running in separate terminal
python -m src.api

# In new terminal:
python scripts/test_api.py

# Check API is accessible
curl http://localhost:8000/health
```

### Issue: `Evaluation endpoint returns 0 scores`

**Solution**:
```bash
# RAGAS requires specific dependencies
pip install ragas datasets --upgrade

# May need OpenAI API key for evaluation
echo "OPENAI_API_KEY=sk-your-key" >> .env
```

---

## 🐳 Docker Issues

### Issue: `Docker build fails`

**Solution**:
```bash
# Clear Docker cache
docker system prune -a

# Rebuild
docker-compose build --no-cache

# Check Dockerfile syntax
docker build -t MedSource .
```

### Issue: `Container exits immediately`

**Solution**:
```bash
# Check logs
docker-compose logs

# Run interactively
docker run -it MedSource /bin/bash

# Common issues:
# - Missing .env file
# - Vector store not created
# - Port conflict
```

---

## 🚀 Deployment Issues

### Issue: `Render build timeout`

**Solution**:
```yaml
# In render.yaml, increase timeout
buildCommand: |
  pip install -r requirements.txt
  timeout 600 python scripts/ingest_data.py --source sample

# Or skip ingestion in build, do it manually via shell
```

### Issue: `Railway deployment out of memory`

**Solution**:
```bash
# Use OpenAI instead of local LLM
railway variables set LLM_PROVIDER=openai
railway variables set OPENAI_API_KEY=sk-your-key

# Upgrade plan for more memory
```

### Issue: `Environment variables not loading`

**Solution**:
```bash
# On Render/Railway:
# Don't use quotes in variable values

# WRONG:
OPENAI_API_KEY="sk-..."

# CORRECT:
OPENAI_API_KEY=sk-...
```

---

## 💾 Performance Issues

### Issue: `Slow retrieval` (> 1 second)

**Solution**:
```bash
# Use smaller embedding model
echo "EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2" >> .env

# Reduce top_k
echo "TOP_K_DOCUMENTS=3" >> .env

# For large datasets, use Qdrant instead of FAISS
# See src/vector_store_qdrant.py
```

### Issue: `High memory usage`

**Solution**:
```bash
# Option 1: Use OpenAI (no local model)
echo "LLM_PROVIDER=openai" >> .env

# Option 2: Reduce vector store size
# Edit scripts/ingest_data.py, limit documents

# Option 3: Use quantized models (future improvement)
```

---

## 🔍 Debugging Tips

### Enable Verbose Logging

```python
# Add to src/api.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Component Status

```python
# Test vector store
python -c "from src.vector_store import VectorStore; vs = VectorStore(); vs.load('./data/vector_store'); print(vs.get_stats())"

# Test RAG pipeline
python examples/direct_usage.py

# Test API
curl http://localhost:8000/stats
```

### Verify Dependencies

```bash
# List installed packages
pip list | grep -E "(faiss|sentence-transformers|langchain|ragas)"

# Check Python version
python --version  # Should be 3.8+

# Check CUDA (for GPU)
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📞 Still Having Issues?

1. **Check logs**:
   ```bash
   # Server logs
   python -m src.api 2>&1 | tee logs/api.log
   ```

2. **Minimal test**:
   ```python
   # test_minimal.py
   from src.vector_store import VectorStore
   from src.data_ingestion import MedicalDocument
   
   doc = MedicalDocument(
       content="Test content",
       title="Test",
       source="Test"
   )
   
   vs = VectorStore()
   vs.add_documents([doc])
   results = vs.search("test", top_k=1)
   print(f"Found {len(results)} results")
   ```

3. **GitHub Issues**:
   - Search existing issues
   - Create new issue with:
     - Error message
     - Steps to reproduce
     - Python version
     - OS

4. **Community Support**:
   - Stack Overflow tag: `MedSource-rag`
   - Discord/Slack (if available)

---

## ✅ Quick Diagnostics Checklist

```bash
# Run this to check everything
python << EOF
import sys
print(f"Python: {sys.version}")

try:
    import fastapi
    print("✅ FastAPI installed")
except:
    print("❌ FastAPI missing")

try:
    import faiss
    print("✅ FAISS installed")
except:
    print("❌ FAISS missing")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ Sentence-Transformers installed")
except:
    print("❌ Sentence-Transformers missing")

import os
from pathlib import Path

if Path(".env").exists():
    print("✅ .env file exists")
else:
    print("❌ .env file missing")

if Path("data/vector_store").exists():
    print("✅ Vector store directory exists")
else:
    print("❌ Vector store missing - run ingestion")
EOF
```

---

**Most issues are solved by:**
1. Running ingestion: `python scripts/ingest_data.py --source sample`
2. Setting API key: `echo "OPENAI_API_KEY=sk-..." >> .env`
3. Using correct Python version: `python --version` (3.8+)

Good luck! 🚀
