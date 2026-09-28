# 🚀 SUPER QUICK START - Skip Model Downloads!

## The Issue You're Facing

When you run the setup, it downloads a 90MB embedding model from HuggingFace. This can be slow (5-20 minutes depending on your internet).

## ✨ SOLUTION: Use OpenAI Instead (No Downloads!)

OpenAI runs in the cloud, so **no model downloads needed**!

### Step 1: Get OpenAI API Key (2 minutes)

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

### Step 2: Setup (30 seconds)

```bash
cd /Users/gugank/MedSource-rag-chatbot

# Create .env file with your key
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-key-here-replace-this
LLM_PROVIDER=openai
EOF

# Activate environment
source venv/bin/activate

# Create sample data (no model download!)
python scripts/ingest_data.py --source sample
```

### Step 3: Start Server (instantly!)

```bash
python -m src.api
```

That's it! Visit http://localhost:8000/docs

---

## Alternative: Wait for Local Model (Free but Slow)

If you want to use Mistral (open-source, free), just wait for the current download to finish.

**Current Progress**: Downloading 90.9MB model...

This takes 5-20 minutes depending on internet speed.

---

## What You Get

### With OpenAI (Recommended for Hackathon)
✅ **No downloads** - Starts instantly  
✅ **Best quality** - GPT-4 responses  
✅ **Fast** - Cloud processing  
❌ **Costs money** - ~$0.01 per query  

### With Mistral/Qwen (Free but Slow Setup)
✅ **Free** - No API costs  
✅ **Private** - Runs locally  
❌ **Slow setup** - 90MB+ downloads  
❌ **Needs memory** - 8GB+ RAM  

---

## Quick Test

Once server is running:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the symptoms of anemia?",
    "top_k": 3
  }'
```

---

## Why Model Downloads?

The RAG system needs:

1. **Embedding Model** (90MB) - Converts text to vectors for semantic search
   - This is what's downloading now: `sentence-transformers/all-MiniLM-L6-v2`
   - Required for FAISS vector database

2. **LLM Model** (7GB+ for Mistral/Qwen) - Generates answers
   - Optional: Can use OpenAI instead (no download)
   - Local models give you privacy but are slow to setup

---

## Recommendation for Hack-A-Cure Hackathon

**Use OpenAI for the hackathon!**

Why?
- ✅ Setup in 2 minutes vs 30 minutes
- ✅ Best quality answers
- ✅ Easy to deploy
- ✅ Cost is minimal (~$1 for entire hackathon)

You can switch to local models later if needed!

---

## Current Download Status

Check progress in your terminal. You'll see:

```
model.safetensors: 100%|████████████| 90.9M/90.9M [XX:XX<00:00, XXkB/s]
```

When it says 100%, continue with the script!

---

## Need Help?

**If download is stuck:**
```bash
# Cancel with Ctrl+C
# Use OpenAI instead (see Step 2 above)
```

**If you want to use Mistral:**
```bash
# Just wait for download to complete
# Then run: python -m src.api
```

---

Good luck with the hackathon! 🚀
