# 📝 Next Steps for Hack-A-Cure Deployment

## ✅ Completed
- [x] Python 3.11.14 environment setup
- [x] 267,211 medical chunks ingested from 9 PDFs
- [x] Google Gemini API integration working
- [x] Hackathon-compliant API format (`query`+`top_k` → `answer`+`contexts`)
- [x] Successful API testing with medical queries
- [x] Git repository initialized (35 files, 5,979 lines)
- [x] Vector store backed up (400MB compressed)

## 🚀 Next Steps (30 minutes total)

### 1. Upload Vector Store to Cloud (5 min)
**Choose ONE option:**

**Option A: Google Drive (Easiest)**
```bash
# 1. Open Google Drive in browser
# 2. Upload vector_store_backup.tar.gz (400MB)
# 3. Right-click file → Share → "Anyone with link can view"
# 4. Copy the shareable link
# 5. Convert format:
#    FROM: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
#    TO:   https://drive.google.com/uc?export=download&id=FILE_ID
```

**Option B: Dropbox**
```bash
# 1. Upload vector_store_backup.tar.gz to Dropbox
# 2. Share → Create link
# 3. Change ?dl=0 to ?dl=1 at end of URL
```

**Save this URL - you'll need it in Step 3!**

---

### 2. Push Code to GitHub (5 min)

```bash
cd /Users/gugank/MedSource-rag-chatbot

# Create a new repository on GitHub:
# 1. Go to https://github.com/new
# 2. Name: MedSource-rag-chatbot (or any name)
# 3. Description: "RAG system for medical Q&A using Gemini + FAISS"
# 4. Public repository (required for free Render deployment)
# 5. Click "Create repository"

# Then push your code:
git remote add origin https://github.com/YOUR_USERNAME/MedSource-rag-chatbot.git
git branch -M main
git push -u origin main

# Verify: You should see 35 files on GitHub
```

---

### 3. Deploy on Render.com (15 min)

**Step 3.1: Create Account**
1. Go to https://render.com
2. Click "Get Started for Free"
3. Sign up with GitHub (easiest - auto-connects repos)

**Step 3.2: Create Web Service**
1. Click "New +" → "Web Service"
2. Click "Connect" next to your `MedSource-rag-chatbot` repository
3. Render will auto-detect `render.yaml` ✅

**Step 3.3: Add Vector Store URL**
1. In the "Environment" section, click "+ Add Environment Variable"
2. Key: `VECTOR_STORE_URL`
3. Value: Paste the download URL from Step 1
4. Click "Add"

**Step 3.4: Deploy!**
1. Click "Create Web Service"
2. Wait 5-10 minutes (watch the logs)
3. Deployment steps:
   - Installing dependencies (pip install)
   - Downloading vector store (400MB)
   - Extracting 267,211 chunks
   - Starting API server
4. ✅ Done! Your URL will be: `https://MedSource-rag-api.onrender.com`

---

### 4. Test Deployed API (2 min)

```bash
# Replace YOUR-APP-NAME with your Render app name
curl -X POST https://YOUR-APP-NAME.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "When to give Tdap booster?",
    "top_k": 3
  }'

# Should return:
# {
#   "answer": "Based on the CDC recommendations, Tdap booster should be given every 10 years for adults over 19... [DOC_1][DOC_2][DOC_3]",
#   "contexts": [
#     "...relevant medical text from PDFs...",
#     "...more context...",
#     "...more context..."
#   ]
# }
```

---

### 5. Submit to Hack-A-Cure (3 min)

1. Go to hackathon submission page
2. Enter your API URL: `https://YOUR-APP-NAME.onrender.com/query`
3. Click "Submit"
4. Platform will test with sample queries
5. Check scores:
   - Answer Relevancy: 30%
   - Answer Correctness: 30%
   - Context Relevance: 25%
   - Faithfulness: 15%

---

## 🐛 Troubleshooting

### Render Build Fails
- Check logs for errors
- Common issue: VECTOR_STORE_URL not set
- Fix: Add environment variable in Render dashboard

### API Returns 500 Error
- Check if vector store downloaded successfully
- Look for "Vector store ready!" in logs
- Verify GEMINI_API_KEY is set

### Slow First Request
- Render free tier sleeps after 15min of inactivity
- First request after sleep takes 30-60 seconds
- Subsequent requests are fast (1-2 seconds)

### Low Hackathon Scores
- Adjust RETRIEVAL_CONFIDENCE_THRESHOLD (currently 0.3)
- Try 0.2 for more results, 0.4 for higher quality
- Change in Render environment variables
- Redeploy to apply changes

---

## 📊 Current Configuration

| Setting | Value |
|---------|-------|
| Vector Store Size | 551MB (400MB compressed) |
| Total Chunks | 267,211 |
| PDFs Ingested | 9 medical textbooks |
| LLM Provider | Google Gemini (FREE) |
| Model | gemini-2.0-flash |
| Embedding Model | all-MiniLM-L6-v2 |
| Retrieval Threshold | 0.3 |
| Response Time | ~2 seconds |

---

## 🎯 Success Criteria

✅ API endpoint: `https://YOUR-APP.onrender.com/query`
✅ Accepts: `{"query": string, "top_k": number}`
✅ Returns: `{"answer": string, "contexts": [strings]}`
✅ Responds in < 5 seconds
✅ Medical accuracy with citations

**You have 10 submission attempts to optimize scores!**

Good luck! 🚀
