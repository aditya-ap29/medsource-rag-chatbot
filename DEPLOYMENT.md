# 🚀 Render.com Deployment Guide for Hack-A-Cure

## ⚠️ IMPORTANT: Vector Store Setup

**The vector store (551MB) is too large for GitHub.** You have 2 options:

### Option 1: Re-ingest on Render (Recommended for Free Tier)
- Upload PDFs to Render
- Run ingestion script during deployment
- Pros: No external dependencies
- Cons: Takes 30-45 minutes on first deployment

### Option 2: Use Cloud Storage (Faster)
- Upload `vector_store_backup.tar.gz` to Google Drive/Dropbox
- Download during Render build process
- Pros: Faster deployment (5 minutes)
- Cons: Needs download URL

**For this hackathon, we'll use Option 2 (faster).**

---

## Quick Deployment Steps (10 minutes)

### Step 1: Push Code to GitHub

```bash
cd /Users/gugank/MedSource-rag-chatbot

# Commit and push (vector_store is excluded via .gitignore)
git add .
git commit -m "Hack-A-Cure RAG submission - Gemini + FAISS"

# Create GitHub repo first at github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/MedSource-rag-chatbot.git
git push -u origin main
```

### Step 2: Upload Vector Store to Cloud

**Option A: Google Drive**
1. Upload `vector_store_backup.tar.gz` to Google Drive
2. Right-click → Share → Anyone with link can view
3. Get shareable link (looks like `https://drive.google.com/file/d/FILE_ID/view`)
4. Convert to direct download: `https://drive.google.com/uc?export=download&id=FILE_ID`

**Option B: Dropbox**
1. Upload `vector_store_backup.tar.gz` to Dropbox
2. Share → Create link
3. Change `?dl=0` to `?dl=1` at end of URL

### Step 3: Create Render Account & Deploy

1. **Create Render Account**
   - Go to https://render.com
   - Sign up with GitHub (easiest)

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml`

3. **Add Vector Store URL**
   - In Render dashboard, add environment variable:
   - Key: `VECTOR_STORE_URL`
   - Value: Your download URL from Step 2

4. **Deploy!**
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment
   - Your API will be at: `https://MedSource-rag-api.onrender.com`

---

## Alternative: Manual Configuration
   ```
   GEMINI_API_KEY=AIzaSyCEcVyJeeQztcxbG-ekdheNh_4tMN9z0P0
   LLM_PROVIDER=gemini
   LLM_MODEL=gemini-2.0-flash
   RETRIEVAL_CONFIDENCE_THRESHOLD=0.3
   OMP_NUM_THREADS=1
   OPENBLAS_NUM_THREADS=1
   VECLIB_MAXIMUM_THREADS=1
   ```

5. **Deploy!**
   - Click "Create Web Service"
   - Wait 5-10 minutes

## 📋 After Deployment

### Test Your Deployed API
```bash
curl -X POST https://YOUR-APP.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"query": "When to give Tdap booster?", "top_k": 3}'
```

### Submit to Hack-A-Cure
1. Go to hackathon submission page
2. Enter: `https://YOUR-APP.onrender.com/query`
3. Test with their evaluator
4. You have 10 submissions to optimize!

## ⚠️ Important Notes

1. **Vector Store**: Your 267K chunks (1.6GB) are already in `data/vector_store/` - they will deploy with your code

2. **Free Tier Limitations**:
   - Sleeps after 15 min inactivity
   - Takes ~30 seconds to wake up
   - 750 hours/month free

3. **Keep It Awake** (optional):
   - Use cron-job.org to ping your `/health` endpoint every 10 minutes
   - Or just let it sleep - it wakes up fast enough for hackathon evaluation

4. **Monitoring**:
   - Check logs in Render dashboard
   - Monitor at: https://dashboard.render.com

## 🔧 Troubleshooting

**Build fails?**
- Check requirements.txt includes all packages
- Verify Python version is 3.11

**App crashes?**
- Check Render logs
- Verify GEMINI_API_KEY is set
- Ensure vector_store files are committed to git

**Slow responses?**
- First request after sleep takes ~30 sec
- Subsequent requests are fast (~2 sec)

## 📊 Current Configuration

- **Vector Store**: 267,211 chunks from 9 medical PDFs
- **Model**: Google Gemini 2.0-Flash (FREE)
- **Response Format**: Hack-A-Cure compliant (`{answer, contexts}`)
- **Average Response Time**: ~2 seconds (when warm)
- **Retrieval Threshold**: 0.3 (balanced for medical content)

Good luck with your hackathon! 🎉
