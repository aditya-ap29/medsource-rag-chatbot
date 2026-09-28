# 🚀 Render.com Deployment - FINAL STEPS

## ✅ Completed
- [x] Code pushed to GitHub: https://github.com/Gaggs-daggs/MedSource-rag-chatbot
- [x] 35 files successfully uploaded
- [x] render.yaml configuration ready

---

## 📋 Next: Deploy on Render (10 minutes)

### Step 1: Create Render Account (2 min)

1. Go to: **https://render.com**
2. Click **"Get Started for Free"**
3. **Sign up with GitHub** (easiest option)
4. Click **"Authorize Render"** when prompted

### Step 2: Create Web Service (3 min)

1. In Render dashboard, click **"New +"** (top right)
2. Select **"Web Service"**
3. You'll see your GitHub repositories
4. Find **"MedSource-rag-chatbot"** and click **"Connect"**

### Step 3: Configure (Auto-detected!) (2 min)

Render will auto-detect your `render.yaml` file and pre-fill everything!

**Verify these settings:**
- ✅ Name: `MedSource-rag-api`
- ✅ Runtime: `Python 3`
- ✅ Build Command: `pip install -r requirements.txt && bash scripts/download_vector_store.sh || echo "..."`
- ✅ Start Command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
- ✅ Plan: `Free`

**Environment Variables** (already set in render.yaml):
- ✅ GEMINI_API_KEY
- ✅ LLM_MODEL
- ✅ All other configs

### Step 4: Deploy! (3 min)

1. Click **"Create Web Service"**
2. Wait for deployment (watch the logs)
   - Installing Python packages... (2-3 min)
   - Starting server...
   - ✅ Deployment successful!

3. **Your public URL will be:**
   ```
   https://MedSource-rag-api.onrender.com
   ```

---

## ⚠️ IMPORTANT: Vector Store Issue

**The deployment will work, BUT the API will return errors** because the vector store (551MB) is not included in the GitHub repo.

### Solution: Upload Vector Store After Deployment

**Option 1: Use Render Shell (Recommended)**

Once deployed:
1. In Render dashboard, click your service
2. Click **"Shell"** tab
3. Run these commands:
   ```bash
   cd /opt/render/project/src
   # Download from your Google Drive link
   curl -L "YOUR_GOOGLE_DRIVE_LINK" -o /tmp/vector_store.tar.gz
   tar -xzf /tmp/vector_store.tar.gz -C /opt/render/project/
   rm /tmp/vector_store.tar.gz
   ```

**Option 2: Add to GitHub with Git LFS**

Or use Git Large File Storage to include the vector store in the repo.

**Option 3: Re-ingest on Render**

Upload PDFs and run the ingestion script on Render (takes 30-45 minutes).

---

## 🧪 Testing Your Deployed API

Once deployed (even without vector store), test the health endpoint:

```bash
curl https://MedSource-rag-api.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "app_name": "MedSource - Medical RAG Chatbot",
  "version": "1.0.0",
  "vector_store_loaded": false
}
```

To test queries (will fail without vector store):
```bash
curl -X POST https://MedSource-rag-api.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "When to give Tdap booster?",
    "top_k": 3
  }'
```

---

## 📝 For Hackathon Submission

**Your API Endpoint URL:**
```
https://MedSource-rag-api.onrender.com/query
```

**Copy this URL and paste it into the Hack-A-Cure submission form!**

---

## 🐛 Troubleshooting

### Deployment Fails
- Check Render logs for errors
- Verify all environment variables are set
- Ensure requirements.txt has all dependencies

### API Returns 500 Error
- Vector store not loaded (expected if not uploaded)
- Check logs in Render dashboard
- Verify GEMINI_API_KEY is correct

### Slow First Request
- Render free tier "sleeps" after 15min inactivity
- First request takes 30-60 seconds to "wake up"
- Subsequent requests are fast (1-2 seconds)

---

## 🎯 Current Status

✅ Code on GitHub
✅ render.yaml configured  
⏳ Need to deploy on Render
⏳ Need to upload vector store
⏳ Need to test API
⏳ Need to submit to hackathon

**Start with Step 1 above!** 🚀
