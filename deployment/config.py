"""
Deployment configurations and instructions
"""

# =============================================================================
# DEPLOYMENT OPTIONS
# =============================================================================

# Option 1: Railway
# -----------------
# 1. Install Railway CLI: npm install -g @railway/cli
# 2. Login: railway login
# 3. Initialize: railway init
# 4. Add environment variables:
#    railway variables set OPENAI_API_KEY=your_key
#    railway variables set LLM_PROVIDER=openai
# 5. Deploy: railway up

# Option 2: Render
# ----------------
# Create render.yaml (see below) and push to GitHub
# Render will auto-deploy

# Option 3: Heroku
# ----------------
# heroku create MedSource-api
# heroku config:set OPENAI_API_KEY=your_key
# git push heroku main

# Option 4: Google Cloud Run
# ---------------------------
# gcloud run deploy MedSource-api \
#   --source . \
#   --platform managed \
#   --region us-central1 \
#   --set-env-vars OPENAI_API_KEY=your_key

# =============================================================================
# PRODUCTION CONFIGURATION
# =============================================================================

# Gunicorn configuration (gunicorn.conf.py)
"""
import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
"""

# =============================================================================
# DOCKER CONFIGURATION
# =============================================================================

DOCKERFILE = """
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/raw_documents data/vector_store data/processed

# Run ingestion (with sample data)
RUN python scripts/ingest_data.py --source sample

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["python", "-m", "src.api"]
"""

# Docker Compose for local development
DOCKER_COMPOSE = """
version: '3.8'

services:
  MedSource:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_PROVIDER=openai
    volumes:
      - ./data:/app/data
    restart: unless-stopped
"""

# =============================================================================
# RENDER CONFIGURATION
# =============================================================================

RENDER_YAML = """
services:
  - type: web
    name: MedSource-api
    env: python
    region: oregon
    plan: starter
    buildCommand: |
      pip install -r requirements.txt
      python scripts/ingest_data.py --source sample
    startCommand: python -m src.api
    envVars:
      - key: OPENAI_API_KEY
        sync: false
      - key: LLM_PROVIDER
        value: openai
      - key: API_HOST
        value: 0.0.0.0
      - key: API_PORT
        value: 8000
    healthCheckPath: /health
"""

# =============================================================================
# NGINX CONFIGURATION (for reverse proxy)
# =============================================================================

NGINX_CONF = """
server {
    listen 80;
    server_name MedSource.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
"""

# =============================================================================
# SYSTEMD SERVICE (for Linux servers)
# =============================================================================

SYSTEMD_SERVICE = """
[Unit]
Description=MedSource RAG Chatbot API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/MedSource-rag-chatbot
Environment="PATH=/home/ubuntu/MedSource-rag-chatbot/venv/bin"
ExecStart=/home/ubuntu/MedSource-rag-chatbot/venv/bin/python -m src.api
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

# =============================================================================
# PERFORMANCE OPTIMIZATION
# =============================================================================

# 1. Use smaller embedding model for faster retrieval
#    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  (384 dim)
#    or sentence-transformers/multi-qa-MiniLM-L6-cos-v1 (384 dim)

# 2. Optimize FAISS index
#    For large datasets (>100k docs), use IVF index:
#    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# 3. Cache embeddings
#    Use Redis or local cache for frequently asked questions

# 4. Load balancing
#    Use multiple workers with Gunicorn

# 5. GPU acceleration
#    For open-source LLMs (Mistral, Qwen), use CUDA:
#    pip install torch --index-url https://download.pytorch.org/whl/cu118

print("Deployment configurations defined. See comments for details.")
