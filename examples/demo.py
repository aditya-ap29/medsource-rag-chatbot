"""
Minimal demo — query the MedSource FastAPI backend via HTTP.

Prerequisites:
  1. Server running:  python -m src.api
  2. .env configured: GEMINI_API_KEY and (optionally) CHATBOT_API_KEY

Usage:
  python examples/demo.py
"""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env so CHATBOT_API_KEY is available
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

API_URL = "http://localhost:8000"
API_KEY = os.getenv("CHATBOT_API_KEY", "")   # blank = auth disabled

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["X-API-Key"] = API_KEY

QUESTIONS = [
    "What are the symptoms of anaemia?",
    "How is type 2 diabetes managed?",
    "What are the risk factors for hypertension?",
]


def query(question: str, top_k: int = 3) -> dict:
    resp = requests.post(
        f"{API_URL}/query",
        json={"query": question, "top_k": top_k},
        headers=HEADERS,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    # Health check
    health = requests.get(f"{API_URL}/health", timeout=10).json()
    print(f"Service: {health['status']} | KB loaded: {health['vector_store_loaded']}\n")

    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}] {question}")
        print("-" * 60)
        data = query(question)
        print(data["answer"][:500])
        print(f"\nConfidence : {data.get('confidence', 0):.2%}")
        print(f"Sources    : {len(data.get('sources') or [])}")
        print(f"Total time : {data.get('total_time_ms', 0):.0f} ms")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Could not connect. Start the server first:  python -m src.api")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e.response.status_code} — {e.response.text}")
    except KeyboardInterrupt:
        pass
