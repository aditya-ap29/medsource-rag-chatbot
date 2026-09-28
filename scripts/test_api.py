"""
Test script to validate the RAG pipeline
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
from typing import List, Dict


BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health endpoint"""
    print("\n" + "="*60)
    print("🏥 Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"   App: {data['app_name']}")
        print(f"   Version: {data['version']}")
        print(f"   Vector Store Loaded: {data['vector_store_loaded']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False


def test_query(question: str):
    """Test a single query"""
    print("\n" + "="*60)
    print(f"❓ Testing Query: {question}")
    print("="*60)
    
    payload = {
        "question": question,
        "top_k": 5,
        "include_sources": True
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"\n📝 Answer:")
        print(f"{data['answer']}\n")
        
        print(f"📊 Metrics:")
        print(f"   Confidence: {data['confidence']:.2f}")
        print(f"   Retrieval Time: {data['retrieval_time_ms']:.2f}ms")
        print(f"   Generation Time: {data['generation_time_ms']:.2f}ms")
        print(f"   Total Time: {data['total_time_ms']:.2f}ms")
        
        print(f"\n📚 Sources ({len(data['sources'])}):")
        for source in data['sources']:
            print(f"   [{source['doc_id']}] {source['title']}")
            print(f"       Score: {source['relevance_score']:.3f}")
            if source.get('url'):
                print(f"       URL: {source['url']}")
        
        return True
    else:
        print(f"❌ Query failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False


def test_stats():
    """Test the stats endpoint"""
    print("\n" + "="*60)
    print("📊 Testing Statistics")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/stats")
    
    if response.status_code == 200:
        data = response.json()
        
        print("\n📚 Vector Store:")
        vs = data['vector_store']
        print(f"   Total Chunks: {vs['total_chunks']}")
        print(f"   Unique Documents: {vs['unique_documents']}")
        print(f"   Dimension: {vs['dimension']}")
        print(f"   Chunk Size: {vs['chunk_size']}")
        
        print("\n🤖 RAG Configuration:")
        rag = data['rag_config']
        print(f"   LLM Model: {rag['llm_model']}")
        print(f"   LLM Provider: {rag['llm_provider']}")
        print(f"   Top K: {rag['top_k']}")
        print(f"   Confidence Threshold: {rag['confidence_threshold']}")
        print(f"   Temperature: {rag['temperature']}")
        
        return True
    else:
        print(f"❌ Stats failed: {response.status_code}")
        return False


def run_test_suite():
    """Run complete test suite"""
    print("\n" + "="*60)
    print("🧪 MedSource RAG Chatbot - Test Suite")
    print("="*60)
    
    # Test health
    if not test_health_check():
        print("\n❌ Server is not healthy. Exiting tests.")
        return
    
    # Test stats
    test_stats()
    
    # Test queries
    test_questions = [
        "What are the symptoms of vitamin D deficiency?",
        "How is anemia treated?",
        "What causes migraine headaches?",
        "What are the risk factors for diabetes?",
        "Can you diagnose me with a condition?",  # Should refuse
        "What is the capital of France?",  # Should admit insufficient info
    ]
    
    passed = 0
    failed = 0
    
    for question in test_questions:
        if test_query(question):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("📋 Test Summary")
    print("="*60)
    print(f"✅ Passed: {passed}/{len(test_questions)}")
    print(f"❌ Failed: {failed}/{len(test_questions)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        run_test_suite()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("   Make sure the server is running at http://localhost:8000")
        print("   Start it with: python -m src.api")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
