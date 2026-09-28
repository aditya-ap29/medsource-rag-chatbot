"""
Example usage of MedSource RAG Chatbot
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json


def main():
    """Demonstrate MedSource usage"""
    
    # API endpoint
    api_url = "http://localhost:8000/query"
    
    # Medical questions to ask
    questions = [
        "What are the symptoms of anemia?",
        "How is vitamin D deficiency treated?",
        "What are the risk factors for hypertension?",
    ]
    
    print("🏥 MedSource - Medical RAG Chatbot Demo")
    print("="*60 + "\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Query {i}] {question}")
        print("-" * 60)
        
        # Make API request
        response = requests.post(
            api_url,
            json={
                "question": question,
                "top_k": 3,
                "include_sources": True
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Print answer
            print(f"\n{data['answer']}\n")
            
            # Print metadata
            print(f"💡 Confidence: {data['confidence']:.2%}")
            print(f"⏱️  Response Time: {data['total_time_ms']:.0f}ms")
            print(f"📚 Sources Used: {len(data['sources'])}")
            
            # Print sources
            print("\n📖 References:")
            for source in data['sources']:
                print(f"   • {source['title']} ({source.get('year', 'N/A')})")
                print(f"     Relevance: {source['relevance_score']:.2%}")
                if source.get('url'):
                    print(f"     {source['url']}")
        else:
            print(f"❌ Error: {response.status_code}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to MedSource API")
        print("Make sure the server is running:")
        print("  python -m src.api")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
