"""
Test script to verify the API matches Hack-A-Cure competition format
"""
import requests
import json

# Test the endpoint format
def test_query_endpoint():
    url = "http://localhost:8000/query"
    
    # Test request in competition format
    payload = {
        "query": "When to give Tdap booster?",
        "top_k": 3
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("🧪 Testing Hack-A-Cure API format...")
    print(f"📤 Request: POST {url}")
    print(f"📝 Payload: {json.dumps(payload, indent=2)}\n")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        print(f"📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS! Response format:")
            print(json.dumps(data, indent=2))
            
            # Verify required fields
            assert "answer" in data, "Missing 'answer' field"
            assert "contexts" in data, "Missing 'contexts' field"
            assert isinstance(data["contexts"], list), "'contexts' must be a list"
            assert all(isinstance(c, str) for c in data["contexts"]), "All contexts must be strings"
            
            print(f"\n✅ Required fields present:")
            print(f"   - answer: {len(data['answer'])} chars")
            print(f"   - contexts: {len(data['contexts'])} items")
            
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    test_query_endpoint()
