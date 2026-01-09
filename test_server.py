"""
Test script for the Maize Disease Detection API
"""

import requests
import json
from pathlib import Path

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    try:
        response = requests.get("http://localhost:8000/model/info")
        print(f"Model Info Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Model info failed: {e}")
        return False

def test_diseases_endpoint():
    """Test the diseases endpoint"""
    try:
        response = requests.get("http://localhost:8000/diseases")
        print(f"Diseases Status: {response.status_code}")
        data = response.json()
        print(f"Total diseases: {data.get('total_diseases', 0)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Diseases endpoint failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Maize Disease Detection API")
    print("=" * 50)
    
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("Model Info", test_model_info),
        ("Diseases Endpoint", test_diseases_endpoint),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The server is working correctly.")
    else:
        print("❌ Some tests failed. Check the server configuration.")

if __name__ == "__main__":
    main()