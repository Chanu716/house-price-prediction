"""
Quick test of the API
"""
import requests
import json
import time

# Wait a moment for the server to be ready
time.sleep(2)

print("🧪 Testing HousePriceAI API")
print("=" * 40)

try:
    # Test health endpoint
    print("1. Testing health endpoint...")
    health_response = requests.get("http://localhost:5000/health", timeout=5)
    print(f"   Status: {health_response.status_code}")
    if health_response.status_code == 200:
        print(f"   Response: {health_response.json()}")
    
    # Test prediction endpoint
    print("\n2. Testing prediction endpoint...")
    sample_data = {
        "OverallQual": 7,
        "YearBuilt": 2003,
        "GrLivArea": 1710,
        "BedroomAbvGr": 3,
        "FullBath": 2,
        "TotRmsAbvGrd": 8,
        "LotArea": 8450,
        "Neighborhood": "CollgCr",
        "HouseStyle": "2Story",
        "GarageCars": 2,
        "Fireplaces": 0,
        "CentralAir": "Y"
    }
    
    pred_response = requests.post("http://localhost:5000/predict", 
                                  json=sample_data, 
                                  timeout=10)
    print(f"   Status: {pred_response.status_code}")
    if pred_response.status_code == 200:
        result = pred_response.json()
        print(f"   Predicted Price: ${result.get('prediction', 0):,.2f}")
        if 'confidence_range' in result:
            cr = result['confidence_range']
            print(f"   Range: ${cr['min']:,.2f} - ${cr['max']:,.2f}")
    else:
        print(f"   Error: {pred_response.text}")
    
    print("\n🎉 API tests completed!")
    
except requests.exceptions.ConnectionError:
    print("❌ Connection error - Flask app may not be running")
except Exception as e:
    print(f"❌ Error: {str(e)}")