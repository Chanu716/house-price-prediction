"""
Simple test script to verify the API is working
"""
import time
import json

# Wait for server to start
print("Waiting for server to start...")
time.sleep(3)

try:
    import requests
    
    print("🧪 Testing HousePriceAI API")
    print("=" * 40)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        health_response = requests.get("http://localhost:5000/health", timeout=5)
        print(f"   Status: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"   Response: {health_response.json()}")
            print("   ✅ Health check passed!")
        else:
            print(f"   ❌ Health check failed: {health_response.text}")
    except Exception as e:
        print(f"   ❌ Health check error: {str(e)}")
    
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
    
    try:
        pred_response = requests.post("http://localhost:5000/predict", 
                                      json=sample_data, 
                                      timeout=10)
        print(f"   Status: {pred_response.status_code}")
        if pred_response.status_code == 200:
            result = pred_response.json()
            print(f"   ✅ Predicted Price: ${result.get('prediction', 0):,.2f}")
            if 'confidence_range' in result:
                cr = result['confidence_range']
                print(f"   Range: ${cr['min']:,.2f} - ${cr['max']:,.2f}")
            print("   🎉 Prediction successful!")
        else:
            print(f"   ❌ Prediction failed: {pred_response.text}")
    except Exception as e:
        print(f"   ❌ Prediction error: {str(e)}")
    
except ImportError:
    print("❌ Requests library not available. Please install it: pip install requests")
except Exception as e:
    print(f"❌ Unexpected error: {str(e)}")

print("\n✅ Test completed!")