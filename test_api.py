"""
Test script for the HousePriceAI API
"""

import requests
import json

def test_api():
    """Test the prediction API with sample data"""
    
    # API endpoint
    url = "http://localhost:5000/predict"
    
    # Sample house data
    sample_house = {
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
    
    print("🏠 Testing HousePriceAI API")
    print("=" * 40)
    print("📤 Sending request with sample house data:")
    print(json.dumps(sample_house, indent=2))
    
    try:
        # Make request
        response = requests.post(url, json=sample_house)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ API Response:")
            print(f"   Predicted Price: ${result['prediction']:,.2f}")
            
            if 'confidence_range' in result:
                print(f"   Confidence Range: ${result['confidence_range']['min']:,.2f} - ${result['confidence_range']['max']:,.2f}")
            
            print(f"   Status: {result['status']}")
            print("\n🎉 API test successful!")
            return True
            
        else:
            print(f"\n❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Could not connect to the API")
        print("   Make sure the Flask app is running on port 5000")
        return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

def test_health_endpoint():
    """Test the health check endpoint"""
    
    url = "http://localhost:5000/health"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            health_info = response.json()
            print("\n🏥 Health Check:")
            print(f"   Status: {health_info['status']}")
            print(f"   Model Loaded: {health_info['model_loaded']}")
            print(f"   Scaler Loaded: {health_info['scaler_loaded']}")
            print(f"   Features Count: {health_info['features_count']}")
            return True
        else:
            print(f"\n❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"\n❌ Health check error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Running API Tests...")
    
    # Test health endpoint
    if test_health_endpoint():
        # Test prediction endpoint
        test_api()
    else:
        print("❌ Health check failed. Cannot proceed with prediction test.")