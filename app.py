"""
Flask backend for House Price Prediction
Using simplified prediction algorithm to avoid feature alignment issues
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

def simple_predict(input_data):
    """
    Simplified prediction algorithm that uses key features to estimate house price
    This bypasses the complex feature alignment issues with the ML model
    """
    try:
        # Extract key features
        overall_qual = float(input_data.get('OverallQual', 5))
        year_built = float(input_data.get('YearBuilt', 2000))
        gr_liv_area = float(input_data.get('GrLivArea', 1500))
        bedrooms = float(input_data.get('BedroomAbvGr', 3))
        full_bath = float(input_data.get('FullBath', 2))
        total_rooms = float(input_data.get('TotRmsAbvGrd', 6))
        lot_area = float(input_data.get('LotArea', 8000))
        garage_cars = float(input_data.get('GarageCars', 0))
        fireplaces = float(input_data.get('Fireplaces', 0))
        neighborhood = input_data.get('Neighborhood', 'NAmes')
        house_style = input_data.get('HouseStyle', '1Story')
        central_air = input_data.get('CentralAir', 'Y')
        
        # Base price calculation using key factors
        # This is based on typical housing market relationships
        
        # Base price from living area (roughly $100-150 per sq ft depending on quality)
        price_per_sqft = 80 + (overall_qual * 10)
        base_price = gr_liv_area * price_per_sqft
        
        # Age adjustment (newer homes worth more)
        current_year = 2024
        age = current_year - year_built
        if age < 5:
            age_multiplier = 1.15
        elif age < 10:
            age_multiplier = 1.10
        elif age < 20:
            age_multiplier = 1.05
        elif age < 40:
            age_multiplier = 1.0
            
            # Initialize and fit scaler
            scaler = StandardScaler()
            X[num_features] = scaler.fit_transform(X[num_features])
            
            # Store feature columns for later use
            feature_columns = X.columns.tolist()
            
            print(f"✅ Preprocessing components initialized with {len(feature_columns)} features")
            return True
        else:
            print("❌ Training data not found")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        traceback.print_exc()
        return False

def preprocess_input(input_data):
    """Preprocess input data to match training format"""
    try:
        # Create a DataFrame with the input data
        df = pd.DataFrame([input_data])
        
        # Add default values for missing features that the model expects
        # These are common defaults based on the dataset
        defaults = {
            'Id': 1,
            'MSSubClass': 60,  # 2-Story 1946 & newer
            'MSZoning': 'RL',  # Residential Low Density
            'LotFrontage': 70,  # Median value
            'Street': 'Pave',
            'Alley': 'NA',
            'LotShape': 'Reg',  # Regular
            'LandContour': 'Lvl',  # Level
            'Utilities': 'AllPub',
            'LotConfig': 'Inside',
            'LandSlope': 'Gtl',  # Gentle slope
            'Condition1': 'Norm',  # Normal
            'Condition2': 'Norm',
            'BldgType': '1Fam',  # Single-family
            'OverallCond': 5,  # Average condition
            'YearRemodAdd': input_data.get('YearBuilt', 2000),
            'RoofStyle': 'Gable',
            'RoofMatl': 'CompShg',  # Composite Shingle
            'Exterior1st': 'VinylSd',
            'Exterior2nd': 'VinylSd',
            'MasVnrType': 'None',
            'MasVnrArea': 0,
            'ExterQual': 'TA',  # Typical/Average
            'ExterCond': 'TA',
            'Foundation': 'PConc',  # Concrete
            'BsmtQual': 'TA',
            'BsmtCond': 'TA',
            'BsmtExposure': 'No',
            'BsmtFinType1': 'Unf',  # Unfinished
            'BsmtFinSF1': 0,
            'BsmtFinType2': 'Unf',
            'BsmtFinSF2': 0,
            'BsmtUnfSF': input_data.get('GrLivArea', 1500) // 2,
            'TotalBsmtSF': input_data.get('GrLivArea', 1500) // 2,
            'Heating': 'GasA',  # Gas forced warm air furnace
            'HeatingQC': 'Ex',  # Excellent
            'Electrical': 'SBrkr',  # Standard Circuit Breakers
            '1stFlrSF': input_data.get('GrLivArea', 1500) // 2,
            '2ndFlrSF': input_data.get('GrLivArea', 1500) // 2,
            'LowQualFinSF': 0,
            'BsmtFullBath': 0,
            'BsmtHalfBath': 0,
            'HalfBath': 0,
            'KitchenAbvGr': 1,
            'KitchenQual': 'TA',
            'Functional': 'Typ',  # Typical
            'FireplaceQu': 'NA' if input_data.get('Fireplaces', 0) == 0 else 'TA',
            'GarageType': 'Attchd' if input_data.get('GarageCars', 0) > 0 else 'NA',
            'GarageYrBlt': input_data.get('YearBuilt', 2000),
            'GarageFinish': 'RFn' if input_data.get('GarageCars', 0) > 0 else 'NA',
            'GarageArea': input_data.get('GarageCars', 0) * 250,  # Estimate 250 sq ft per car
            'GarageQual': 'TA' if input_data.get('GarageCars', 0) > 0 else 'NA',
            'GarageCond': 'TA' if input_data.get('GarageCars', 0) > 0 else 'NA',
            'PavedDrive': 'Y',  # Yes
            'WoodDeckSF': 0,
            'OpenPorchSF': 0,
            'EnclosedPorch': 0,
            '3SsnPorch': 0,
            'ScreenPorch': 0,
            'PoolArea': 0,
            'PoolQC': 'NA',
            'Fence': 'NA',
            'MiscFeature': 'NA',
            'MiscVal': 0,
            'MoSold': 6,  # June (median)
            'YrSold': 2023,
            'SaleType': 'WD',  # Warranty Deed
            'SaleCondition': 'Normal'
        }
        
        # Add defaults for missing values
        for key, value in defaults.items():
            if key not in df.columns:
                df[key] = value
        
        # Apply the same preprocessing as training
        y_dummy = pd.Series([0])  # Dummy target
        X = df.copy()
        
        # Get numerical and categorical features
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X.select_dtypes(include=['object']).columns
        
        # Fill missing values
        for col in num_features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
        
        for col in cat_features:
            if col in X.columns:
                mode_val = X[col].mode()
                X[col] = X[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')
        
        # One-hot encode categoricals
        X = pd.get_dummies(X, columns=cat_features, drop_first=True)
        
        # Scale numerical features
        if scaler is not None:
            # Only scale numerical features that exist in both training and current data
            num_cols_to_scale = [col for col in num_features if col in X.columns]
            if num_cols_to_scale:
                X[num_cols_to_scale] = scaler.transform(X[num_cols_to_scale])
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Select only the features that were in training
        X = X[feature_columns]
        
        return X
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        traceback.print_exc()
        raise e

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Get input data from request
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate required fields
        required_fields = ['OverallQual', 'YearBuilt', 'GrLivArea', 'BedroomAbvGr', 
                          'FullBath', 'TotRmsAbvGrd', 'LotArea']
        for field in required_fields:
            if field not in input_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Preprocess the input data
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Ensure prediction is positive and reasonable
        prediction = max(prediction, 50000)  # Minimum reasonable price
        
        # Calculate confidence range (rough estimate based on typical model uncertainty)
        confidence_margin = prediction * 0.15  # ±15% confidence range
        confidence_range = {
            'min': max(prediction - confidence_margin, 30000),
            'max': prediction + confidence_margin
        }
        
        response = {
            'prediction': float(prediction),
            'confidence_range': confidence_range,
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_count': len(feature_columns) if feature_columns else 0
    }
    return jsonify(status)

if __name__ == '__main__':
    print("🏠 HousePriceAI Backend Starting...")
    
    # Load model and preprocessing components
    if load_model_and_preprocessor():
        print("🚀 Server ready! Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please run the Jupyter notebook first to train the model.")
        print("   1. Open HousePricePrediction.ipynb")
        print("   2. Run all cells to train and save the model")
        print("   3. Then restart this Flask app")