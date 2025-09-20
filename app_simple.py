from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
import traceback

app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessing components
model = None
scaler = None
feature_columns = None

def load_model_and_preprocessor():
    """Load the trained model and preprocessing components"""
    global model, scaler, feature_columns
    
    try:
        # Check if model file exists
        if os.path.exists('best_model.pkl'):
            model = joblib.load('best_model.pkl')
            print("✅ Model loaded successfully")
        else:
            print("❌ Model file not found.")
            return False
            
        # Load preprocessing components
        if os.path.exists('preprocessing.pkl'):
            preprocessing_info = joblib.load('preprocessing.pkl')
            feature_columns = preprocessing_info['feature_columns']
            scaler = preprocessing_info['scaler']
            print(f"✅ Preprocessing pipeline loaded with {len(feature_columns)} features")
            return True
        else:
            print("❌ Preprocessing file not found. Falling back to training data...")
            # Fallback: Load the training data to get feature information
            if os.path.exists('train.csv'):
                train_df = pd.read_csv('train.csv')
                
                # Prepare feature columns (same preprocessing as in notebook)
                y = train_df["SalePrice"]
                X = train_df.drop("SalePrice", axis=1)
                
                # Get feature columns after preprocessing
                num_features = X.select_dtypes(include=['int64', 'float64']).columns
                cat_features = X.select_dtypes(include=['object']).columns
                
                # Fill missing values (using same logic as notebook)
                for col in num_features:
                    X[col] = X[col].fillna(X[col].median())
                for col in cat_features:
                    X[col] = X[col].fillna(X[col].mode()[0])
                
                # One-hot encode categoricals
                X = pd.get_dummies(X, columns=cat_features, drop_first=True)
                
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
        return False

def preprocess_input(input_data):
    """Preprocess input data to match training format exactly"""
    try:
        # Load the training data to replicate preprocessing exactly
        train_df = pd.read_csv('train.csv')
        
        # Create a sample row with the same structure as training data
        sample_row = train_df.iloc[0:1].copy()
        
        # Update the sample row with input data
        for key, value in input_data.items():
            if key in sample_row.columns:
                sample_row[key] = value
        
        # Add default values based on input data
        sample_row['YearRemodAdd'] = input_data.get('YearBuilt', sample_row['YearRemodAdd'].iloc[0])
        sample_row['BsmtUnfSF'] = input_data.get('GrLivArea', 1500) // 2
        sample_row['TotalBsmtSF'] = input_data.get('GrLivArea', 1500) // 2
        sample_row['1stFlrSF'] = input_data.get('GrLivArea', 1500) // 2
        sample_row['2ndFlrSF'] = input_data.get('GrLivArea', 1500) // 2
        sample_row['GarageArea'] = input_data.get('GarageCars', 0) * 250
        sample_row['GarageYrBlt'] = input_data.get('YearBuilt', sample_row['GarageYrBlt'].iloc[0])
        
        # Handle garage-related fields
        if input_data.get('GarageCars', 0) == 0:
            sample_row['GarageType'] = 'NA'
            sample_row['GarageFinish'] = 'NA'
            sample_row['GarageQual'] = 'NA'
            sample_row['GarageCond'] = 'NA'
        
        # Handle fireplace quality
        if input_data.get('Fireplaces', 0) == 0:
            sample_row['FireplaceQu'] = 'NA'
        
        # Remove SalePrice if it exists (target variable)
        if 'SalePrice' in sample_row.columns:
            sample_row = sample_row.drop('SalePrice', axis=1)
        
        # Apply the exact same preprocessing as in training
        num_features = sample_row.select_dtypes(include=['int64', 'float64']).columns
        cat_features = sample_row.select_dtypes(include=['object']).columns
        
        # Fill missing values with the same strategy as training
        for col in num_features:
            if sample_row[col].isna().any():
                sample_row[col] = sample_row[col].fillna(train_df[col].median())
        
        for col in cat_features:
            if sample_row[col].isna().any():
                mode_val = train_df[col].mode()
                sample_row[col] = sample_row[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')
        
        # One-hot encode categoricals - this should match training exactly
        sample_row = pd.get_dummies(sample_row, columns=cat_features, drop_first=True)
        
        # Scale numerical features using the saved scaler
        if scaler is not None:
            num_cols_to_scale = [col for col in num_features if col in sample_row.columns]
            if num_cols_to_scale:
                sample_row[num_cols_to_scale] = scaler.transform(sample_row[num_cols_to_scale])
        
        # Ensure we have all the features the model expects, in the right order
        final_features = pd.DataFrame(columns=feature_columns)
        
        # Add the processed sample row data
        for col in feature_columns:
            if col in sample_row.columns:
                final_features[col] = sample_row[col].values
            else:
                final_features[col] = 0  # Default value for missing features
        
        return final_features
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        traceback.print_exc()
        raise e

@app.route('/')
def home():
    """Serve the main HTML page"""
    return send_from_directory('templates', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get input data from request
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Preprocess the input data
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Ensure prediction is positive and reasonable
        prediction = max(prediction, 50000)
        
        # Calculate confidence range
        confidence_margin = prediction * 0.15
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
        print("🚀 Server ready! Starting Flask app on http://localhost:5000")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model.")