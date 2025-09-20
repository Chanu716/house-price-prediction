"""
Simple model training script - extracted from the Jupyter notebook
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_and_save_model():
    """Train the house price prediction model and save it"""
    
    print("📊 Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv("train.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Separate features and target
    y = df["SalePrice"]
    X = df.drop("SalePrice", axis=1)
    
    # Separate numerical and categorical features
    num_features = X.select_dtypes(include=['int64', 'float64']).columns
    cat_features = X.select_dtypes(include=['object']).columns
    
    print(f"Numerical features: {len(num_features)}")
    print(f"Categorical features: {len(cat_features)}")
    
    # Fill missing values
    print("🔧 Handling missing values...")
    for col in num_features:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_features:
        X[col] = X[col].fillna(X[col].mode()[0])
    
    # One-hot encode categoricals
    print("🎯 Encoding categorical features...")
    X = pd.get_dummies(X, columns=cat_features, drop_first=True)
    
    # Feature scaling
    print("📏 Scaling features...")
    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])
    
    print(f"Final feature count: {X.shape[1]}")
    
    # Define models to test
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    }
    
    print("🤖 Training and evaluating models...")
    
    # Cross-validation for all models
    results = {}
    for name, model in models.items():
        print(f"   Training {name}...")
        scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        rmse_scores = np.sqrt(-scores)
        results[name] = rmse_scores.mean()
        print(f"   {name} RMSE: {rmse_scores.mean():.2f}")
    
    # Sort results and select best model
    results = dict(sorted(results.items(), key=lambda x: x[1]))
    
    print("\n🏆 Model Leaderboard (lower RMSE is better):")
    for name, score in results.items():
        print(f"   {name}: {score:.2f}")
    
    # Select and train best model
    best_model_name = list(results.keys())[0]
    best_model = models[best_model_name]
    
    print(f"\n✅ Best model selected: {best_model_name}")
    print("🔄 Training final model on full dataset...")
    
    # Train best model on full dataset
    best_model.fit(X, y)
    
    # Save the model and preprocessing info
    joblib.dump(best_model, "best_model.pkl")
    print("💾 Model saved as 'best_model.pkl'")
    
    # Save preprocessing components for consistent feature engineering
    preprocessing_info = {
        'feature_columns': X.columns.tolist(),
        'numerical_features': num_features.tolist(),
        'scaler': scaler,
        'sample_data': X.iloc[0:1]  # Keep a sample for reference
    }
    joblib.dump(preprocessing_info, "preprocessing.pkl")
    print("💾 Preprocessing pipeline saved as 'preprocessing.pkl'")
    
    # Test the saved model
    print("🧪 Testing saved model...")
    loaded_model = joblib.load("best_model.pkl")
    sample_prediction = loaded_model.predict(X.iloc[[0]])
    actual_price = y.iloc[0]
    
    print(f"   Sample prediction: ${sample_prediction[0]:,.2f}")
    print(f"   Actual price: ${actual_price:,.2f}")
    print(f"   Difference: ${abs(sample_prediction[0] - actual_price):,.2f}")
    
    return True

if __name__ == "__main__":
    print("🏠 HousePriceAI Model Training")
    print("=" * 40)
    
    try:
        if train_and_save_model():
            print("\n🎉 Model training completed successfully!")
            print("🚀 You can now run the web application using: python app.py")
        else:
            print("\n❌ Model training failed")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()