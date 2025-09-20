"""
HousePriceAI - Complete Workflow Script
This script trains the model and starts the web application
"""

import subprocess
import sys
import os
import time

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def run_notebook_training():
    """Run the Jupyter notebook to train the model"""
    print("🤖 Training the machine learning model...")
    
    # Convert notebook to Python script and run it
    try:
        # First, convert notebook to Python script
        subprocess.check_call([
            "jupyter", "nbconvert", "--to", "script", 
            "HousePricePrediction.ipynb", "--output", "train_model"
        ])
        
        # Then run the Python script
        subprocess.check_call([sys.executable, "train_model.py"])
        
        print("✅ Model training completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Model training failed: {e}")
        print("💡 Try running the Jupyter notebook manually:")
        print("   1. Open HousePricePrediction.ipynb in Jupyter")
        print("   2. Run all cells")
        print("   3. Make sure 'best_model.pkl' is saved")
        return False

def start_web_app():
    """Start the Flask web application"""
    print("🌐 Starting the web application...")
    
    # Check if model file exists
    if not os.path.exists("best_model.pkl"):
        print("❌ Model file not found. Training model first...")
        if not run_notebook_training():
            return False
    
    # Start Flask app
    try:
        print("🚀 Starting Flask server at http://localhost:5000")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        
        # Import and run the Flask app
        from app import app, load_model_and_preprocessor
        
        # Load model
        if load_model_and_preprocessor():
            app.run(debug=True, host='0.0.0.0', port=5000)
        else:
            print("❌ Failed to load model")
            return False
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down the server...")
        return True
    except Exception as e:
        print(f"❌ Failed to start web application: {e}")
        return False

def main():
    """Main workflow"""
    print("🏠 Welcome to HousePriceAI Setup!")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("train.csv"):
        print("❌ train.csv not found. Please run this script from the HousePredictions directory.")
        return
    
    print("🔄 Starting complete workflow...")
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed. Please install requirements manually.")
        return
    
    # Step 2: Check for existing model or train new one
    if os.path.exists("best_model.pkl"):
        print("✅ Model file found. Skipping training...")
        user_input = input("🤔 Do you want to retrain the model? (y/N): ").lower()
        if user_input in ['y', 'yes']:
            run_notebook_training()
    else:
        print("🤖 No model found. Training new model...")
        if not run_notebook_training():
            # If automatic training fails, provide manual instructions
            print("\n📋 Manual Setup Instructions:")
            print("1. Install Jupyter: pip install jupyter")
            print("2. Start Jupyter: jupyter notebook")
            print("3. Open HousePricePrediction.ipynb")
            print("4. Run all cells in the notebook")
            print("5. Make sure 'best_model.pkl' is saved")
            print("6. Run this script again")
            return
    
    # Step 3: Start web application
    print("\n🌐 Starting web application...")
    start_web_app()

if __name__ == "__main__":
    main()