# HousePriceAI - Smart Home Valuation System

A modern, AI-powered web application for predicting house prices using machine learning. This system features a beautiful dark-themed UI and integrates with trained ML models to provide accurate house price predictions.

## 👨‍💻 **Developer Information**
- **Name**: Chanikya
- **Email**: karrichanikya@gmail.com
- **Phone**: +91 9182789929
- **Purpose**: Learning Project

> 🎓 **Note**: This project is developed for educational and learning purposes to demonstrate machine learning integration with web applications.

## 🏗️ Project Structure

```
d:\ML-Projects\HousePredictions\
├── templates/
│   └── index.html              # Modern web interface
├── HousePricePrediction.ipynb  # Original ML notebook
├── train.csv                   # Training dataset (download from Kaggle)
├── test.csv                    # Test dataset (download from Kaggle)
├── data_description.txt        # Dataset descriptions (download from Kaggle)
├── sample_submission.csv       # Sample format (download from Kaggle)
├── app.py                      # Main Flask application
├── app_simple.py               # Simplified Flask app
├── app_fixed.py                # Fixed Flask app (recommended)
├── train_model.py              # Standalone model training script
├── run_app.py                  # Complete workflow automation
├── test_api.py                 # API testing utilities
├── quick_test.py               # Quick API test
├── simple_test.py              # Simple API test
├── requirements.txt            # Python dependencies
├── best_model.pkl              # Trained ML model (generated)
└── README.md                   # This file
```

## � Dataset

This project uses the **House Prices - Advanced Regression Techniques** dataset from Kaggle.

**📥 Dataset Download**: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

### Required Files:
- `train.csv` - Training dataset (1460 rows × 81 columns)
- `test.csv` - Test dataset (1459 rows × 80 columns)  
- `data_description.txt` - Detailed feature descriptions
- `sample_submission.csv` - Sample submission format

> **Note**: Download these files from Kaggle and place them in the project root directory before running the application.

## �🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
cd "d:\ML-Projects\HousePredictions"
python run_app.py
```

### Option 2: Manual Setup

1. **Download Dataset**
   - Go to: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
   - Download and extract the dataset files to the project root
   - Ensure you have: `train.csv`, `test.csv`, `data_description.txt`, `sample_submission.csv`

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**
   ```bash
   python train_model.py
   ```

4. **Start the Web Application**
   ```bash
   python app_fixed.py
   ```

4. **Open Your Browser**
   Navigate to: http://localhost:5000

## � Features

### Web Interface
- **Modern Dark Theme**: Beautiful, professional UI with glassmorphism effects
- **Responsive Design**: Works perfectly on mobile, tablet, and desktop
- **Interactive Form**: User-friendly form with organized sections
- **Real-time Validation**: Smart form validation and auto-suggestions
- **Loading States**: Smooth loading animations and feedback
- **Error Handling**: Graceful error messages and fallbacks

### Machine Learning
- **Multiple Models**: Compares Linear Regression, Decision Tree, Random Forest, and XGBoost
- **Best Model Selection**: Automatically selects the best performing model
- **Feature Engineering**: Handles missing values, encoding, and scaling
- **Confidence Ranges**: Provides prediction confidence intervals
- **Robust Preprocessing**: Handles various input scenarios gracefully

### API Endpoints
- `GET /` - Main web interface
- `POST /predict` - House price prediction
- `GET /health` - System health check

## 📊 Model Performance

The system automatically trains and compares multiple models:

- **XGBoost**: ~27,768 RMSE (Best)
- **Random Forest**: ~30,103 RMSE
- **Decision Tree**: ~40,799 RMSE
- **Linear Regression**: ~42,923 RMSE

## 🏠 Input Features

The web interface collects the following information:

### Basic Information
- Overall Quality (1-10)
- Year Built
- Above Grade Living Area (sq ft)

### House Features
- Bedrooms Above Grade
- Full Bathrooms
- Total Rooms Above Grade

### Property Details
- Lot Size (sq ft)
- Neighborhood
- House Style

### Additional Features
- Garage Size (number of cars)
- Fireplaces
- Central Air Conditioning

## 🔧 API Usage

### Prediction Request
```javascript
POST /predict
Content-Type: application/json

{
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
```

### Response
```javascript
{
  "prediction": 208500.00,
  "confidence_range": {
    "min": 177225.00,
    "max": 239775.00
  },
  "status": "success"
}
```

## 🧪 Testing

Run the API tests:
```bash
python test_api.py
```

Quick health check:
```bash
python quick_test.py
```

## 📌 Overview
This project predicts house prices using machine learning models on the **Kaggle House Prices Dataset**.  
The workflow covers:
- Data cleaning & preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Training multiple ML models
- Automatic model selection
- Saving and loading the best model

---

## 📂 Dataset
- Source: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
- Features: Various property attributes (lot area, overall quality, year built, etc.)  
- Target: `SalePrice` (house sale price in USD)

---

## ⚙️ Project Workflow
1. **Day 1** → Data loading & preprocessing  
2. **Day 2** → Exploratory Data Analysis (EDA)  
3. **Day 3** → Feature engineering (encoding, scaling)  
4. **Day 4** → Baseline models (Linear Regression, Decision Tree, Random Forest)  
5. **Day 5** → Advanced models (XGBoost, LightGBM)  
6. **Day 6** → Automatic model selection with cross-validation  
7. **Day 7** → Model saving, final prediction, and deployment prep  

---

## 📊 Results
- Best Model: **(auto-selected from leaderboard)**  
- Cross-validation Score: **X.XX (RMSE)**  
- Example Prediction: `203,835 USD` for a sample input  

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/Chanu716/house-price-prediction.git
   cd house-price-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook:
   ```bash
   jupyter notebook HousePricePrediction.ipynb
   ```

---

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)  
- **Scikit-Learn** (ML models, preprocessing, evaluation)  
- **XGBoost / LightGBM** (boosted tree models)  
- **Joblib** (saving best model)  

---

## 📌 Future Improvements
- Hyperparameter tuning with GridSearchCV / Optuna  
- Feature importance visualization  
- Web app deployment with Streamlit/Flask  

---

## 👤 Author & Contact
**Developer**: Chanikya  
**Email**: karrichanikya@gmail.com  
**Phone**: +91 9182789929  
**GitHub**: https://github.com/Chanu716  

> 💡 **Learning Project**: This application was built as part of a machine learning learning journey. Feel free to reach out for discussions about the implementation or improvements!

---

**HousePriceAI** - Making home valuation intelligent and accessible! 🏠✨  
*A learning project by Chanikya*  
