# 🏠 HousePriceAI - Smart Home Valuation System

A modern, production-ready **house price prediction web application** built with Flask and Machine Learning. Features a beautiful dark-themed UI and achieves accurate predictions using ensemble ML models.

## 🌟 Live Demo

**🔗 Live App:** [https://houseprice-ai-xxxx.onrender.com](https://house-price-prediction-dhlh.onrender.com)

Try it now! Enter house details and get instant price predictions.

![Status](https://img.shields.io/badge/Status-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-green)
![ML](https://img.shields.io/badge/ML-XGBoost-orange)

**🚀 Quick Start:** `python app.py` → Open `http://localhost:5000`

## ✨ Features

### 🎯 **Core Capabilities**
- **Accurate Predictions** - Ensemble ML models (XGBoost, Random Forest)
- **81 Features** - Comprehensive house attributes analysis
- **Real-time API** - Instant predictions via REST endpoints
- **Professional UI** - Modern, responsive dark-themed interface

### 🖥️ **Web Application**
- **Interactive Form** - User-friendly input with organized sections
- **Smart Validation** - Real-time form validation and auto-suggestions
- **Loading States** - Smooth animations and feedback
- **Mobile Responsive** - Works on all devices
- **Error Handling** - Graceful error messages

### 🧠 **Machine Learning**
- **Multiple Models** - Compares Linear Regression, Decision Tree, Random Forest, XGBoost
- **Auto Selection** - Picks best performing model (~27,768 RMSE)
- **Feature Engineering** - Advanced preprocessing pipeline
- **Confidence Intervals** - Prediction uncertainty estimates

## 🏗️ Project Structure

```
HousePredictions/
├── 📊 HousePricePrediction.ipynb  # Training notebook & analysis
├── 🌐 app.py                      # Flask web server
├── 🎨 templates/
│   └── index.html                 # Modern web interface
├── 🏆 best_model.pkl              # Trained ML model
├── 🔧 preprocessing.pkl           # Feature preprocessing pipeline
├── 📋 requirements.txt            # Dependencies
├── 📄 README.md                   # This file
├── 📜 LICENSE                     # MIT License
└── 📊 Dataset files               # train.csv, test.csv, etc.
```

## 🚀 Quick Start

### 1️⃣ **Clone Repository**
```bash
git clone https://github.com/Chanu716/house-price-prediction.git
cd house-price-prediction
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Download Dataset** (if training from scratch)
Download from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques):
- `train.csv` (1,460 rows × 81 columns)
- `test.csv` (1,459 rows × 80 columns)
- `data_description.txt`
- `sample_submission.csv`

### 4️⃣ **Launch Web App**
```bash
python app.py
```

### 5️⃣ **Open Browser**
Navigate to `http://localhost:5000`

## 💻 Usage Guide

### 🌐 **Web Interface**
1. **Fill Form** - Enter house details (or use example button)
2. **Submit** - Click "Predict Price"
3. **View Results** - Get predicted price with confidence range
4. **Try More** - Test different combinations

### 🔧 **API Endpoints**

#### **Predict House Price**
```bash
POST /predict
Content-Type: application/json

{
  "MSSubClass": 60,
  "LotArea": 8450,
  "OverallQual": 7,
  "YearBuilt": 2003,
  "TotalBsmtSF": 856,
  "GrLivArea": 1710,
  "FullBath": 2,
  "BedroomAbvGr": 3,
  "GarageArea": 548
  # ... (all 81 features)
}

# Response
{
  "prediction": 208500.00,
  "confidence_range": {
    "min": 177225.00,
    "max": 239775.00
  },
  "status": "success"
}
```

#### **Health Check**
```bash
GET /health
# Response: {"status": "healthy", "model_loaded": true}
```

## 📊 Dataset Information

**House Prices - Advanced Regression Techniques**
- **Source:** [Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Training Data:** 1,460 houses
- **Features:** 81 attributes
- **Target:** SalePrice (house sale price in USD)

### **Key Features:**
- **Location:** MSSubClass, MSZoning, Neighborhood
- **Size:** LotArea, GrLivArea, TotalBsmtSF
- **Quality:** OverallQual, OverallCond
- **Age:** YearBuilt, YearRemodAdd
- **Rooms:** BedroomAbvGr, KitchenAbvGr
- **Bathrooms:** FullBath, HalfBath
- **Garage:** GarageArea, GarageCars

## 🏆 Model Performance

| Model | RMSE | Status |
|-------|------|--------|
| **XGBoost** | **~27,768** | ✅ Best |
| Random Forest | ~30,103 | ✅ Good |
| Decision Tree | ~40,799 | ⚠️ Fair |
| Linear Regression | ~42,923 | ⚠️ Baseline |

*Best model automatically selected based on cross-validation*

## 🛠️ Technical Details

### 🧠 **ML Pipeline**
- **Preprocessing:** Missing value imputation, one-hot encoding
- **Feature Engineering:** Handling categorical and numerical features
- **Scaling:** StandardScaler for numerical features
- **Models:** Linear Regression, Decision Tree, Random Forest, XGBoost
- **Validation:** Cross-validation for model selection

### 🌐 **Web Stack**
- **Backend:** Flask 2.x with CORS support
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Styling:** Modern dark theme with glassmorphism effects
- **Responsive:** Mobile-first design

## 🚀 Deployment Guide

### **Option 1: Render (Recommended - Free & Easy)**

1. **Add `gunicorn` to requirements.txt:**
```bash
echo "gunicorn==20.1.0" >> requirements.txt
```

2. **Create `render.yaml` in project root:**
```yaml
services:
  - type: web
    name: houseprice-ai
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
```

3. **Deploy:**
   - Push code to GitHub
   - Go to [render.com](https://render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect settings
   - Click "Create Web Service"

**Your Live URL:** `https://houseprice-ai-xxxx.onrender.com`

### **Option 2: Heroku**

1. **Create `Procfile` in project root:**
```bash
echo "web: gunicorn app:app" > Procfile
```

2. **Deploy:**
```bash
heroku login
heroku create houseprice-ai
git push heroku master
heroku open
```

**Your Live URL:** `https://houseprice-ai.herokuapp.com`

### **Option 3: Railway**

1. **Deploy:**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub"
   - Select your repository
   - Railway auto-detects Python app
   
2. **Add start command (if needed):**
   - Settings → Start Command: `gunicorn app:app`

**Your Live URL:** `https://houseprice-ai.up.railway.app`

### **Option 4: PythonAnywhere**

1. **Upload files** to PythonAnywhere
2. **Configure WSGI file:**
```python
import sys
path = '/home/yourusername/HousePredictions'
if path not in sys.path:
    sys.path.append(path)

from app import app as application
```

**Your Live URL:** `https://yourusername.pythonanywhere.com`

## 🔧 Development

### 📓 **Jupyter Notebook**
Explore the complete ML pipeline:
```bash
jupyter notebook HousePricePrediction.ipynb
```

### 🔨 **Local Development**
```bash
# Run with hot reload
export FLASK_ENV=development  # On Windows: set FLASK_ENV=development
python app.py
```

## ⚠️ Limitations

- **Dataset Specific:** Trained on Ames, Iowa housing data (2006-2010)
- **Feature Required:** All 81 features must be provided for prediction
- **US Market:** Optimized for US housing market patterns
- **Time Sensitive:** Model may need retraining with current market data

## 🚀 Future Enhancements

- [ ] **Feature Importance** visualization
- [ ] **Interactive charts** for prediction trends
- [ ] **Batch predictions** for multiple houses
- [ ] **User accounts** and prediction history
- [ ] **Model retraining** pipeline
- [ ] **Mobile app** version

## 👨‍💻 Developer Information

**Name:** Chanikya  
**Email:** [karrichanikya@gmail.com](mailto:karrichanikya@gmail.com)  
**Phone:** [+91 9182789929](tel:+919182789929)  
**GitHub:** [Chanu716](https://github.com/Chanu716)

### 🎓 **About This Project**
This project demonstrates:
- **Machine Learning:** Regression models and ensemble techniques
- **Web Development:** Flask API and modern UI design
- **Data Science:** Feature engineering and model selection
- **Deployment:** Production-ready application deployment
- **Full-stack Skills:** End-to-end ML application development

> 🎓 **Note**: This project is developed for educational and learning purposes to demonstrate machine learning integration with web applications.

## 📝 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## ⭐ Support

If you found this project helpful:

1. **⭐ Star this repository**
2. **🍴 Fork for your own projects**
3. **📢 Share with others**
4. **🐛 Report issues**
5. **💡 Suggest improvements**

---

<div align="center">

### 🏠 **Ready to predict house prices? Start now!**

```bash
git clone https://github.com/Chanu716/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
python app.py
```

**[🌐 Open http://localhost:5000](http://localhost:5000)**

---

**Built with ❤️ for learning ML deployment**

</div>

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
