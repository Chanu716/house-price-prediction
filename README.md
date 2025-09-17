# 🏠 House Price Prediction (ML Project)

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

## 👤 Author
- K Chanikya (https://github.com/Chanu716)  
