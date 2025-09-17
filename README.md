# Integrated-Retail-Analytics-Sales-Prediction-using-Machine-Learning
This project delivers an end-to-end solution for optimizing retail store operations through data-driven insights and predictive modeling. Integrate three heterogeneous datasets — weekly sales, store attributes, and external economic features — into a single, clean analytical base

## 📌 Project Overview
This project integrates sales, store, and external feature datasets to:
- Perform **EDA** (15 charts: Univariate, Bivariate, Multivariate)
- Validate business hypotheses (holidays, promotions, price–sales)
- Train and tune supervised ML models (Linear Regression, Random Forest, Gradient Boosting)
- Deploy the best model via CLI and an interactive **Streamlit app**

## 🚀 Features
- Data cleaning, outlier handling, and feature engineering
- Hypothesis testing with statistical significance
- Model evaluation (RMSE, MAE, R²) & feature importance
- Streamlit dashboard with:
  - Sales prediction form
  - EDA charts
  - RandomForest feature importance

## 🏗️ Project Structure
### integrated-retail-analytics/
- ├─ retail_artifacts/
- │   ├─ app.py
- │   ├─ predict_sales.py
- │   ├─ models/
- │   │   └─ final_model_pipeline.pkl
- │   ├─ plots/
- │   │   └─ [all charts]
- │   └─ rf_feature_importance.csv
- │
- ├─ notebook/
- │   └─ integrated_retail_analytics.ipynb
- │
- └─ README.md


## 📊 Tech Stack
- Python 3.11
- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy, joblib
- Streamlit

## ▶️ Run Locally
1. **Clone the repo**  
   ```bash
   git clone https://github.com/SuziSharma2/integrated-retail-analytics.git
   cd integrated-retail-analytics
