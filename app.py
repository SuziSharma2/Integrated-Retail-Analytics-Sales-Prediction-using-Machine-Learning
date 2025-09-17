import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ----------------------------
# Paths & Load model
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model_pipeline.pkl")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
FI_CSV = os.path.join(BASE_DIR, "models", "rf_feature_importance.csv")

model = joblib.load(MODEL_PATH)

# Try to load feature importance
fi = None
if os.path.exists(FI_CSV):
    fi = pd.read_csv(FI_CSV, index_col=0).squeeze()

# ----------------------------
# Streamlit layout
# ----------------------------
st.set_page_config(page_title="Retail Sales Prediction", layout="wide")
st.title("🛍️ Integrated Retail Analytics for Store Optimization")
st.markdown(
    """
    Use this app to:
    - Explore key insights from your data.
    - Predict **sales_amount** for any store configuration.
    - Understand feature importance.
    """
)

# Tabs
tab_pred, tab_charts, tab_fi = st.tabs(["🔮 Prediction", "📊 EDA Charts", "🌟 Feature Importance"])

# ----------------------------
# Tab 1: Prediction
# ----------------------------
with tab_pred:
    st.header("Enter Store Details")

    col1, col2 = st.columns(2)
    with col1:
        store = st.number_input("Store", min_value=1, step=1)
        dept = st.number_input("Dept", min_value=1, step=1)
        weekly_sales = st.number_input("Weekly Sales", value=20000.0)
        temperature = st.number_input("Temperature", value=60.0)
        fuel_price = st.number_input("Fuel Price", value=3.5)
        cpi = st.number_input("CPI", value=210.0)
        unemployment = st.number_input("Unemployment", value=7.0)
        size = st.number_input("Size", value=150000)

    with col2:
        isholiday_x = st.checkbox("Is Holiday (sales)?", value=False)
        isholiday_y = st.checkbox("Is Holiday (features)?", value=False)
        store_type = st.selectbox("Store Type", ["A", "B", "C"])
        year = st.number_input("Year", value=2012)
        month = st.number_input("Month", min_value=1, max_value=12, value=3)
        dayofweek = st.number_input("Day of Week (0=Mon)", min_value=0, max_value=6, value=4)
        is_weekend = st.selectbox("Is Weekend?", [0, 1])

    if st.button("Predict Sales Amount"):
        row = pd.DataFrame([{
            "store": store,
            "dept": dept,
            "weekly_sales": weekly_sales,
            "isholiday_x": isholiday_x,
            "temperature": temperature,
            "fuel_price": fuel_price,
            "cpi": cpi,
            "unemployment": unemployment,
            "isholiday_y": isholiday_y,
            "type": store_type,
            "size": size,
            "year": year,
            "month": month,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend
        }])
        try:
            pred = model.predict(row)[0]
            st.success(f"Predicted **sales_amount**: {pred:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ----------------------------
# Tab 2: EDA Charts
# ----------------------------
with tab_charts:
    st.header("Exploratory Data Analysis")

    if os.path.exists(PLOTS_DIR):
        imgs = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")])
        for img in imgs:
            st.image(os.path.join(PLOTS_DIR, img), caption=img)
    else:
        st.info("No plots found. Make sure you ran the notebook and saved plots to 'plots/'.")

# ----------------------------
# Tab 3: Feature Importance
# ----------------------------
with tab_fi:
    st.header("RandomForest Feature Importance")
    if fi is not None and not fi.empty:
        top_n = st.slider("Show top N features", min_value=5, max_value=min(30, len(fi)), value=15)
        top_fi = fi.head(top_n)

        st.bar_chart(top_fi)

        st.markdown(
            """
            **Interpretation:**  
            - Features with higher bars have greater impact on predicted sales.  
            - Use these insights to guide promotions, inventory & staffing.
            """
        )
    else:
        st.info("Feature importance not found. Run the notebook to generate 'rf_feature_importance.csv'.")
