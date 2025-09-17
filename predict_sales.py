import joblib
import pandas as pd

# ==== 1. Load the trained model ====
MODEL_PATH = "retail_artifacts/models/final_model_pipeline.pkl"
model = joblib.load(MODEL_PATH)

# ==== 2. Define the expected columns ====
cols = [
    "store", "dept", "weekly_sales", "isholiday_x",
    "temperature", "fuel_price", "cpi", "unemployment",
    "isholiday_y", "type", "size", "year", "month",
    "dayofweek", "is_weekend"
]

# ==== 3. Create your input data ====
# Replace these values with real data or read from a CSV
data = [[
    1,          # store
    5,          # dept
    0.0,        # weekly_sales (0 if unknown)
    0,          # isholiday_x
    42.5,       # temperature
    2.95,       # fuel_price
    211.3,      # cpi
    8.2,        # unemployment
    0,          # isholiday_y
    "A",        # type
    151315,     # size
    2012,       # year
    2,          # month
    3,          # dayofweek
    0           # is_weekend
]]

df_new = pd.DataFrame(data, columns=cols)

# ==== 4. Make prediction ====
pred = model.predict(df_new)
print("Predicted sales_amount:", float(pred[0]))
