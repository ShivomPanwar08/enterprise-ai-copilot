import sqlite3
import joblib
import pandas as pd

# Check SQLite tables
conn = sqlite3.connect("data/analytics.db")
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
print("Tables in SQLite DB:", tables)
conn.close()

# Check Model Inference
model = joblib.load("models/churn_model.pkl")
test_sample = pd.DataFrame([{
    'tenure': 1,
    'contract_type': 'Month-to-month',
    'monthly_charges': 85.0,
    'total_charges': 85.0,
    'internet_service': 'Fiber optic',
    'online_security': 'No',
    'tech_support': 'No'
}])
prob = model.predict_proba(test_sample)[0][1]
print(f"Test Sample Churn Probability: {prob:.2%}")