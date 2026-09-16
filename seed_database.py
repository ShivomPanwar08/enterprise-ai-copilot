import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

DB_PATH = "data/analytics.db"
MODEL_PATH = "models/churn_model.pkl"

# Reset old DB if it exists so seeding is repeatable
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

print("[1/4] Downloading raw Telco churn data...")
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_URL)

# Clean types
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce').fillna(0.0)
df['Churn_Binary'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

df = df.rename(columns={
    'customerID': 'customer_id',
    'SeniorCitizen': 'senior_citizen',
    'MonthlyCharges': 'monthly_charges',
    'TotalCharges': 'total_charges',
    'Contract': 'contract_type',
    'PaymentMethod': 'payment_method',
    'OnlineSecurity': 'online_security',
    'TechSupport': 'tech_support',
    'InternetService': 'internet_service'
})

print("[2/4] Normalizing into 3 SQLite relational tables...")
customers_df = df[['customer_id', 'gender', 'senior_citizen', 'Partner', 'Dependents']].copy()
customers_df.columns = ['customer_id', 'gender', 'senior_citizen', 'has_partner', 'has_dependents']

subscriptions_df = df[['customer_id', 'tenure', 'contract_type', 'payment_method', 'monthly_charges', 'total_charges']].copy()

usage_df = df[['customer_id', 'internet_service', 'online_security', 'tech_support', 'Churn_Binary']].copy()
usage_df.columns = ['customer_id', 'internet_service', 'online_security', 'tech_support', 'is_churned']

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()  #The Cursor object allows us to execute SQL commands in Python. It acts as a bridge between the database and the Python code, enabling us to perform operations such as creating tables, inserting data, and querying the database.
cursor.execute("PRAGMA foreign_keys = ON;")

cursor.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    gender TEXT,
    senior_citizen INTEGER,
    has_partner TEXT,
    has_dependents TEXT
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS subscriptions (
    sub_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    tenure INTEGER,
    contract_type TEXT,
    payment_method TEXT,
    monthly_charges REAL,
    total_charges REAL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS service_usage (
    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    internet_service TEXT,
    online_security TEXT,
    tech_support TEXT,
    is_churned INTEGER,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);
""")

customers_df.to_sql('customers', conn, if_exists='append', index=False)  #it has 3 options here append, replace, and fail.  Alse, index tells the Pandas not to write the DataFrame's row index into the database table as a separate column. 
subscriptions_df.to_sql('subscriptions', conn, if_exists='append', index=False)
usage_df.to_sql('service_usage', conn, if_exists='append', index=False)
conn.commit() #saves all pending changes to the dataase permanently. 
conn.close() #closes the active bridgebetween the Pythona nd the SQLite database file.
print(f" Database written to {DB_PATH}")

print("[3/4] Training baseline RandomForestBoosting churn model...")
conn = sqlite3.connect(DB_PATH)
query = """
SELECT 
    s.tenure,
    s.contract_type,
    s.monthly_charges,
    s.total_charges,
    u.internet_service,
    u.online_security,
    u.tech_support,
    u.is_churned
FROM subscriptions s
JOIN service_usage u ON s.customer_id = u.customer_id
"""
ml_df = pd.read_sql_query(query, conn)
conn.close()

X = ml_df.drop(columns=['is_churned'])
y = ml_df['is_churned']

categorical_cols = ['contract_type', 'internet_service', 'online_security', 'tech_support']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
print(f"Model ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

print("[4/4] Serializing model artifact...")
joblib.dump(pipeline, MODEL_PATH)
print(f" Pipeline artifact saved to {MODEL_PATH}")