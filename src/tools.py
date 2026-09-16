import joblib
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from src.database import execute_query
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.pkl"

# Load serialized pipeline once
MODEL = joblib.load(MODEL_PATH)


def predict_churn(customer_id: str) -> Dict[str, Any]:
    """Queries customer telemetry from SQLite and runs model inference."""
    query = f"""
    SELECT 
        s.tenure,
        s.contract_type,
        s.monthly_charges,
        s.total_charges,
        u.internet_service,
        u.online_security,
        u.tech_support
    FROM subscriptions s
    JOIN service_usage u ON s.customer_id = u.customer_id
    WHERE s.customer_id = '{customer_id}'
    """
    rows = execute_query(query)
    if not rows:
        return {"error": f"Customer ID '{customer_id}' not found in database."}

    df = pd.DataFrame(rows)
    probability = float(MODEL.predict_proba(df)[:, 1][0])

    return {
        "customer_id": customer_id,
        "churn_probability": round(probability, 4),
        "risk_level": "High" if probability >= 0.7 else "Medium" if probability >= 0.4 else "Low",
    }