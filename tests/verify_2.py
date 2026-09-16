from src.database import get_db_schema, execute_query
from src.tools import predict_churn

print("=== 1. Checking Database Schema Introspection ===")
schema = get_db_schema()
print(schema[:300] + "...\n")

print("=== 2. Testing Sample SQL Query ===")
sample = execute_query("SELECT customer_id FROM customers LIMIT 1;")
test_id = sample[0]["customer_id"]
print(f"Retrieved Test Customer ID: {test_id}\n")

print("=== 3. Testing Model Inference Tool ===")
result = predict_churn(test_id)
print(f"Tool Output: {result}")