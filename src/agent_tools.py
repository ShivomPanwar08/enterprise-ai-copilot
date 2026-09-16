from langchain_core.tools import tool
from src.database import execute_query, get_db_schema
from src.tools import predict_churn


@tool
def db_schema_tool() -> str:
    """Returns the SQL schema of the SQLite database tables."""
    return get_db_schema()


@tool
def run_sql_query(query: str) -> str:
    """Executes a SQL query against the database and returns records or error message."""
    try:
        results = execute_query(query)
        return str(results)
    except Exception as e:
        return f"SQL_ERROR: {str(e)}"


@tool
def customer_churn_tool(customer_id: str) -> str:
    """Predicts churn probability and risk level for a specific customer ID."""
    return str(predict_churn(customer_id))