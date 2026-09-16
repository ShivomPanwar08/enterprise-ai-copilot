import sqlite3
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "analytics.db"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_db_schema() -> str:
    """Introspects SQLite and returns table definitions for the LLM context."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()
    conn.close()
    return "\n\n".join([row["sql"] for row in tables if row["sql"]])


def execute_query(sql_query: str) -> List[Dict[str, Any]]:
    """Executes a SQL query and returns rows as dictionaries."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()