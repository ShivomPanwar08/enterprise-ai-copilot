# Enterprise Self-Healing AI Data Copilot 🤖

An autonomous, multi-agent enterprise copilot engineered to bridge structured SQL databases and predictive machine learning models. Built with **LangGraph**, **Gemini**, and **Streamlit**, the system features dynamic schema introspection, stateful conversational memory, parallel tool execution, and an automated self-healing execution loop that resolves schema drift and hallucinated query errors without human intervention.

---

## 🌟 Key Architecture & Capabilities

* **Dynamic Schema Introspection:** Evaluates live SQLite table definitions (`customers`, `subscriptions`, `service_usage`) on demand rather than relying on static system prompts.
* **Hybrid Multi-Hop Reasoning:** Seamlessly merges multi-table SQL aggregations with pre-trained Scikit-Learn predictive churn pipelines in a single reasoning cycle.
* **Self-Healing Loop:** Automatically catches SQL execution errors and missing schema attributes (e.g., non-existent metrics), inspects metadata, and recovers gracefully without crashing.
* **Stateful Conversational Memory:** Leverages LangGraph's checkpointer mechanism (`MemorySaver`) to retain multi-turn context across queries.
* **Telemetry & Traceability:** Real-time visibility into tool selection, raw generated SQL queries, and ML prediction confidences via both interactive CLI and Streamlit UI.

---

## 🛠️ Tech Stack

* **Agent Orchestration:** LangGraph, LangChain Core
* **Language Model:** Google Gemini (`gemini-flash-3.1-lite` / function calling)
* **Storage & Relational Layer:** SQLite3
* **Machine Learning Pipeline:** Scikit-Learn, Pandas, NumPy, Joblib (Baseline Churn Model ROC-AUC: ~0.8041)
* **Frontend & Monitoring:** Streamlit

---

## 📁 Repository Structure

```text
SELF HEALING COPILOT/
│
├── data/
│   └── analytics.db             # Normalized SQLite production database
├── models/
│   └── churn_model.pkl          # Scikit-Learn churn prediction pipeline
├── src/
│   ├── database.py              # SQLite connection, execution, and schema introspection
│   ├── tools.py                 # Core ML inference wrapper
│   ├── agent_tools.py           # LangChain tool interfaces for agent invocation
│   ├── state.py                 # LangGraph AgentState schema
│   └── graph.py                 # Graph nodes, conditional edges, and memory checkpointer
├── ui/
│   └── app.py                   # Full-stack Streamlit dashboard with live telemetry
├── main.py                      # Interactive multi-turn CLI interface
├── requirements.txt             # Dependency specification
└── README.md                    # System architecture and documentation