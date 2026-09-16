import sys
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import subprocess 

# Adding the root directory to sys.path for module imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Ensure SQLite database exists on cloud runtime
db_path = os.path.join(os.path.dirname(__file__), "..", "data", "analytics.db")
if not os.path.exists(db_path):
    subprocess.run(["python", "seed_database.py"], check=True)

import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from src.graph import app
from src.database import execute_query



st.set_page_config(
    page_title="Self-Healing AI Data Copilot",
    page_icon="🤖",
    layout="wide"
)

# 1. Session State Initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Sidebar: Database & Model Telemetry
with st.sidebar:
    st.title("⚙️ Telemetry & System Specs")
    st.markdown(f"**Session Thread ID:** `{st.session_state.thread_id}`")
    
    st.markdown("---")
    st.subheader("Model Status")
    st.success("Churn Prediction Pipeline: Active")
    st.caption("Baseline Metric: ROC-AUC ~0.8041")
    
    st.markdown("---")
    st.subheader("Database Health")
    try:
        cust_count = execute_query("SELECT COUNT(*) as cnt FROM customers")[0]['cnt']
        sub_count = execute_query("SELECT COUNT(*) as cnt FROM subscriptions")[0]['cnt']
        st.write(f"👥 **Customers:** {cust_count}")
        st.write(f"💳 **Subscriptions:** {sub_count}")
    except Exception as e:
        st.error(f"DB Error: {e}")

    if st.button("Clear Chat Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# 3. Main Chat Interface
st.title("🤖 Enterprise Self-Healing Data Copilot")
st.caption("Autonomous querying over SQLite schema with dynamic ML churn predictions.")

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Box
if prompt := st.chat_input("Ask about customer metrics, churn probability, or SQL aggregations..."):
    # Display user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Execution with Live Tracing
    with st.chat_message("assistant"):
        config = {
            "configurable": {"thread_id": st.session_state.thread_id},
            "recursion_limit": 15
        }
        state = {"messages": [HumanMessage(content=prompt)], "retry_count": 0}
        
        status_placeholder = st.status("Thinking & querying...", expanded=True)
        final_text = ""

        try:
            for event in app.stream(state, config=config, stream_mode="values"):
                latest_msg = event["messages"][-1]
                msg_type = latest_msg.__class__.__name__

                if msg_type == "AIMessage" and getattr(latest_msg, "tool_calls", None):
                    for call in latest_msg.tool_calls:
                        status_placeholder.write(f"⚡ **Triggered Tool:** `{call['name']}` with args `{call['args']}`")
                
                if msg_type == "ToolMessage":
                    status_placeholder.write(f"📥 **Tool Result received** (length: {len(str(latest_msg.content))} chars)")

            status_placeholder.update(label="Complete", state="complete", expanded=False)

            # Synthesize final response text
            raw_content = latest_msg.content
            if isinstance(raw_content, list):
                final_text = "".join(part.get("text", "") for part in raw_content if isinstance(part, dict))
            else:
                final_text = str(raw_content)

            st.markdown(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})

        except Exception as err:
            status_placeholder.update(label="Failed", state="error")
            st.error(f"Error during graph execution: {err}")