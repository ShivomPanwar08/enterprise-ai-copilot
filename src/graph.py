import os
import time
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent_tools import customer_churn_tool, db_schema_tool, run_sql_query
from src.state import AgentState

tools = [db_schema_tool, run_sql_query, customer_churn_tool]

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
).bind_tools(tools)

SYSTEM_PROMPT = """You are an autonomous AI data copilot.
You have access to a database via `run_sql_query` and a churn ML model via `customer_churn_tool`.

STRICT RULES:
1. If a requested column does not exist in schema, DO NOT keep looping PRAGMA queries. State clearly that the metric is unavailable.
2. Call `db_schema_tool` at most once per conversation.
3. For churn risk, always invoke `customer_churn_tool`.
4. Remember context from previous turns in the conversation.
"""


def agent_node(state: AgentState):
  time.sleep(1)  # Rate-limit safety buffer
  messages = state["messages"]
  if not isinstance(messages[0], SystemMessage):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
  response = llm.invoke(messages)
  return {"messages": [response]}


def should_continue(state: AgentState):
  messages = state["messages"]
  last_message = messages[-1]

  if not getattr(last_message, "tool_calls", None):
    return END

  tool_count = sum(
      1 for m in messages if m.__class__.__name__ == "ToolMessage"
  )
  if tool_count >= 5:
    return END

  return "tools"


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")

# Memory Checkpointer bind kar rahe hain
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)