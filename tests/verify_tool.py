from langchain_core.messages import HumanMessage
from src.graph import app

# Deliberately tricky query to test schema correction / healing
query = "Tell me the internet plan and risk for customer 7590-VHVEG. Also check what their wifi_type is."
print(f"User Query: {query}\n" + "-" * 40)

state = {"messages": [HumanMessage(content=query)], "retry_count": 0}

for event in app.stream(state, stream_mode="values"):
    latest_msg = event["messages"][-1]
    sender = latest_msg.__class__.__name__
    if sender in ["AIMessage", "ToolMessage"]:
        print(f"\n[{sender}]:\n{latest_msg.content}")
        if getattr(latest_msg, "tool_calls", None):
            print(f"Tool Invoked: {latest_msg.tool_calls}")