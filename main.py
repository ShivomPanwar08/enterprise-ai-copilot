import uuid
from langchain_core.messages import HumanMessage
from src.graph import app


def run_cli():
  session_id = str(uuid.uuid4())[:8]
  config = {"configurable": {"thread_id": session_id}, "recursion_limit": 15}

  print("=" * 60)
  print(f"🤖 AI Data Copilot Active | Session ID: {session_id}")
  print("Type 'exit', 'quit', or 'q' to end.")
  print("=" * 60 + "\n")

  while True:
    try:
      user_input = input("\nYou > ").strip()
      if not user_input:
        continue
      if user_input.lower() in ["exit", "quit", "q"]:
        print("Closing session. Goodbye!")
        break

      state = {"messages": [HumanMessage(content=user_input)], "retry_count": 0}

      for event in app.stream(state, config=config, stream_mode="values"):
        latest_msg = event["messages"][-1]
        msg_type = latest_msg.__class__.__name__

        # Show live tool calls clearly
        if msg_type == "AIMessage" and getattr(latest_msg, "tool_calls", None):
          for tool in latest_msg.tool_calls:
            print(f"  ⚡ [Tool Invoked]: {tool['name']}({tool['args']})")

      # Final synthesized response
      final_response = latest_msg.content
      if isinstance(final_response, list):
        final_text = "".join(
            part.get("text", "")
            for part in final_response
            if isinstance(part, dict)
        )
      else:
        final_text = str(final_response)

      print(f"\nCopilot > {final_text}")

    except KeyboardInterrupt:
      print("\nSession interrupted. Exiting.")
      break
    except Exception as e:
      print(f"\n[Error]: {str(e)}")


if __name__ == "__main__":
  run_cli()