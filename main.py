"""
Multi-Agent AI News Assistant - entry point.

    python main.py "Latest developments in Generative AI"
    python main.py            # prompts for a topic

Flow (LangGraph):  research -> summarize -> notify -> END
                        \\          \\
                         +-> error <-+   (empty topic / <3 results / summary failed)
"""
import logging
import sys

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

load_dotenv()

from agents.notifier_agent import notifier_node  # noqa: E402
from agents.research_agent import research_node, route_after_research  # noqa: E402
from agents.summarizer_agent import summarizer_node  # noqa: E402
from state import NewsState  # noqa: E402


def error_node(state: NewsState) -> dict:
    """Terminal node for failed runs; errors are printed by main()."""
    return {}


def route_after_summary(state: NewsState) -> str:
    return "notify" if len(state.get("summary_points") or []) == 5 else "error"


def build_graph():
    g = StateGraph(NewsState)
    g.add_node("research", research_node)
    g.add_node("summarize", summarizer_node)
    g.add_node("notify", notifier_node)
    g.add_node("error", error_node)

    g.set_entry_point("research")
    g.add_conditional_edges("research", route_after_research,
                            {"summarize": "summarize", "error": "error"})
    g.add_conditional_edges("summarize", route_after_summary,
                            {"notify": "notify", "error": "error"})
    g.add_edge("notify", END)
    g.add_edge("error", END)
    return g.compile()


app = build_graph()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Quiet noisy HTTP client logs
    for noisy in ("httpx", "httpcore", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    topic = " ".join(sys.argv[1:]).strip()
    if not topic:
        try:
            topic = input("Enter a topic: ").strip()
        except EOFError:
            topic = ""
    result = app.invoke({"topic": topic, "errors": []})

    print("\n" + "=" * 60)
    print(f"TOPIC: {topic}")
    print(f"Research results: {len(result.get('research_results') or [])}")

    if result.get("summary_points"):
        print("\nSUMMARY:")
        for i, p in enumerate(result["summary_points"], 1):
            print(f"  {i}. {p}")
        print("\nSOURCES:")
        for u in result.get("sources", []):
            print(f"  - {u}")

    if result.get("email_text"):
        print("\nEMAIL:\n" + "-" * 60 + "\n" + result["email_text"] + "\n" + "-" * 60)
    if result.get("output_file"):
        print(f"Saved to: {result['output_file']}")

    if result.get("errors"):
        print("\nERRORS:")
        for e in result["errors"]:
            print(f"  - {e}")

    return 0 if result.get("output_file") else 1


if __name__ == "__main__":
    sys.exit(main())
