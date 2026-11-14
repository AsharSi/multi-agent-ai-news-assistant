import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from agents.research_agent import research_node
from agents.summarizer_agent import summarizer_node
from agents.notifier_agent import notifier_node

load_dotenv()

g = StateGraph()
g.add_node("research", research_node)
g.add_node("summary", summarizer_node)
g.add_node("notify", notifier_node)

g.set_entry_point("research")

g.add_edge("research", "summary")
g.add_edge("summary", "notify")
g.add_edge("notify", END)

app = g.compile()

# OpenRouter exposes an OpenAI-compatible API, so ChatOpenAI works with a custom base_url.
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=2000,
)