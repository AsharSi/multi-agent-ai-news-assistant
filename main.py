import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

g = StateGraph()

load_dotenv()

# OpenRouter exposes an OpenAI-compatible API, so ChatOpenAI works with a custom base_url.
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=2000,
)