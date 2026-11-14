"""
Member 1 - Research Agent.

Exposes:
    research_node(state)          -> LangGraph node (returns a partial state update)
    route_after_research(state)   -> conditional-edge router for Member 2

Search: Tavily (via LangChain) first, DuckDuckGo as a no-key fallback.
Never raises: every failure is written to `errors`.
"""
import logging
import os
from urllib.parse import urlparse
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from state import NewsState

load_dotenv()
log = logging.getLogger("research_agent")

MIN_RESULTS = 3
MAX_RESULTS = 6
MAX_CONTENT_CHARS = 1500
MAX_TOPIC_CHARS = 200

# ---------- providers (each returns a list of {title, url, content}) ----------

def _search_tavily(query: str) -> list[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY is not set")

    tool = TavilySearch(max_results=MAX_RESULTS)
    resp = tool.invoke({"query": query})
    if not isinstance(resp, dict):
        raise RuntimeError(f"Unexpected Tavily response: {str(resp)[:200]}")
    if resp.get("error"):
        raise RuntimeError(f"Tavily error: {resp['error']}")
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in resp.get("results", [])
    ]


def _search_duckduckgo(query: str) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as ddg:
        raw = ddg.text(query, max_results=MAX_RESULTS) or []
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")}
        for r in raw
    ]


PROVIDERS = [("Tavily", _search_tavily), ("DuckDuckGo", _search_duckduckgo)]


# ---------- cleaning ----------

def _valid_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except ValueError:
        return False


def _clean(raw: list[dict]) -> list[dict]:
    seen, cleaned = set(), []
    for r in raw:
        url = (r.get("url") or "").strip()
        content = " ".join((r.get("content") or "").split())
        if not _valid_url(url) or url in seen or not content:
            continue
        seen.add(url)
        cleaned.append({
            "title": (r.get("title") or "").strip() or url,
            "url": url,
            "content": content[:MAX_CONTENT_CHARS],
        })
        if len(cleaned) >= MAX_RESULTS:
            break
    return cleaned


# ---------- LangGraph node ----------

def research_node(state: NewsState) -> dict:
    topic = (state.get("topic") or "").strip()

    if not topic:
        return {"research_results": [], "errors": ["Research: topic is empty. Please enter a topic."]}
    if len(topic) > MAX_TOPIC_CHARS:
        return {"research_results": [],
                "errors": [f"Research: topic is too long (max {MAX_TOPIC_CHARS} characters)."]}

    failures: list[str] = []
    best: list[dict] = []

    for name, search in PROVIDERS:
        try:
            log.info("Searching %s for %r", name, topic)
            results = _clean(search(topic))
        except Exception as e:  # missing key, network, rate limit, bad response
            failures.append(f"{name} failed: {e}")
            log.warning("%s failed: %s", name, e)
            continue

        if len(results) > len(best):
            best = results
        if len(results) >= MIN_RESULTS:
            log.info("Got %d results from %s", len(results), name)
            return {"research_results": results}   # success: fallback failures stay in logs only
        failures.append(f"{name} returned only {len(results)} usable result(s)")

    msg = f"Research: fewer than {MIN_RESULTS} usable results for '{topic}'. " + "; ".join(failures)
    return {"research_results": best, "errors": [msg]}


def route_after_research(state: NewsState) -> str:
    """For Member 2's add_conditional_edges: 'summarize' or 'error'."""
    return "summarize" if len(state.get("research_results") or []) >= MIN_RESULTS else "error"


# ---------- standalone test: python research_agent.py "your topic" ----------

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    topic = " ".join(sys.argv[1:]) or "latest AI regulation news"
    out = research_node({"topic": topic, "errors": []})
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("Route ->", route_after_research(out))