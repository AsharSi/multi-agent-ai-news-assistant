"""
Agent 3 - Notifier Agent

Receives the 5 summary points + sources from the Summarizer Agent, turns them
into a professional email notification, and saves it to
output/notification_<TIMESTAMP>.txt.

LangGraph node: reads "topic", "summary_points", "sources";
returns {"email_text", "output_file"} (or {"errors": [...]}).
"""
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import llm
from state import NewsState

log = logging.getLogger("notifier_agent")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a corporate communications assistant. Write a short, professional "
        "email to all employees sharing the latest developments on a topic. "
        "Start with a 'Subject:' line, then a greeting, the key points as a bulleted "
        "list, a 'Sources:' section listing every source link, and a polite sign-off "
        "from 'AI News Assistant'. Use only the information given. "
        "Return plain text only (no markdown code fences).",
    ),
    ("human", "Topic: {topic}\n\nKey points:\n{points}\n\nSources:\n{sources}"),
])

chain = prompt | llm | StrOutputParser()


def _fallback_email(topic: str, points: str, sources: str) -> str:
    """Plain template used if the LLM call fails."""
    return (
        f"Subject: Latest Update on {topic}\n\n"
        "Hello Team,\n\n"
        f"Here is a quick summary of the latest developments on {topic}:\n\n"
        f"{points}\n\n"
        f"Sources:\n{sources}\n\n"
        "Best regards,\nAI News Assistant"
    )


def _save(email: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"notification_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text(email.strip() + "\n", encoding="utf-8")
    return path


def notifier_node(state: NewsState) -> dict:
    """LangGraph node: summary in -> email out (and saved to file)."""
    topic = state.get("topic") or "Technology"
    summary_points = state.get("summary_points") or []
    sources = state.get("sources") or []

    if not summary_points:
        return {"errors": ["Notifier: summary is missing - email not generated."]}

    points = "\n".join(f"- {p}" for p in summary_points)
    links = "\n".join(f"- {u}" for u in sources) or "- (no sources)"

    log.info("Generating email notification...")
    try:
        email = chain.invoke({"topic": topic, "points": points, "sources": links}).strip()
        if not email:
            raise ValueError("empty LLM response")
    except Exception as e:
        log.warning("LLM failed (%s); using template email.", e)
        email = _fallback_email(topic, points, links)

    try:
        path = _save(email)
    except OSError as e:
        return {"email_text": email, "errors": [f"Notifier: could not write file: {e}"]}

    log.info("Email saved to %s", path)
    return {"email_text": email, "output_file": str(path)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = notifier_node({
        "topic": "Generative AI",
        "summary_points": [
            "New multimodal models released.",
            "Companies adopting AI agents.",
            "Open-source models improving fast.",
            "New AI regulations proposed.",
            "Lower inference costs.",
        ],
        "sources": ["https://example.com/genai-news"],
    })
    print(out.get("email_text"), "\n->", out.get("output_file"), out.get("errors"))
