"""
Agent 3 - Notifier Agent

Receives the summary produced by the Summarizer Agent, turns it into a
professional email notification, and saves it to notifications.txt.

Designed to be used as a LangGraph node: it takes the shared state dict
(expects "topic" and "summary") and returns {"email": ...}.
"""

from datetime import datetime
from main import llm
import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

NOTIFICATIONS_FILE = "notifications.txt" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a corporate communications assistant. Write a short, professional "
        "email to all employees sharing the latest developments on a topic. "
        "Include a 'Subject:' line first, a greeting, the key points as a bulleted "
        "list, the source links, and a polite sign-off from 'AI News Assistant'. "
        "Use only the information given in the summary.",
    ),
    ("human", "Topic: {topic}\n\nSummary:\n{summary}"),
])

chain = prompt | llm | StrOutputParser()


def fallback_email(topic: str, summary: str) -> str:
    """Plain template used if the LLM call fails."""
    return (
        f"Subject: Latest Update on {topic}\n\n"
        "Hello Team,\n\n"
        f"Here is a quick summary of the latest developments on {topic}:\n\n"
        f"{summary}\n\n"
        "Best regards,\nAI News Assistant"
    )


def save_notification(email: str) -> None:
    """Append the email to notifications.txt with a timestamp."""
    with open(NOTIFICATIONS_FILE, "a", encoding="utf-8") as f:
        f.write(f"===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        f.write(email.strip() + "\n\n")


def notifier_agent(state: dict) -> dict:
    """LangGraph node: summary in -> email out (and saved to file)."""
    topic = state.get("topic") or "Technology"
    summary = state.get("summary")

    # Handle missing information
    if not summary:
        print("[Notifier] No summary received - skipping email.")
        return {"email": "", "error": "Notifier: summary is missing"}

    print("[Notifier] Generating email notification...")
    try:
        email = chain.invoke({"topic": topic, "summary": summary})
    except Exception as e:
        print(f"[Notifier] LLM failed ({e}); using template email.")
        email = fallback_email(topic, summary)

    save_notification(email)
    print(f"[Notifier] Email saved to {NOTIFICATIONS_FILE}")
    return {"email": email}


if __name__ == "__main__":
    # Quick standalone test with a sample summary
    sample_state = {
        "topic": "Generative AI",
        "summary": (
            "1. New multimodal models released.\n"
            "2. Companies adopting AI agents.\n"
            "3. Open-source models improving fast.\n"
            "4. New AI regulations proposed.\n"
            "5. Lower inference costs.\n"
            "Sources: https://example.com/genai-news"
        ),
    }
    result = notifier_agent(sample_state)
    print("\n" + result["email"])
