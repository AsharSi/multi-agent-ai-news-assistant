"""Shared LangGraph state. ALL members import NewsState from here - don't redefine it."""
import operator
from typing import Annotated, TypedDict


class NewsState(TypedDict, total=False):
    topic: str
    research_results: list[dict]   # [{title, url, content}]
    summary_points: list[str]      # exactly 5
    sources: list[str]
    email_text: str
    output_file: str
    # Reducer: each node returns ONLY its new errors; LangGraph appends them.
    # Without this, a later node's "errors" overwrites earlier ones.
    errors: Annotated[list[str], operator.add]