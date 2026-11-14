
"""
Member 2 - Summarizer Agent
 
Exposes:
    summarizer_node(state) -> LangGraph node
 
Responsibilities:
    1. Receive research results from Research Agent.
    2. Generate exactly 5 key summary points.
    3. Include relevant source URLs.
    4. Handle errors without crashing.
"""
 
import os
import json
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
 
from state import NewsState
 
load_dotenv()
 
logger = logging.getLogger("summarizer_agent")
 
 
def summarizer_node(state: NewsState) -> dict:
    """
    Receives research_results and returns:
        summary_points: exactly 5 strings
        sources: relevant research URLs
 
    On failure, returns only newly generated errors.
    """
 
    try:
        # 1. Receive research findings
        research_results = state.get("research_results") or []
 
        if not research_results:
            return {
                "errors": [
                    "Summarizer: No research results received."
                ]
            }
 
        # 2. Validate research data
        valid_results = []
 
        for item in research_results:
            if not isinstance(item, dict):
                continue
 
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or "").strip()
 
            if url and content:
                valid_results.append({
                    "title": title,
                    "url": url,
                    "content": content
                })
 
        if not valid_results:
            return {
                "errors": [
                    "Summarizer: Research results contain no usable content."
                ]
            }
 
        # 3. Read OpenRouter configuration
        api_key = os.getenv("OPENROUTER_API_KEY")
        model_name = os.getenv("OPENROUTER_MODEL")
 
        if not api_key:
            return {
                "errors": [
                    "Summarizer: OPENROUTER_API_KEY is missing."
                ]
            }
 
        if not model_name:
            return {
                "errors": [
                    "Summarizer: OPENROUTER_MODEL is missing."
                ]
            }
 
        # 4. Prepare research content
        research_text = ""
 
        for index, item in enumerate(valid_results, start=1):
            research_text += (
                f"\nResearch Article {index}\n"
                f"Title: {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Content: {item['content']}\n"
                f"{'-' * 50}\n"
            )
 
        available_urls = list(dict.fromkeys(
            item["url"] for item in valid_results
        ))
 
        # 5. Initialize OpenRouter LLM
        llm = ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-5-mini"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0,  # makes the output more consistent run to run
        )
 
        # 6. Create summarization prompt
        prompt = f"""
You are an expert technology news summarizer.
 
Analyze the research articles provided below.
 
YOUR TASKS:
 
1. Generate exactly 5 key summary points.
2. Each point must be concise, clear and informative.
3. Include important facts, developments and findings.
4. Do not repeat information.
5. Do not invent facts.
6. Select relevant source URLs from the supplied articles.
 
IMPORTANT:
- Return exactly 5 summary points.
- Use only the supplied source URLs.
- Do not create or modify URLs.
- Return valid JSON only.
- Do not include Markdown or explanations outside JSON.
 
EXPECTED JSON FORMAT:
 
{{
    "summary_points": [
        "First key point",
        "Second key point",
        "Third key point",
        "Fourth key point",
        "Fifth key point"
    ],
    "sources": [
        "https://example.com/article1",
        "https://example.com/article2"
    ]
}}
 
AVAILABLE SOURCE URLS:
{json.dumps(available_urls, indent=2)}
 
RESEARCH FINDINGS:
{research_text}
"""
 
        # 7. Generate summary
        response = llm.invoke(prompt)
 
        response_text = response.content
 
        if isinstance(response_text, list):
            response_text = "".join(
                part.get("text", "")
                if isinstance(part, dict)
                else str(part)
                for part in response_text
            )
 
        if not isinstance(response_text, str):
            raise ValueError("LLM returned an invalid response format.")
 
        response_text = response_text.strip()
 
        # 8. Extract JSON if the model adds code fences
        if "```" in response_text:
            response_text = response_text.split("```")[1].strip()
 
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()
 
        # Handle any text surrounding the JSON object
        start = response_text.find("{")
        end = response_text.rfind("}")
 
        if start == -1 or end == -1:
            raise ValueError("LLM did not return valid JSON.")
 
        response_text = response_text[start:end + 1]
 
        result = json.loads(response_text)
 
        # 9. Validate summary points
        summary_points = result.get("summary_points")
 
        if (
            not isinstance(summary_points, list)
            or len(summary_points) != 5
            or not all(
                isinstance(point, str) and point.strip()
                for point in summary_points
            )
        ):
            return {
                "errors": [
                    "Summarizer: LLM did not return exactly 5 valid summary points."
                ]
            }
 
        # 10. Validate source URLs
        generated_sources = result.get("sources", [])
 
        if not isinstance(generated_sources, list):
            generated_sources = []
 
        allowed_urls = set(available_urls)
 
        sources = list(dict.fromkeys(
            url.strip()
            for url in generated_sources
            if isinstance(url, str) and url.strip() in allowed_urls
        ))
 
        # If the model omitted sources, retain the supplied
        # research URLs rather than losing the references.
        if not sources:
            sources = available_urls
 
        logger.info(
            "Summarization successful: %d points and %d sources",
            len(summary_points),
            len(sources)
        )
 
        # 11. Return only fields changed by this agent
        return {
            "summary_points": [
                point.strip() for point in summary_points
            ],
            "sources": sources
        }
 
    except Exception as e:
 
        logger.exception("Summarizer Agent failed")
 
        return {
            "errors": [
                f"Summarizer: {str(e)}"
            ]
        }
 
 
if __name__ == "__main__":
    print("Summarizer Agent module loaded successfully.")