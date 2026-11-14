import os

import json

import logging

import re

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from state import NewsState

load_dotenv()

logger = logging.getLogger("summarizer_agent")


def summarizer_node(state: NewsState) -> dict:
    try:

        research_results = state.get("research_results", [])

        if not research_results:
            return {"errors": ["Summarizer: No research results received."]}

        valid_results = [

            item for item in research_results

            if isinstance(item, dict)

               and item.get("title")

               and item.get("url")

               and item.get("content")

        ]

        if not valid_results:
            return {"errors": ["Summarizer: No valid research articles found."]}

        sources = list(dict.fromkeys(

            item["url"] for item in valid_results

        ))

        articles_text = ""

        for index, article in enumerate(valid_results, start=1):
            articles_text += f"""

Article {index}

Title: {article['title']}

URL: {article['url']}

Content: {article['content']}

"""

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            return {"errors": ["Summarizer: OPENROUTER_API_KEY is missing."]}

        model_name = os.getenv(

            "OPENROUTER_MODEL",

            "openai/gpt-5-mini"

        )

        llm = ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-5-mini"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_tokens=1200# makes the output more consistent run to run
        )


        prompt = f"""

You are an expert technology news summarizer.

Analyze the research articles provided below.

Instructions:

1. Generate exactly FIVE summary points.

2. Use simple, clear and professional English.

3. Focus on important facts and developments.

4. Do not invent information.

5. Avoid repeating information.

6. Use only the provided research articles.

Return ONLY valid JSON in this format:

{{

    "summary_points": [

        "Point 1",

        "Point 2",

        "Point 3",

        "Point 4",

        "Point 5"

    ]

}}

RESEARCH ARTICLES:

{articles_text}

"""

        response = llm.invoke(prompt)

        response_text = response.content

        if not isinstance(response_text, str):
            raise ValueError("Unexpected LLM response format.")

        cleaned_response = re.sub(

            r"^```(?:json)?\s*|\s*```$",

            "",

            response_text.strip()

        )

        result = json.loads(cleaned_response)

        summary_points = result.get("summary_points", [])

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

                    "Summarizer: LLM did not return exactly 5 valid points."

                ]

            }

        summary_points = [

            point.strip() for point in summary_points

        ]


        logger.info("Successfully generated 5 summary points.")

        return {

            "summary_points": summary_points,

            "sources": sources

        }

    except Exception as e:

        logger.exception("Summarizer Agent failed.")

        return {

            "errors": [f"Summarizer: {str(e)}"]

        }