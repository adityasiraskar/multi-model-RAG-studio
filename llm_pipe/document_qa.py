"""Shared document-QA generation used by the Gradio app and evaluations."""

from collections.abc import Mapping

import requests

from config import app_config


# def generate_answer_with_ollama(context: str, query: str, doc_page_mapping: Mapping) -> str:
#     """Generate an answer using the application's existing Ollama prompt."""
#     page_references = "".join(
#         f"- Document '{doc_id}' has relevant information on pages: {', '.join(map(str, pages))}\n"
#         for doc_id, pages in doc_page_mapping.items()
#     )

#     prompt = f"""You are a helpful assistant that answers questions about financial reports and SEC filings.
# Answer the user's question based solely on the provided context.

# IMPORTANT INSTRUCTIONS:
# 1. Always cite the specific page numbers when referencing information in your answer.
# 2. Format page citations as [Page X] within your answer text.
# 3. If information comes from multiple pages, cite each relevant page.
# 4. If you don't know the answer based on the context, just say so.
# 5. Don't make up information.

# The following pages contain relevant information for this query:
# {page_references}

# Context:
# {context}

# User Question:
# {query}

# Answer (remember to cite specific page numbers in your response):"""

#     try:
#         response = requests.post(
#             f"{app_config.OLLAMA_API_BASE}/generate",
#             json={
#                 "model": app_config.OLLAMA_MODEL,
#                 "prompt": prompt,
#                 "stream": False,
#                 "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 512},
#             },
#             timeout=180,
#         )
#         if response.status_code == 200:
#             return response.json().get("response", "")
#         return f"Error: Ollama returned status code {response.status_code}"
#     except Exception as exc:
#         return f"Error connecting to Ollama: {exc}"

from typing import Mapping
from openai import OpenAI

client = OpenAI(
    api_key=app_config.OPENAI_API_KEY,
    base_url=getattr(app_config, "OPENAI_API_BASE", None),  # None => default OpenAI endpoint
)


def generate_answer_with_openai(context: str, query: str, doc_page_mapping: Mapping) -> str:
    """Generate an answer using OpenAI's chat completion API, with a prompt tuned
    for RAG evaluation metrics: faithfulness, answer relevancy, context precision/recall,
    and citation accuracy."""

    page_references = "".join(
        f"- Document '{doc_id}': pages {', '.join(map(str, pages))}\n"
        for doc_id, pages in doc_page_mapping.items()
    )

    system_prompt = """You are a precise financial research assistant that answers questions about \
financial reports and SEC filings using ONLY the context provided to you.

STRICT RULES (do not violate these):
1. Use ONLY information present in the provided context. Never use outside knowledge, \
prior training data, or assumptions about companies/financials not stated in the context.
2. Every factual claim in your answer must be immediately followed by a page citation in \
the format [Page X]. If a claim draws on multiple pages, cite all of them, e.g. [Page 12, Page 15].
3. Only cite pages that you actually used to support a claim. Do not cite a page just because \
it appeared in the candidate list — ignore candidate pages that turn out to be irrelevant.
4. If the context contains relevant information spread across multiple pages, synthesize across \
ALL of them rather than answering from a single page.
5. If the answer is not contained in the context, respond exactly with: \
"I cannot answer this based on the provided documents." Do not guess or partially answer.
6. Do not fabricate figures, dates, names, or page numbers under any circumstances.
7. Be direct: answer the question in the first sentence, then support it with cited evidence. \
Do not include unrelated background information.
8. Keep the answer concise and free of filler phrases like "based on the context provided."
"""

    user_prompt = f"""Candidate pages retrieved for this query (for reference only — verify relevance yourself):
{page_references}

Context:
{context}

Question:
{query}

Answer the question following all rules above. Cite pages inline as you go."""

    try:
        response = client.chat.completions.create(
            model=app_config.OPENAI_MODEL,  # e.g. "gpt-4o" or "gpt-4o-mini"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,   # lower temp = more faithful/deterministic for factual QA
            top_p=0.9,
            max_tokens=250,
            timeout=180,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"Error connecting to OpenAI: {exc}"