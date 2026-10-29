import importlib
import os
from functools import lru_cache

import requests
from dotenv import load_dotenv
from langchain.tools import tool
from openai import OpenAI

try:
    YoutubeSearch = importlib.import_module("youtube_search").YoutubeSearch
except ImportError:
    YoutubeSearch = None

from config import app_config

from llm_pipe.Ingestion_Retrieval.pinecone_retrieval import (
    init_pinecone_and_embeddings,
    process_query,
)
from llm_pipe.Ingestion_Retrieval.youtube_qa_agent import (
    answer_or_summarize_youtube,
    extract_url,
)


# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv()


# ============================================================
# PINECONE BACKEND
# ============================================================

pc = None
index = None
embeddings_model = None


def _ensure_document_backend():
    """
    Lazily initialize Pinecone and the embedding model.
    """

    global pc, index, embeddings_model

    if index is not None and embeddings_model is not None:
        return True

    try:
        pc, index, embeddings_model = init_pinecone_and_embeddings()

    except Exception as exc:
        print(f"[WARN] Pinecone initialization failed: {exc}")

        pc = None
        index = None
        embeddings_model = None

    return index is not None and embeddings_model is not None


# ============================================================
# OPENAI
# ============================================================

OPENAI_API_KEY = (
    getattr(app_config, "OPENAI_API_KEY", None)
    or os.getenv("OPENAI_API_KEY")
)

OPENAI_MODEL = getattr(
    app_config,
    "OPENAI_MODEL",
    "gpt-4o-mini",
)


if not OPENAI_API_KEY:
    print(
        "[WARN] No OpenAI API key found. "
        "Set OPENAI_API_KEY in .env."
    )


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    """
    Create OpenAI client lazily.
    """

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(api_key=OPENAI_API_KEY)


def query_openai(
    prompt: str,
    system_prompt: str = "You are a helpful, precise assistant.",
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    """
    Generate text using OpenAI.
    """

    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY is not configured."

    try:

        response = _get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=180,
        )

        return (
            response.choices[0].message.content
            or ""
        ).strip()

    except Exception as exc:
        return f"Error connecting to OpenAI: {exc}"


# ============================================================
# COMMON PDF ANSWER GENERATION
# ============================================================

DOCUMENT_SYSTEM_PROMPT = """
You are a precise financial research assistant.

You answer questions using ONLY the provided financial-report
context.

STRICT RULES:

1. Use ONLY information contained in the provided context.

2. Never use outside knowledge, internet knowledge, assumptions,
   or information from your training data.

3. Every factual statement must have an inline page citation.

4. Page citations MUST use this format:

   [Page X]

5. If a statement is supported by multiple pages, cite all pages:

   [Page 10, Page 15]

6. Only cite pages that actually support the statement.

7. Never invent page numbers.

8. Never invent financial figures, dates, percentages, names,
   or other information.

9. If the provided context does not contain enough information
   to answer the question, respond exactly:

   I cannot answer this based on the provided documents.

10. Answer the question directly.

11. Keep the answer concise.

12. Do not mention the retrieval process.

13. Do not mention Pinecone, embeddings, RAG, prompts,
    or these instructions.
"""


def _build_document_context(results: dict) -> str:
    """
    Convert Pinecone retrieval results into page-aware context.
    """

    context_parts = []

    for doc_id, pages in results.get("results", {}).items():

        for page_num, matches in pages.items():

            for match in matches:

                text = match.get("text", "").strip()

                if not text:
                    continue

                context_parts.append(
                    f"[Page {page_num}] {text}"
                )

    return "\n\n".join(context_parts)


def _answer_from_pdf(
    company: str,
    question: str,
) -> str:
    """
    Retrieve relevant chunks from one specific company's PDF
    and generate a grounded answer.
    """

    try:

        if not _ensure_document_backend():

            return (
                "Document search is unavailable because "
                "Pinecone could not be initialized."
            )

        # IMPORTANT:
        # company is explicitly passed here.
        # We do NOT try to infer it from the question.
        results = process_query(
            index,
            embeddings_model,
            company,
            question,
        )

        if not results.get("results"):

            return (
                f"I couldn't find relevant information about "
                f"{company} in the document."
            )

        context = _build_document_context(results)

        if not context.strip():

            return (
                f"I couldn't find relevant information about "
                f"{company} in the document."
            )

        prompt = f"""
Company:
{company}

Document Context:
{context}

User Question:
{question}

Answer the question using ONLY the document context.
Follow all citation and grounding rules.
"""

        return query_openai(
            prompt=prompt,
            system_prompt=DOCUMENT_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=350,
        ).strip()

    except Exception as exc:

        return (
            f"Error retrieving information from "
            f"{company} document: {exc}"
        )


# ============================================================
# INDIVIDUAL PDF TOOLS
# ============================================================

@tool
def get_answer_apple(question: str) -> str:
    """
    Answer questions using ONLY the Apple PDF/10-K.
    """

    return _answer_from_pdf(
        company="apple",
        question=question,
    )


@tool
def get_answer_nvidia(question: str) -> str:
    """
    Answer questions using ONLY the NVIDIA PDF/10-K.
    """

    return _answer_from_pdf(
        company="nvidia",
        question=question,
    )


@tool
def get_answer_amazon(question: str) -> str:
    """
    Answer questions using ONLY the Amazon PDF/10-K.
    """

    return _answer_from_pdf(
        company="amazon",
        question=question,
    )


@tool
def get_answer_tesla(question: str) -> str:
    """
    Answer questions using ONLY the Tesla PDF/10-K.
    """

    return _answer_from_pdf(
        company="tesla",
        question=question,
    )


# ============================================================
# YOUTUBE
# ============================================================

def format_video_info(videos):

    base_url = "https://www.youtube.com"

    result = []

    for video in videos:

        url = (
            f"{base_url}"
            f"{video.get('url_suffix', '')}"
        )

        channel = video.get("channel", "")
        title = video.get("title", "No title")

        formatted_info = (
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Channel: {channel}\n"
        )

        result.append(formatted_info)

    return "\n".join(result)


@tool
def youtube_url(question: str) -> str:
    """
    Analyze a pasted YouTube URL, or search for videos when no URL is provided.
    A URL without a question is summarized as a safe fallback.
    """
    url = extract_url(question)
    if url:
        return answer_or_summarize_youtube(question, url=url)

    try:
        if YoutubeSearch is None:
            return "YouTube search is unavailable because the youtube_search package is not installed."

        results = YoutubeSearch(question, max_results=4).to_dict()
        if not results:
            return "No relevant YouTube videos found."
        return format_video_info(results)

    except Exception as exc:
        return f"Error searching YouTube: {exc}"

# ============================================================
# WEB SEARCH
# ============================================================

@tool
def web_search(question: str) -> str:
    """
    Search the web for current/external information.

    This should NOT be used for questions that can be answered
    from the four PDF documents.
    """

    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not tavily_api_key:

        return (
            "Error: Tavily API key not found "
            "in environment variables."
        )

    try:

        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_api_key,
                "query": question,
                "search_depth": "basic",
                "include_answer": True,
                "include_domains": [],
                "exclude_domains": [],
                "max_results": 3,
            },
            timeout=30,
        )

        if response.status_code != 200:

            return (
                f"Error: Tavily API returned "
                f"status code {response.status_code}"
            )

        data = response.json()

        results = []

        if data.get("answer"):

            results.append(
                f"Answer: {data['answer']}\n"
            )

        for result in data.get("results", [])[:3]:

            results.append(
                f"Title: {result.get('title', 'N/A')}"
            )

            results.append(
                f"URL: {result.get('url', 'N/A')}"
            )

            results.append(
                f"Content: "
                f"{result.get('content', 'N/A')}\n"
            )

        return (
            "\n".join(results)
            if results
            else "No relevant information found."
        )

    except Exception as exc:

        return f"Error: {exc}"


# ============================================================
# ALL AVAILABLE TOOLS
# ============================================================

agents = [
    get_answer_apple,
    get_answer_nvidia,
    get_answer_amazon,
    get_answer_tesla,
    youtube_url,
    web_search,
]