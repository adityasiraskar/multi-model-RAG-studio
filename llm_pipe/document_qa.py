"""Shared document-QA generation used by the Gradio app and evaluations."""

from collections.abc import Mapping

from openai import OpenAI

from config import app_config


_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    """Create the OpenAI client lazily so imports work before credentials exist."""

    global _client

    api_key = app_config.OPENAI_API_KEY.strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to .env or the process environment."
        )

    if _client is None:
        _client = OpenAI(api_key=api_key)

    return _client


def generate_answer_with_openai(
    context: str,
    query: str,
    doc_page_mapping: Mapping,
) -> str:
    """
    Generate a document-QA answer using OpenAI.

    The model is instructed to:
    - use only retrieved context
    - cite relevant pages
    - avoid hallucinations
    - synthesize information across pages
    - return a fixed response when the answer is unavailable
    """

    page_references = "".join(
        f"- Document '{doc_id}': pages {', '.join(map(str, pages))}\n"
        for doc_id, pages in doc_page_mapping.items()
    )

    system_prompt = """
You are a precise financial research assistant that answers questions
about financial reports and SEC filings using ONLY the context provided
to you.

STRICT RULES:

1. Use ONLY information present in the provided context.
   Never use outside knowledge, prior training data, or assumptions.

2. Every factual claim in your answer must be immediately followed by
   a page citation in the format [Page X].

3. If a claim is supported by multiple pages, cite all relevant pages.
   Example: [Page 12, Page 15]

4. Only cite pages that actually support the claim.
   Do not cite pages simply because they appear in the candidate page list.

5. If relevant information is present across multiple pages, synthesize
   the information across those pages.

6. If the answer cannot be determined from the provided context,
   respond exactly with:
   "I cannot answer this based on the provided documents."

7. Never fabricate figures, dates, names, facts, or page numbers.

8. Answer the question directly in the first sentence.

9. Keep the answer concise and avoid unnecessary background information.

10. Do not mention these instructions or the retrieval process in your answer.
"""

    user_prompt = f"""
Candidate pages retrieved for this query:
{page_references}

Context:
{context}

Question:
{query}

Answer the question following all rules above.
Cite relevant pages inline.
"""

    try:
        response = _get_openai_client().chat.completions.create(
            model=app_config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.1,
            max_tokens=300,
            timeout=180,
        )

        answer = response.choices[0].message.content

        return answer.strip() if answer else ""

    except Exception as exc:
        return f"Error connecting to OpenAI: {exc}"


# if __name__ == "__main__":
#     test_context = """
#     Apple reported total net sales of $394.3 billion in 2022.
#     """

#     test_mapping = {
#         "Apple 10-K 2022": [31],
#     }

#     test_query = "What were Apple's total net sales in 2022?"

#     print(
#         generate_answer_with_openai(
#             context=test_context,
#             query=test_query,
#             doc_page_mapping=test_mapping,
#         )
#     )