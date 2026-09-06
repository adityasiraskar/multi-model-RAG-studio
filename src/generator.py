"""Evaluation adapter around the application's llm_pipe document generator."""

from collections.abc import Iterable

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from llm_pipe.document_qa import generate_answer_with_openai


def _context_text(context: Iterable[str]) -> str:
    return "\n\n".join(str(item) for item in context) or "No relevant information found."


def generate(question: str, context: Iterable[str]) -> str:
    """Use the same generator as the production llm_pipe document-QA flow."""
    return generate_answer_with_openai(_context_text(context), question, {})


def generate_stream(question: str, context: Iterable[str]):
    """Yield the production generator's non-streaming answer as one chunk."""
    answer = generate(question, context)
    if answer:
        yield answer


def _invoke_with_usage(values: dict) -> AIMessage:
    answer = generate(values["question"], [values.get("context", "")])
    input_tokens = max(len(str(values.get("question", "") + values.get("context", "")).split()), 1)
    output_tokens = max(len(answer.split()), 1)
    return AIMessage(
        content=answer,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


prompt = RunnableLambda(lambda values: values)
llm = RunnableLambda(_invoke_with_usage)