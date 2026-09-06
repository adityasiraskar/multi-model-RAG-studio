"""Single shared pipeline entry point for application and evaluation runs."""

from src.generator import generate
from src.reranker import RerankingRetriever


class RagPipeline:
    def __init__(self, retriever=None):
        self.retriever = retriever or RerankingRetriever()

    def invoke(self, question: str) -> dict:
        documents = self.retriever.invoke(question)
        context = [document.page_content for document in documents]
        answer = generate(question, context)
        return {"query": question, "context": context, "answer": answer}