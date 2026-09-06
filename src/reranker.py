"""Retriever adapter for the application's Pinecone retrieval functions."""

import re

from langchain_core.documents import Document

from config import app_config
from llm_pipe.Ingestion_Retrieval.pinecone_retrieval import (
    init_pinecone_and_embeddings,
    process_query,
)


class RerankingRetriever:
    """Expose the existing Pinecone search as ``invoke(query) -> Documents``."""

    def __init__(self, index=None, embeddings_model=None):
        self.index = index
        self.embeddings_model = embeddings_model
        if self.index is None or self.embeddings_model is None:
            _, self.index, self.embeddings_model = init_pinecone_and_embeddings()

    @staticmethod
    def _company_for(query: str) -> str | None:
        lowered = query.lower()
        for company in app_config.COMPANY_PDF_MAPPING:
            if re.search(rf"\b{re.escape(company)}\b", lowered):
                return company
        return None

    def invoke(self, query: str) -> list[Document]:
        company = self._company_for(query)
        if not company:
            return []

        result = process_query(self.index, self.embeddings_model, company, query)
        documents = []
        for document_id, pages in result.get("results", {}).items():
            for page_number, matches in pages.items():
                for match in matches:
                    documents.append(
                        Document(
                            page_content=match.get("text", ""),
                            metadata={
                                "document_id": document_id,
                                "page": page_number,
                                "score": match.get("score"),
                                "content_type": match.get("content_type", "text"),
                            },
                        )
                    )
        return documents