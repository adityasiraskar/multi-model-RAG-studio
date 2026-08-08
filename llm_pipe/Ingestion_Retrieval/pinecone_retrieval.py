import os
from typing import Dict, List, Any
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Load environment variables from project root first
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("api_key") or os.getenv("PINECONE_API_KEY")
INDEX_NAME = "document-analysis"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Embedding model

def init_pinecone_and_embeddings():
    """Initialize Pinecone and embedding model"""
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists(MODELS_DIR):
            print(f"Creating models directory at {MODELS_DIR}...")
            os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Check if embedding model exists
        embedding_model_path = os.path.join(MODELS_DIR, "embedding_model")
        if os.path.exists(embedding_model_path):
            print("Loading embedding model from disk...")
            embeddings_model = SentenceTransformer(embedding_model_path)
        else:
            print("Embedding model not found locally, downloading...")
            embeddings_model = SentenceTransformer(EMBEDDING_MODEL)
            
            # Save the embedding model for future use
            try:
                print(f"Saving embedding model to {embedding_model_path}...")
                embeddings_model.save(embedding_model_path)
                print("Embedding model saved successfully!")
            except Exception as e:
                print(f"Warning: Failed to save embedding model: {e}")
                # Continue without saving
    except Exception as e:
        print(f"Warning: Error with models directory: {e}")
        print("Continuing with in-memory embedding model only...")
        embeddings_model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Models loaded successfully!")
    return pc, index, embeddings_model

def semantic_search(index, embeddings_model, query, namespace, top_k=5, score_threshold=0.4, max_results=3):
    """
    Perform semantic search with score filtering and result limiting.

    Args:
        index: Pinecone index object
        embeddings_model: Embedding model
        query: Query string
        namespace: Namespace to search within
        top_k: Number of candidates to retrieve from Pinecone
        score_threshold: Minimum similarity score
        max_results: Max results to return after filtering

    Returns:
        Dictionary of results grouped by document_id and page number
    """
    query_embedding = embeddings_model.encode(query, convert_to_numpy=True).tolist()

    response = index.query(
        vector=query_embedding,
        namespace=namespace,
        top_k=top_k,
        include_metadata=True
    )

    # Filter and sort matches
    filtered_matches = sorted(
        [m for m in response.get("matches", []) if m["score"] >= score_threshold],
        key=lambda x: x["score"],
        reverse=True
    )[:max_results]

    # Group by document_id and include page numbers
    grouped_results = defaultdict(lambda: defaultdict(list))
    
    for match in filtered_matches:
        metadata = match.get("metadata", {})
        doc_id = metadata.get("document_id", f"unknown-{match['id']}")
        
        # Extract page number - check both possible keys
        page_number = metadata.get("page_number", metadata.get("page", "unknown"))
        
        # Extract content type
        content_type = metadata.get("content_type", "text")
        
        # Extract table info if available
        table_info = ""
        if content_type == "table" and "table_id" in metadata:
            table_info = f" (Table {metadata['table_id']})"
        
        grouped_results[doc_id][page_number].append({
            "score": match["score"],
            "text": metadata.get("text_snippet", metadata.get("text", "No preview available")),
            "content_type": content_type,
            "table_info": table_info,
            "id": match["id"]
        })

    return grouped_results

def format_documents(results_by_doc: Dict) -> str:
    """
    Format retrieved documents for inclusion in the context, 
    including page number if present.
    """
    if not results_by_doc:
        return "No relevant information found."
    
    formatted_text = ""
    for doc_id, pages in results_by_doc.items():
        formatted_text += f"\n=== Document: {doc_id} ===\n"
        
        for page_num, matches in pages.items():
            for match in matches:
                formatted_text += f"\n--- Page {page_num}{match['table_info']} (score: {match['score']:.2f}) ---\n"
                formatted_text += match["text"].strip() + "\n"
    
    return formatted_text

def get_document_page_mapping(results_by_doc: Dict) -> Dict[str, List[str]]:
    """
    Create a mapping of document IDs to their relevant page numbers.
    
    Args:
        results_by_doc: Dictionary of search results grouped by document_id and page
        
    Returns:
        Dictionary mapping document IDs to lists of page numbers
    """
    doc_page_mapping = {}
    
    for doc_id, pages in results_by_doc.items():
        doc_page_mapping[doc_id] = list(pages.keys())
    
    return doc_page_mapping

def process_query(index, embeddings_model, company: str, query: str) -> Dict:
    """
    Process a query for a specific company and return relevant document chunks
    with their page numbers.
    
    Args:
        index: Pinecone index object
        embeddings_model: Embedding model
        company: Company namespace to search
        query: Query string
        
    Returns:
        Dictionary containing:
        - results: Search results grouped by document_id and page
        - formatted_text: Formatted text for display
        - doc_page_mapping: Mapping of document IDs to page numbers
    """
    if not query.strip():
        return {
            "results": {},
            "formatted_text": "No query provided.",
            "doc_page_mapping": {}
        }
    
    # Search Pinecone for relevant context
    results = semantic_search(
        index=index,
        embeddings_model=embeddings_model,
        query=query,
        namespace=company.lower(),
        top_k=15,
        score_threshold=0.35,
        max_results=5
    )
    
    # Format the results for display
    formatted_text = format_documents(results)
    
    # Create document-to-page mapping
    doc_page_mapping = get_document_page_mapping(results)
    
    return {
        "results": results,
        "formatted_text": formatted_text,
        "doc_page_mapping": doc_page_mapping
    }

def search_by_document_and_page(index, embeddings_model, company: str, 
                               document_id: str, page_number: str, top_k=3):
    """
    Search for content from a specific document and page number.
    
    Args:
        index: Pinecone index object
        embeddings_model: Embedding model
        company: Company namespace to search
        document_id: Document ID to filter by
        page_number: Page number to filter by
        top_k: Number of results to return
        
    Returns:
        List of matching content from the specified page
    """
    # Create the filter with both document_id and page number
    filter_dict = {
        "document_id": document_id
    }
    
    # Add page filter - check both possible key names
    if page_number:
        # We'll try both "page" and "page_number" in the query
        filter_dict["$or"] = [
            {"page": page_number},
            {"page_number": page_number}
        ]
    
    # Query without a specific vector (empty search)
    # This will return matches based only on the filters
    response = index.query(
        namespace=company.lower(),
        vector=[0] * 768,  # Dummy vector since we're filtering by metadata
        filter=filter_dict,
        top_k=top_k,
        include_metadata=True
    )
    
    results = []
    for match in response.get("matches", []):
        metadata = match.get("metadata", {})
        content_type = metadata.get("content_type", "text")
        
        # Add table info if applicable
        table_info = ""
        if content_type == "table" and "table_id" in metadata:
            table_info = f" (Table {metadata['table_id']})"
            
        results.append({
            "text": metadata.get("text_snippet", metadata.get("text", "No preview available")),
            "content_type": content_type,
            "table_info": table_info,
            "id": match["id"]
        })
    
    return results
