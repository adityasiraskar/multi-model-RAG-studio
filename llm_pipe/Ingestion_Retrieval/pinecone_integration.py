import os
import time
from dotenv import load_dotenv
from tqdm import tqdm
import uuid
import json
import fitz  # PyMuPDF
from typing import List, Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Import our custom PDF parser
from .pdf_parser import parse_pdf

# Import embedding model
from sentence_transformers import SentenceTransformer

# Import Pinecone
from pinecone import Pinecone, ServerlessSpec

# Import our updated chunker
from .document_chunker import process_documents

# Configuration
PINECONE_API_KEY = os.getenv("api_key") or os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = "all-mpnet-base-v2"
DIMENSION = 768  # mpnet's embedding size
INDEX_NAME = "document-analysis"

# Company to PDF mapping - maps company names to their respective PDF files
COMPANY_PDF_MAPPING = {
       "enersys": ["EnerSys-2017-10K.pdf", "EnerSys-2023-10K.pdf"],
       "amazon": ["Amazon10k2022.pdf"],
       "apple": ["Apple_10-K-2021.pdf"],
       "nvidia": ["Nvidia.pdf"],
      "tesla": ["Tesla.pdf"],
      "lockheed": ["Lockheed_martin_10k.pdf"],
      "advent": ["Advent_Technologies_2022_10K.pdf"],
     "transdigm": ["TransDigm-2022-10K.pdf"]
}

def init_pinecone(api_key, index_name, dimension):
    """
    Initialize Pinecone client and create index if needed
    
    Args:
        api_key: Pinecone API key
        index_name: Name of the index to create
        dimension: Vector dimension
        
    Returns:
        Pinecone client and index object
    """
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: {index_name}")
        # Change this to use a supported region for free plan
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1") #'us-east1'
        )
        # Wait for index to be ready 
        time.sleep(60)
        
    # Connect to the index
    index = pc.Index(index_name)
    print(f"Connected to Pinecone index: {index_name}")
    
    return pc, index

def save_pdf_page_images(pdf_path, output_dir="page_images"):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_image_paths = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        image_path = os.path.join(output_dir, f"{os.path.basename(pdf_path).replace('.pdf','')}_page_{page_num + 1}.png")
        pix.save(image_path)
        page_image_paths[str(page_num + 1)] = image_path  # Save as string for consistent key use

    doc.close()
    return page_image_paths


def prepare_metadata(chunk_text, chunk_metadata, pdf_path):
    """
    Prepare metadata for a chunk
    
    Args:
        chunk_text: Text content of the chunk
        chunk_metadata: Metadata from the chunking process
        pdf_path: Path to the source PDF
        
    Returns:
        Dictionary of metadata
    """
    metadata = {
        "source": pdf_path,
        "document_id": os.path.basename(pdf_path).replace(".pdf", ""),
        "text_snippet": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
    }
    
    # Add any additional metadata from the chunking process
    if chunk_metadata:
        metadata.update(chunk_metadata)
    
    # Ensure page number is included (required for better retrieval)
    if "page" in chunk_metadata:
        metadata["page_number"] = chunk_metadata["page"]
    
    # Add special handling for table data
    if chunk_metadata and chunk_metadata.get("is_table", False):
        # Extract and store table-specific metadata that will help with queries
        metadata["content_type"] = "table"
        if "table_id" in chunk_metadata:
            metadata["table_id"] = chunk_metadata["table_id"]
    
    # Add content type if not already set
    if "content_type" not in metadata:
        metadata["content_type"] = "text"
        
    return metadata

class Document:
    """Simple document class to match the expected format for process_documents"""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def process_and_ingest_pdf(index, embeddings_model, pdf_path, namespace, chunk_size=700, chunk_overlap=150):
    """
    Process and ingest a PDF document into Pinecone using our custom parser
    
    Args:
        index: Pinecone index object
        embeddings_model: Model to use for generating embeddings
        pdf_path: Path to the PDF file
        namespace: Namespace to store the vectors in
        chunk_size: Size of chunks to split the document into
        chunk_overlap: Overlap between chunks
    """
    print(f"Processing {pdf_path} for namespace {namespace}...")
    
    #New
    image_map = save_pdf_page_images(pdf_path)
    JSON_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "jsons")
    os.makedirs(JSON_DIR, exist_ok=True)

    json_filename = os.path.basename(pdf_path).replace(".pdf", "_images.json")
    json_path = os.path.join(JSON_DIR, json_filename)
    with open(json_path, "w") as f:
        json.dump(image_map, f)

    # # Use our custom PDF parser
    # extracted_content = parse_pdf(pdf_path)

    combined_text, page_text_map = parse_pdf(pdf_path)

    # Create a Document object to match expected format for process_documents
    document = Document(
        page_content=combined_text,
        metadata={"source": pdf_path}
    )


    # Process the document through our updated chunking pipeline
    chunks = process_documents(
        [document],  # Pass as a list of documents
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        strategy="recursive"
    )
    
    print(f"Created {len(chunks)} chunks from {pdf_path}")
    
    # Count tables vs. text chunks for reporting
    table_chunks = sum(1 for chunk in chunks if chunk.metadata.get("is_table", False))
    text_chunks = len(chunks) - table_chunks
    print(f"Text chunks: {text_chunks}, Table chunks: {table_chunks}")
    
    # Process chunks in batches
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(chunks))
        current_batch = chunks[start_idx:end_idx]
        
        # Create embeddings for the batch
        texts = [chunk.page_content for chunk in current_batch]
        metadatas = [prepare_metadata(chunk.page_content, chunk.metadata, pdf_path) for chunk in current_batch]
        ids = [f"{namespace}-{chunk.metadata.get('page', '0')}-{idx}-{str(uuid.uuid4())[:8]}" 
               for idx, chunk in enumerate(current_batch)]
        
        # Generate embeddings
        embeddings = embeddings_model.encode(texts, convert_to_numpy=True).tolist()
        
        # Prepare vectors for upsert
        vectors = [
            {"id": id, "values": embedding, "metadata": metadata}
            for id, embedding, metadata in zip(ids, embeddings, metadatas)
        ]
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors, namespace=namespace)
        
        print(f"Batch {batch_idx+1}/{total_batches} uploaded to namespace {namespace}")

def get_index_stats(index):
    """
    Get statistics about the index
    
    Args:
        index: Pinecone index object
        
    Returns:
        Statistics about the index
    """
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")
    return stats

def search_by_page(index, embeddings_model, query, namespace, page_number=None, top_k=5):
    """
    Search for documents by query with optional page filter
    
    Args:
        index: Pinecone index object
        embeddings_model: Model to use for generating embeddings
        query: Search query
        namespace: Namespace to search in
        page_number: Optional page number to filter results by
        top_k: Number of results to return
        
    Returns:
        Search results
    """
    # Generate embedding for the query
    query_embedding = embeddings_model.encode(query, convert_to_numpy=True).tolist()
    
    # Create filter for page number if specified
    filter_dict = {}
    if page_number is not None:
        filter_dict["page_number"] = str(page_number)
    
    # Perform the search
    results = index.query(
        namespace=namespace,
        vector=query_embedding,
        filter=filter_dict if filter_dict else None,
        top_k=top_k,
        include_metadata=True
    )
    
    return results

def ingest_all_documents():
    """Ingest all documents into Pinecone with appropriate namespaces"""
    # Initialize Pinecone and create embeddings model
    pc, index = init_pinecone(
        api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME,
        dimension=DIMENSION
    )
    
    # Initialize embedding model
    embeddings_model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Process each company and its PDFs
    for company, pdf_files in COMPANY_PDF_MAPPING.items():
        print(f"\nProcessing {company} documents...")
        PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "pdfs")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            process_and_ingest_pdf(
                index=index,
                embeddings_model=embeddings_model,
                pdf_path=pdf_path,
                namespace=company
            )

    # Get final stats
    get_index_stats(index)
    print("\nAll documents processed and ingested successfully!")

if __name__ == "__main__":
    ingest_all_documents()