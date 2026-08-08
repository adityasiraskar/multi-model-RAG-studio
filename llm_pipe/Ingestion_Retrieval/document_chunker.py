import uuid
import re

# LangChain imports
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document


def create_text_splitter(chunk_size=500, chunk_overlap=100, strategy="recursive"):
    """
    Create a text splitter based on the specified strategy.
    """
    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    elif strategy == "character":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif strategy == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")


def extract_page_sections(content):
    """
    Extract page sections from the content with their page numbers.
    """
    page_pattern = r'\n\n## Page (\d+)\n\n(.*?)(?=\n\n## Page \d+|\n\n### Table|$)'
    page_matches = re.finditer(page_pattern, content, re.DOTALL)
    
    page_sections = []
    for match in page_matches:
        page_num = match.group(1)
        page_content = match.group(2).strip()
        if page_content:  # Only add non-empty pages
            page_sections.append({
                "page_num": page_num,
                "content": page_content,
                "start": match.start(),
                "end": match.end()
            })
            
    return page_sections


def extract_table_sections(content):
    """
    Extract table sections from the content with their page numbers.
    """
    table_pattern = r'\n\n### Table (\d+) \(Page (\d+)\)(.*?)(?=\n\n## Page|\n\n### Table|\Z)'
    table_matches = re.finditer(table_pattern, content, re.DOTALL)
    
    table_sections = []
    for match in table_matches:
        table_id = match.group(1)
        page_num = match.group(2)
        table_content = match.group(3).strip()
        
        if table_content:  # Only add non-empty tables
            table_sections.append({
                "table_id": table_id,
                "page_num": page_num,
                "content": f"### Table {table_id} (Page {page_num}){table_content}",
                "start": match.start(),
                "end": match.end()
            })
            
    return table_sections


def chunk_documents(docs, chunk_size=500, chunk_overlap=100, strategy="recursive"):
    """
    Split documents into smaller chunks with metadata.
    Tables are automatically detected and processed as whole units.
    Page numbers are included with each chunk.
    """
    chunked_docs = []
    
    for doc in docs:
        content = doc.page_content
        
        # Extract page and table sections
        page_sections = extract_page_sections(content)
        table_sections = extract_table_sections(content)
        
        # Process tables first (each table as one chunk)
        for table in table_sections:
            chunked_docs.append(Document(
                page_content=table["content"],
                metadata={
                    **doc.metadata,
                    "chunk_id": str(uuid.uuid4()),
                    "content_type": "table",
                    "is_table": True,
                    "table_id": table["table_id"],
                    "page": table["page_num"],
                    "doc_id": doc.metadata.get("doc_id", str(uuid.uuid4())),
                }
            ))
        
        # Process text sections by page
        text_splitter = create_text_splitter(chunk_size, chunk_overlap, strategy)
        
        for page in page_sections:
            page_content = f"## Page {page['page_num']}\n\n{page['content']}"
            
            # Create document for this page section
            page_doc = Document(
                page_content=page_content,
                metadata={
                    **doc.metadata,
                    "content_type": "text",
                    "is_table": False,
                    "page": page["page_num"],
                    "doc_id": doc.metadata.get("doc_id", str(uuid.uuid4())),
                }
            )
            
            # Split the page into chunks
            page_chunks = text_splitter.split_documents([page_doc])
            
            # Add metadata to text chunks
            for i, chunk in enumerate(page_chunks):
                chunk.metadata.update({
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "total_page_chunks": len(page_chunks),
                    "chunking_strategy": strategy,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                })
                chunked_docs.append(chunk)
    
    # Update total chunks count in metadata
    for i, doc in enumerate(chunked_docs):
        doc.metadata.update({
            "chunk_index_global": i,
            "total_chunks": len(chunked_docs),
        })
    
    return chunked_docs


def process_documents(docs, chunk_size=500, chunk_overlap=100, strategy="recursive"):
    """
    Main processing function for the chunking pipeline.
    
    Tables are automatically detected and processed as whole units.
    Page numbers are included with each chunk.
    
    Args:
        docs: List of Document objects to process
        chunk_size: Size of chunks for regular text
        chunk_overlap: Overlap between chunks for regular text
        strategy: Chunking strategy for regular text
        
    Returns:
        List of Document objects with chunks of both text and tables
    """
    return chunk_documents(docs, chunk_size, chunk_overlap, strategy)
