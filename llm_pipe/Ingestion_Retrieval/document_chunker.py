import uuid
import re

# LangChain imports
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document

# Embedding models used in this pipeline (e.g. mpnet) have a limited max
# sequence length. A large financial table (balance sheet, 40+ line items)
# stored as a single chunk would get silently truncated on encode, so any
# table chunk longer than this many characters gets split into row-groups.
MAX_TABLE_CHUNK_CHARS = 1500


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
    Each section's content includes the "### Table N (Page P)" header line,
    an optional caption line (added by the parser), and the markdown table.
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
                "content": f"### Table {table_id} (Page {page_num})\n{table_content}",
                "start": match.start(),
                "end": match.end()
            })

    return table_sections


def split_large_table(table_content, max_chars=MAX_TABLE_CHUNK_CHARS):
    """
    Split an oversized table chunk into smaller pieces so nothing gets
    truncated at embedding time. Every piece repeats the table's header
    line, caption (if any), and the markdown header/separator row, so each
    piece is independently understandable and searchable on its own -
    important since a query might only match rows from the middle of a
    large financial table.
    """
    lines = table_content.strip().split("\n")

    # The markdown table itself starts at the first line beginning with "|"
    table_start_idx = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("|")), None
    )

    if table_start_idx is None or len(table_content) <= max_chars:
        return [table_content]

    prefix_lines = lines[:table_start_idx]  # "### Table N (Page P)" + caption
    header_line = lines[table_start_idx]
    separator_line = lines[table_start_idx + 1] if table_start_idx + 1 < len(lines) else ""
    data_lines = [line for line in lines[table_start_idx + 2:] if line.strip()]

    if not data_lines:
        return [table_content]

    prefix = "\n".join(prefix_lines)
    header_block = "\n".join([header_line, separator_line])
    fixed_overhead = len(prefix) + len(header_block) + 2

    parts = []
    current_rows = []
    current_len = fixed_overhead

    for row in data_lines:
        if current_rows and current_len + len(row) + 1 > max_chars:
            parts.append("\n".join([prefix, header_block] + current_rows))
            current_rows = []
            current_len = fixed_overhead
        current_rows.append(row)
        current_len += len(row) + 1

    if current_rows:
        parts.append("\n".join([prefix, header_block] + current_rows))

    return parts if parts else [table_content]


def chunk_documents(docs, chunk_size=500, chunk_overlap=100, strategy="recursive"):
    """
    Split documents into smaller chunks with metadata.
    Tables are automatically detected and processed as whole units (split
    further only if they exceed the embedding model's safe input size).
    Page numbers are included with each chunk.
    """
    chunked_docs = []

    for doc in docs:
        content = doc.page_content

        # Derive a stable doc_id once per source document, instead of a new
        # random uuid on every metadata.get() call (which previously meant
        # every chunk of the same PDF got a different, meaningless doc_id).
        stable_doc_id = (
            doc.metadata.get("doc_id")
            or doc.metadata.get("source")
            or str(uuid.uuid4())
        )

        # Extract page and table sections
        page_sections = extract_page_sections(content)
        table_sections = extract_table_sections(content)

        # Process tables first (each table as one or more chunks)
        for table in table_sections:
            table_parts = split_large_table(table["content"])
            for part_idx, part_content in enumerate(table_parts):
                chunked_docs.append(Document(
                    page_content=part_content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": str(uuid.uuid4()),
                        "content_type": "table",
                        "is_table": True,
                        "table_id": table["table_id"],
                        "table_part": part_idx,
                        "total_table_parts": len(table_parts),
                        "page": table["page_num"],
                        "doc_id": stable_doc_id,
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
                    "doc_id": stable_doc_id,
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

    Tables are automatically detected and processed as whole units, split
    only when they exceed a safe size for the embedding model.
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
