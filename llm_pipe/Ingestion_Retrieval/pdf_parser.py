import os
import warnings
import pdfplumber
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Suppress UserWarnings from pdfplumber (like CropBox missing warnings)
warnings.filterwarnings("ignore", category=UserWarning, module='pdfplumber')

# How far above a detected table (in PDF points) to look for a caption/title line
MAX_CAPTION_DISTANCE = 40


def convert_table_to_markdown(table_data):
    """
    Convert a raw table (list of rows) extracted by pdfplumber into a markdown table.
    Handles missing/duplicate/None headers safely.
    """
    if not table_data or len(table_data) < 2:
        return ""

    raw_header = table_data[0]
    header = [str(c).strip() if c is not None else "" for c in raw_header]

    # Guarantee non-empty, unique column names so pandas doesn't choke
    seen = {}
    clean_header = []
    for i, h in enumerate(header):
        name = h if h else f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        clean_header.append(name)

    rows = [
        ["" if cell is None else str(cell).strip() for cell in row]
        for row in table_data[1:]
    ]

    try:
        df = pd.DataFrame(rows, columns=clean_header)
    except ValueError:
        # Row width mismatch (ragged table) - pad/truncate rows defensively
        fixed_rows = []
        for row in rows:
            if len(row) < len(clean_header):
                row = row + [""] * (len(clean_header) - len(row))
            elif len(row) > len(clean_header):
                row = row[: len(clean_header)]
            fixed_rows.append(row)
        df = pd.DataFrame(fixed_rows, columns=clean_header)

    markdown = "| " + " | ".join(df.columns) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    for _, row in df.iterrows():
        markdown += "| " + " | ".join(str(cell).strip() for cell in row) + " |\n"
    return markdown


def find_caption_above_table(page, table_bbox, max_distance: float = MAX_CAPTION_DISTANCE) -> Optional[str]:
    """
    Look for a short line of text directly above a table's bounding box.
    Financial reports typically place a title like 'Segment Revenue (in millions)'
    one or two lines above the actual table. Attaching this to the table's
    stored content makes it embed and retrieve far better than raw numbers/markdown.
    """
    x0, top, x1, bottom = table_bbox
    search_top = max(0, top - max_distance)

    if search_top >= top:
        return None

    try:
        strip = page.crop((0, search_top, page.width, top))
        text = strip.extract_text() or ""
    except Exception:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    candidate = lines[-1]
    # Filter out junk: too short to be meaningful, or too long (probably a
    # wrapped paragraph rather than a title)
    if len(candidate) < 3 or len(candidate.split()) > 20:
        return None
    return candidate


def extract_page_text_excluding_tables(page, table_bboxes: List[Tuple[float, float, float, float]]) -> str:
    """
    Extract narrative text for a page while excluding regions already claimed
    by a detected table. This keeps table numbers out of the narrative text
    and keeps narrative sentences (which often also contain numbers) intact,
    instead of relying on a heuristic that strips any line with 3+ numbers.
    """
    cropped_page = page
    for bbox in table_bboxes:
        try:
            cropped_page = cropped_page.outside_bbox(bbox)
        except Exception:
            # Skip a bbox that is degenerate or already outside the current crop
            continue
    text = cropped_page.extract_text() or ""
    return text.strip()


def parse_pdf(file_path: str) -> Tuple[str, Dict[int, str]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    combined_text = ""
    page_text_map = {}  # page number -> text (including its tables)

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages to process: {total_pages}")

        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                # Detect tables geometrically first. Using the same bounding
                # boxes for both text exclusion and table extraction keeps
                # the two extraction paths consistent (no double-counting or
                # silently dropped table rows).
                found_tables = page.find_tables()
                table_bboxes = [t.bbox for t in found_tables]

                page_text_cleaned = extract_page_text_excluding_tables(page, table_bboxes)
                full_page_text = f"## Page {page_number}\n\n{page_text_cleaned}\n"

                combined_text += f"\n\n{full_page_text}"
                page_text_map[page_number] = page_text_cleaned

                for idx, table in enumerate(found_tables):
                    try:
                        table_data = table.extract()
                    except Exception as e:
                        print(f"Warning: Failed to extract table {idx + 1} on page {page_number}: {e}")
                        continue

                    markdown = convert_table_to_markdown(table_data)
                    if not markdown:
                        continue

                    caption = find_caption_above_table(page, table.bbox)
                    caption_line = f"{caption}\n" if caption else ""

                    table_section = (
                        f"\n### Table {idx + 1} (Page {page_number})\n"
                        f"{caption_line}{markdown}\n"
                    )
                    combined_text += table_section
                    page_text_map[page_number] += table_section

            except Exception as e:
                print(f"Warning: Skipping page {page_number} due to error: {e}")

    return combined_text, page_text_map


# Example usage
# if __name__ == "__main__":
#     input_pdf = "test.pdf"
#     result = parse_pdf(input_pdf)
#     print(result)
