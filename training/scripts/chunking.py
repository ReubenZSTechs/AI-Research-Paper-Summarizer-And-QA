import fitz
import re
from dataclasses import dataclass, field
from typing import TypedDict, List, Optional


CONFIG = {
    'SECTION_KEYWORDS': [
        'abstract', 'introduction', 'background', 'related work', 'methodology', 'methods', 'experiments', 'results', 'discussion', 'conclusion', 'references', 'appendix'
    ]
}


@dataclass
class Chunk:
    chunk_id: str
    level: str
    text: str
    paper_id: str
    page_num: int
    font_size: float
    section_heading: str = ""
    section_idx: int = 0
    section_type: str = "body"
    chunk_type: str = "text"
    parent_section_id: Optional[str] = None
    citations: list = field(default_factory=list)
    char_count: int = 0


def classify_section(heading: str):
    heading = heading.lower().strip()

    for keyword in CONFIG['SECTION_KEYWORDS']:
        if keyword in heading:
            return heading.replace(" ", "_")
        
    return 'body'


def extract_chunks(pdf_path: str, paper_id: str):
    try:
        doc = fitz.open(pdf_path)
    except FileNotFoundError as e:
        print(f"File not found. Got {pdf_path}\n\nError:\n{e}")

    chunks = []
    current_section_idx = 0
    current_heading = "preamble"
    current_section_type = "body"
    paragraph_buffer = []
    chunk_counter = 0

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")['blocks']

        for block in blocks:
            # Got images
            if block['type'] != 0:
                continue

            for line in block['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    size = round(span['size'], 1)
                    bold = "bold" in span['font'].lower()

                    if not text:
                        continue

                    # --- Heading detection ---
                    is_heading = (
                        bold and size >= 10.5 and len(text) < 120
                        and any(kw in text.lower() for kw in CONFIG['SECTION_KEYWORDS'])
                    )

                    if is_heading:
                        # Flush buffer into a paragraph chunk
                        if paragraph_buffer:
                            full_text = " ".join(paragraph_buffer)
                            section_id = f"{paper_id}_s{current_section_idx}"
                            chunks.append(Chunk(
                                chunk_id=f"{section_id}_p{chunk_counter}",
                                level='paragraph',
                                text=full_text,
                                paper_id=paper_id,
                                page_num=page_num,
                                font_size=size,
                                section_heading=current_heading,
                                section_idx=current_section_idx,
                                section_type=current_section_type,
                                parent_section_id=section_id,
                                char_count=len(full_text),
                            ))
                            paragraph_buffer = []
                            chunk_counter += 1

                        # New section
                        current_section_idx += 1
                        current_heading = text
                        current_section_type = classify_section(text)

                        section_id = f"{paper_id}_s{current_section_idx}"
                        chunks.append(Chunk(
                            chunk_id=section_id,
                            level='section',
                            text=text,
                            paper_id=paper_id,
                            page_num=page_num,
                            font_size=size,
                            section_heading=text,
                            section_idx=current_section_idx,
                            section_type=current_section_type,
                        ))
                    else:
                        paragraph_buffer.append(text)

                        # Flush at ~400 token boundary (approx 1600 chars)
                        if len(" ".join(paragraph_buffer)) > 1600:
                            full_text = " ".join(paragraph_buffer)
                            section_id = f"{paper_id}_s{current_section_idx}"
                            chunks.append(Chunk(
                                chunk_id=f"{section_id}_p{chunk_counter}",
                                level='paragraph',
                                text=full_text,
                                paper_id=paper_id,
                                page_num=page_num,
                                font_size=size,
                                section_heading=current_heading,
                                section_idx=current_section_idx,
                                section_type=current_section_type,
                                parent_section_id=section_id,
                                char_count=len(full_text),
                            ))
                            # Keep last 50-token overlap
                            overlap = " ".join(paragraph_buffer[-3:])
                            paragraph_buffer = [overlap]
                            chunk_counter += 1

    doc.close()
    return chunks


if __name__ == "__main__":
    chunks = extract_chunks(pdf_path=f"training/datasets/processed/pdf/2401.00005.pdf", paper_id='2401.00005')
    print(chunks)