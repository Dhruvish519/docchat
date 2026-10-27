"""Step 1 of the RAG pipeline: read a PDF and cut it into small, overlapping chunks.

Every chunk remembers which file and which page it came from, so the
chatbot can later cite its sources ("report.pdf, page 4").
"""

import re
from dataclasses import dataclass, asdict

import pymupdf  # PyMuPDF


@dataclass
class Chunk:
    text: str
    file_name: str
    page: int  # 1-based page number

    def to_dict(self):
        return asdict(self)


def extract_pages(pdf_bytes: bytes) -> list[str]:
    """Return the plain text of every page in the PDF."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return [page.get_text("text") for page in doc]


def _clean(text: str) -> str:
    text = text.replace("­", "")              # soft hyphens
    text = re.sub(r"-\n(\w)", r"\1", text)         # re-join hyphenated words
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into windows of ~chunk_size characters that overlap by ~overlap.

    Cuts are moved back to the nearest paragraph / sentence / word boundary so
    that we don't chop words in half. The overlap keeps sentences that sit on a
    boundary available in both neighbouring chunks.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep)
                if cut > chunk_size * 0.5:      # don't make tiny chunks
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        # move start forward to a word boundary
        nxt = text.find(" ", start)
        if 0 <= nxt - start < 30:
            start = nxt + 1
    return chunks


def pdf_to_chunks(pdf_bytes: bytes, file_name: str,
                  chunk_size: int = 1000, overlap: int = 200) -> tuple[list[Chunk], int]:
    """Read a PDF and return (chunks, number_of_pages)."""
    pages = extract_pages(pdf_bytes)
    chunks = []
    for page_no, page_text in enumerate(pages, start=1):
        for piece in split_text(_clean(page_text), chunk_size, overlap):
            if len(piece) >= 20:                 # skip page numbers, headers etc.
                chunks.append(Chunk(piece, file_name, page_no))
    return chunks, len(pages)
