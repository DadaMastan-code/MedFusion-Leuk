"""
PDF blood report extraction using pdfplumber.
Handles both tabular and free-text CBC layouts.
"""

import pdfplumber
import fitz  # PyMuPDF fallback
from pathlib import Path
from typing import Any


def extract_text_pdfplumber(pdf_path: str | Path) -> str:
    """Extract all text from a PDF using pdfplumber (preferred)."""
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)

            # Also extract any tables as plain text rows
            for table in (page.extract_tables() or []):
                for row in table:
                    clean = [str(cell) if cell else "" for cell in row]
                    text_chunks.append("\t".join(clean))
    return "\n".join(text_chunks)


def extract_text_pymupdf(pdf_path: str | Path) -> str:
    """Fallback extraction using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    return "\n".join(page.get_text() for page in doc)


def extract_text(pdf_path: str | Path) -> str:
    """Try pdfplumber; fall back to PyMuPDF if extraction is empty."""
    text = extract_text_pdfplumber(pdf_path).strip()
    if not text:
        text = extract_text_pymupdf(pdf_path).strip()
    return text
