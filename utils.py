import os
import json
import logging
import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, asdict
import base64

# --- Dependency Imports with Graceful Fallbacks ---
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    import markdown
except ImportError:
    markdown = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

import io
try:
    from PIL import Image
except ImportError:
    Image = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def _cell_text(cell_html: str) -> str:
    """Strip tags and return bare visible text."""
    return re.sub(r'<[^>]+>', '', cell_html).strip()

def _cells_of(row_html: str) -> list:
    """Return list of raw cell HTML inner contents."""
    return re.findall(
        r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL | re.I
    )

def _last_cell_text(row_html: str) -> str:
    """Plain text of the last <td>/<th> in row_html."""
    cells = _cells_of(row_html)
    return _cell_text(cells[-1]) if cells else ''

def _first_cell_text(row_html: str) -> str:
    """Plain text of the first <td>/<th> in row_html."""
    cells = _cells_of(row_html)
    return _cell_text(cells[0]) if cells else ''

def _extract_serial(text: str):
    """Return the integer if text is a bare integer, else None."""
    m = re.match(r'^\s*(\d+)\s*$', text.strip())
    return int(m.group(1)) if m else None

def _is_incomplete_end(text: str) -> bool:
    """True when text ends with a dangling dash or colon."""
    t = text.strip()
    if not t:
        return False
    if t.endswith('-') or t.endswith(':'):
        return True
    if re.search(r'\w+-\s*$', t):
        return True
    return False

_NOISE_WORDS = {
    "kg", "nos", "qty", "no", "set", "lot", "pcs",
    "each", "unit", "units", "ltr", "mtr", "mm", "cm",
}

def _is_orphan_row(row_html: str) -> tuple:
    """Return (is_orphan, last_cell_text)."""
    cells = _cells_of(row_html)
    if len(cells) < 2:
        return False, ''
    leading = cells[:-1]
    last_text = _cell_text(cells[-1])
    if len(last_text) < 20:
        return False, ''
    meaningful = [
        t for t in (_cell_text(c) for c in leading)
        if len(t) > 3
        and t.lower() not in _NOISE_WORDS
        and not re.match(r'^[\d\.\-\|\s]+$', t)
    ]
    if len(meaningful) > 1:
        return False, ''
    return True, last_text

_INCOMPLETE_END_RE = re.compile(
    r'(<t[dh][^>]*>)(.*?)(</t[dh]>)(\s*(?:</tr>)?\s*)$',
    re.DOTALL | re.I,
)

def _apply_merge(prev_html: str, extra_text: str) -> str:
    """Append extra_text to the last cell of prev_html."""
    def _inject(m: re.Match, _extra: str = extra_text) -> str:
        return m.group(1) + m.group(2).rstrip() + _extra + m.group(3) + m.group(4)
    return _INCOMPLETE_END_RE.sub(_inject, prev_html, count=1)


# =============================================================
# OCR Data Models (From V4.5)
# =============================================================

@dataclass
class HTMLCleaner:
    @staticmethod
    def clean_to_text(html_content: str) -> str:
        if not BeautifulSoup:
            no_tags = re.sub(r"<[^>]+>", " ", html_content)
            no_entities = re.sub(r"&[a-zA-Z#0-9]+;", " ", no_tags)
            return re.sub(r"\s+", " ", no_entities).strip()
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get clean text
        text = soup.get_text(separator=" ")
        
        # Normalize whitespace and strip out zero-width/useless characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        
        return text.strip()

