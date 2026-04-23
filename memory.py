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

class TableFingerprint:
    """Represents a unique logical signature for a table."""
    col_count: int
    x_start: float
    width: float
    header_html: str

    def matches(self, other: "TableFingerprint", tolerance_x=20, tolerance_w=30) -> bool:
        if self.col_count != other.col_count:
            return False
        x_match = abs(self.x_start - other.x_start) < tolerance_x
        w_match = abs(self.width - other.width) < tolerance_w
        return x_match and w_match

class DocumentTableMemory:
    """Manages table identities across multiple pages."""
    def __init__(self):
        self.active_fingerprint: Optional[TableFingerprint] = None
        self.last_table_page: int = -1

    def update(self, fingerprint: TableFingerprint, page_num: int):
        self.active_fingerprint = fingerprint
        self.last_table_page = page_num

    def is_continuation(self, new_fingerprint: TableFingerprint, page_num: int) -> bool:
        if not self.active_fingerprint: return False
        if page_num - self.last_table_page > 3: return False
        return self.active_fingerprint.matches(new_fingerprint)

    def get_header_html(self) -> Optional[str]:
        return self.active_fingerprint.header_html if self.active_fingerprint else None

    def update_from_ocr_html(self, table_html: str, col_count: int, page_num: int):
        thead_match = re.search(r'<thead>.*?</thead>', table_html, re.DOTALL | re.I)
        if not thead_match: return
        fp = TableFingerprint(
            col_count=col_count, x_start=0.0, width=1000.0,
            header_html=thead_match.group(0)
        )
        self.update(fp, page_num)

    def get_header_text_plain(self) -> str:
        if not self.active_fingerprint: return ""
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', self.active_fingerprint.header_html, re.DOTALL | re.I)
        return ", ".join([re.sub(r'<[^>]*>', '', c).strip() for c in cells])

    def is_ocr_continuation(self, col_count: int, page_num: int) -> bool:
        if not self.active_fingerprint: return False
        if page_num - self.last_table_page > 2: return False
        return abs(self.active_fingerprint.col_count - col_count) <= 2 and col_count >= 2


# =============================================================
# Core Extractor Interfaces (From V4.5)
# =============================================================

