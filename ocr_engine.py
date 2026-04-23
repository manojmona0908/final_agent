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

from .utils import _is_orphan_row, _last_cell_text, _extract_serial, _first_cell_text, _is_incomplete_end, _apply_merge, _cells_of, _cell_text
from .memory import TableFingerprint, DocumentTableMemory

class OllamaOCREngine:
    """Handles interactions with the Ollama local model (glm-ocr:q8_0)."""

    def __init__(self, model: str = "glm-ocr:q8_0"):
        self.model = model

    def perform_ocr(self, image_data: bytes, retry_count: int = 0) -> str:
        if not ollama:
            return "<p>Error: 'ollama' library is not installed.</p>"

        ladder = [1120, 896, 672]
        target_dim = ladder[retry_count] if retry_count < len(ladder) else 448
        proc_data = self._preprocess_image(image_data, max_dim=target_dim)

        try:
            prompt = (
                "MANDATORY: Return clean HTML body only. No boilerplate.\n"
                "MANDATORY: Extract tables in their FULL WIDTH from the left edge (S.No/Item) to the right edge. Never omit leading columns.\n"
                "RULE: You MUST include every column (e.g. S.No, Type, Part No, Description, etc). No horizontal truncation permitted."
            )

            img_b64 = base64.b64encode(proc_data).decode('utf-8')

            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                images=[img_b64],
                stream=False,
                options={
                    "num_ctx": 3072,
                    "temperature": 0.0,
                    "top_p": 0.1
                }
            )

            raw_content = response.get('response', 'No content extracted.')
            normalized = self._normalize_table_html(raw_content)
            return self._clean_llm_output(normalized)

        except Exception as e:
            if retry_count < 2 and ("500" in str(e) or "assertion" in str(e).lower() or "ggml" in str(e).lower()):
                logger.warning(f"Ollama FAILURE. Retrying at lower resolution...")
                return self.perform_ocr(image_data, retry_count + 1)
            logger.error(f"Ollama failure: {e}", exc_info=True)
            return f"<p>OCR processing error: {e}</p>"

    def perform_ocr_with_prompt(self, image_data: bytes, custom_prompt: str, retry_count: int = 0) -> str:
        proc_data = self._preprocess_image(image_data, max_dim=1120)
        try:
            img_b64 = base64.b64encode(proc_data).decode('utf-8')
            response = ollama.generate(
                model=self.model,
                prompt=custom_prompt,
                images=[img_b64],
                stream=False,
                options={"num_ctx": 3072}
            )
            raw = response.get('response', '')
            normalized = self._normalize_table_html(raw)
            return self._clean_llm_output(normalized)
        except:
            return self.perform_ocr(image_data)

    def _preprocess_image(self, image_data: bytes, max_dim: int = 1120) -> bytes:
        if not Image: return image_data
        try:
            img = Image.open(io.BytesIO(image_data))
            if img.mode != 'RGB': img = img.convert('RGB')
            w, h = img.size
            scale = max_dim / max(w, h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            square_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            square_img.paste(img_resized, ((max_dim - new_w) // 2, (max_dim - new_h) // 2))

            with io.BytesIO() as output:
                square_img.save(output, format="JPEG", quality=90)
                return output.getvalue()
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image_data

    def _is_full_page_table(self, img: 'Image.Image') -> bool:
        if not Image: return False
        try:
            THUMB_W, THUMB_H = 256, 320
            grey = img.convert("L").resize((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)
            pixels = grey.load()

            col_darkness = []
            for x in range(THUMB_W):
                col_darkness.append(sum(1 for y in range(THUMB_H) if pixels[x, y] < 200))
                
            peaks = 0
            for x in range(1, THUMB_W - 1):
                if col_darkness[x] > THUMB_H * 0.3:
                    if col_darkness[x] > col_darkness[x-1] and col_darkness[x] >= col_darkness[x+1]:
                        peaks += 1
            if peaks >= 4:
                return True
            return False
        except:
            return False

    def perform_regional_ocr(
        self, img: 'Image.Image', page_num: int = 0, memory: Optional['DocumentTableMemory'] = None
    ) -> str:
        try:
            with io.BytesIO() as buf:
                img.save(buf, format="PNG")
                raw_data = buf.getvalue()
            full_html = self.perform_ocr(raw_data)
            if '</table>' in full_html.lower() and len(full_html) > 250:
                return full_html
        except Exception as e:
            logger.warning(f"    Single-pass fallback: {e}")

        width, height = img.size
        chosen_splits: List[int] = []

        SPEC_TABLE_CUES = [
            "general specification", "data acquisition", "input impedance",
            "technical specification", "electrical specification",
            "performance specification", "measurement range",
        ]
        NOTICE_CUES = [
            "scope of work", "annexure", "enclosure", "schedule",
            "terms and conditions", "list of documents", "notice inviting tender",
            "empanelment of", "university of", "no.", "date:", "letterhead",
            "logo", "institution", "department", "office of",
        ]
        LIST_MARKER_RE = re.compile(r'(?:^|\n)\s*(?:\d+[\.]\s+|[a-z][\.]\s+)', re.I)

        if self._is_full_page_table(img):
            grey = img.convert("L")
            px = grey.load()
            TARGET_HEIGHT = 1100
            num_bands = max(1, height // TARGET_HEIGHT)
            ideal_targets = [int((i + 1) * height / (num_bands + 1)) for i in range(num_bands)]
            
            band_splits: List[int] = []
            row_density: List[int] = []
            in_gap, gap_start = False, 0
            for y in range(height):
                dark = sum(1 for x in range(width) if px[x, y] < 180)
                row_density.append(dark)
                if dark < width * 0.04:
                    if not in_gap: gap_start, in_gap = y, True
                else:
                    if in_gap:
                        if y - gap_start >= int(height * 0.012):
                            band_splits.append((gap_start + y) // 2)
                        in_gap = False
            
            for target in ideal_targets:
                match = None
                if band_splits:
                    match = min(band_splits, key=lambda y: abs(y - target))
                    if abs(match - target) > 350: match = None
                if not match:
                    search_y0 = max(0, target - 200)
                    search_y1 = min(height - 1, target + 200)
                    densities = row_density[search_y0:search_y1]
                    if densities:
                        min_val = min(densities)
                        rel_y = densities.index(min_val)
                        match = search_y0 + rel_y
                if match is not None:
                    chosen_splits.append(match)
                    band_splits = [b for b in band_splits if abs(b - match) > 150]
            chosen_splits = sorted(list(set(chosen_splits)))
        else:
            try:
                thumb = img.copy()
                scale = 512 / max(width, height)
                tw, th = max(1, int(width * scale)), max(1, int(height * scale))
                thumb = thumb.resize((tw, th), Image.Resampling.LANCZOS)
                with io.BytesIO() as buf:
                    thumb.save(buf, format="PNG")
                    thumb_bytes = buf.getvalue()

                resp = ollama.generate(
                    model=self.model,
                    prompt="Extract all visible text. Return plain text only.",
                    images=[base64.b64encode(thumb_bytes).decode('utf-8')],
                    stream=False,
                    options={"num_ctx": 1024},
                )
                raw_thumb = resp.get("response", "")

                if raw_thumb:
                    lines = raw_thumb.splitlines()
                    split_line_idx = None
                    split_kind = None

                    for i, ln in enumerate(lines):
                        low = ln.strip().lower()
                        if any(cue in low for cue in SPEC_TABLE_CUES):
                            split_line_idx, split_kind = i, "spec"
                            break
                        if any(cue in low for cue in NOTICE_CUES):
                            split_line_idx, split_kind = i, "notice"
                            break

                    if split_line_idx is None:
                        joined = "\n".join(lines)
                        m = LIST_MARKER_RE.search(joined)
                        if m:
                            char_pos = m.start()
                            acc = 0
                            for i, ln in enumerate(lines):
                                acc += len(ln) + 1
                                if acc >= char_pos:
                                    split_line_idx, split_kind = i, "list"
                                    break

                    if split_line_idx is not None and split_line_idx > 0:
                        frac = max(0.10, min(split_line_idx / max(len(lines), 1), 0.85))
                        split_y = int(frac * height)
                        if split_kind in ("notice", "list"):
                            split_y = max(int(height * 0.08), split_y - int(height * 0.15))
                        chosen_splits = [split_y]
            except Exception as e:
                pass

            if not chosen_splits:
                grey = img.convert("L")
                pixels = grey.load()
                row_density = []
                for y in range(height):
                    row_density.append(sum(1 for x in range(width) if pixels[x, y] < 200))

                band_splits: List[int] = []
                in_gap, gap_start = False, 0
                threshold, min_gap = width * 0.02, int(height * 0.04)
                for y, d in enumerate(row_density):
                    if d <= threshold:
                        if not in_gap: gap_start, in_gap = y, True
                    else:
                        if in_gap:
                            if y - gap_start >= min_gap:
                                band_splits.append((gap_start + y) // 2)
                            in_gap = False

                preferred = [height // 3, (2 * height) // 3]
                for pref in preferred:
                    if band_splits:
                        closest = min(band_splits, key=lambda y: abs(y - pref))
                        if abs(closest - pref) < height * 0.15:
                            chosen_splits.append(closest)
                            band_splits.remove(closest)
                chosen_splits = sorted(set(chosen_splits))

            if not chosen_splits and height > width * 1.2:
                chosen_splits = [int(height * 0.15)]

        cuts = [0] + chosen_splits + [height]
        html_parts = []
        for idx in range(len(cuts) - 1):
            y0, y1 = cuts[idx], cuts[idx+1]
            if y1 - y0 < int(height * 0.05): continue
            crop = img.crop((0, y0, width, y1))
            with io.BytesIO() as buf:
                crop.save(buf, format="PNG")
                r_bytes = buf.getvalue()
            
            reg_prompt = None
            if memory:
                h_text = memory.get_header_text_plain()
                if h_text:
                    reg_prompt = f"Extract table. EXPECTED COLUMNS: {h_text}. Ensure all {len(h_text.split(','))} columns are extracted even if some values are sparsely populated."

            region_html = self.perform_ocr(r_bytes) if not reg_prompt else self.perform_ocr_with_prompt(r_bytes, reg_prompt)

            if memory:
                tables = re.findall(r'<table[^>]*>.*?</table>', region_html, re.DOTALL | re.I)
                for tbl in tables:
                    m = re.search(r'<tr[^>]*>(.*?)</tr>', tbl, re.DOTALL | re.I)
                    if m:
                        cols = len(re.findall(r'</t[dh]>', m.group(1), re.I))
                        if cols >= 2:
                            if memory.is_ocr_continuation(cols, page_num):
                                head = memory.get_header_html()
                                if head:
                                    if '<thead>' in tbl.lower():
                                        patched = re.sub(r'<thead>.*?</thead>', head, tbl, count=1, flags=re.DOTALL | re.I)
                                    else:
                                        t_tag = re.search(r'<table[^>]*>', tbl, re.I).group(0)
                                        patched = tbl.replace(t_tag, t_tag + head, 1)
                                    region_html = region_html.replace(tbl, patched)
                            else:
                                memory.update_from_ocr_html(tbl, cols, page_num)
            html_parts.append(region_html)
        return "\n".join(html_parts)

    def _normalize_table_html(self, html: str) -> str:
        html = re.sub(r'<th[^>]*>ead>', '<thead>', html, flags=re.I)
        html = re.sub(r'<td[^>]*>ody>', '<tbody>', html, flags=re.I)
        html = re.sub(r'<tb[^>]*>ody>', '<tbody>', html, flags=re.I)
        html = re.sub(r'</tr>\s*</tr>', '</tr>', html)
        html = re.sub(r'</td>\s*</td>', '</td>', html)

        def table_fixer(match):
            table_raw = match.group(0)
            rows = re.findall(r'<tr[^>]*>.*?</tr>', table_raw, re.DOTALL | re.I)
            if not rows:
                rows = [('<tr' + r) for r in re.split(r'<tr[^>]*>', table_raw, flags=re.I)[1:]]

            repaired_rows = []
            max_cols = 0

            for row in rows:
                if len(re.findall(r'</t[dh]>', row, re.I)) >= 2:
                    cells = re.findall(r'<t[dh][^>]*>.*?</t[dh]>', row, re.DOTALL | re.I)
                    row_logical_cols = sum(int((re.search(r'colspan=[\'"]?(\d+)[\'"]?', c, re.I) or [None, 1])[1]) for c in cells)
                    max_cols = max(max_cols, row_logical_cols)
                    repaired_rows.append(row)
                    continue

                inner = re.sub(r'^<tr[^>]*>', '', row, flags=re.I)
                inner = re.sub(r'</tr>$', '', inner, flags=re.I).strip()
                if not inner: continue

                parts = re.split(r'(<t[dh][^>]*>)', inner, flags=re.I)
                repaired_cells = []
                current_tag = "<td>"
                row_logical_cols = 0

                for part in parts:
                    if not part: continue
                    if part.lower().startswith('<t'):
                        current_tag = part
                        m = re.search(r'colspan=[\'"]?(\d+)[\'"]?', current_tag, re.I)
                        row_logical_cols += int(m.group(1)) if m else 1
                    else:
                        txt = re.sub(r'</t[dh]>', '', part, flags=re.I).strip()
                        if txt:
                            closing = "</th>" if current_tag.lower().startswith("<th") else "</td>"
                            repaired_cells.append(f"{current_tag}{txt}{closing}")

                if repaired_cells:
                    max_cols = max(max_cols, row_logical_cols)
                    repaired_rows.append(f"<tr>{''.join(repaired_cells)}</tr>")

            if not repaired_rows: return table_raw

            balanced_rows = []
            for r in repaired_rows:
                cells_in_row = len(re.findall(r'</t[dh]>', r, re.I))
                if cells_in_row <= 2 and max_cols - cells_in_row >= 2:
                    r = re.sub(r'colspan=[\'"]?\d+[\'"]?', f'colspan="{max_cols}"', r, flags=re.I)
                    if 'colspan=' not in r.lower():
                        r = re.sub(r'<(t[dh])', rf'<\1 colspan="{max_cols}"', r, count=1, flags=re.I)
                balanced_rows.append(r)

            merged_rows: List[str] = []
            skip_next = False
            for idx, row_html in enumerate(balanced_rows):
                if skip_next:
                    skip_next = False
                    continue

                is_orphan, cont_text = _is_orphan_row(row_html)
                merged = False

                if is_orphan and merged_rows:
                    prev = merged_rows[-1]
                    prev_last = _last_cell_text(prev)

                    if not merged:
                        prev_serial = _extract_serial(_first_cell_text(prev))
                        if prev_serial is not None:
                            for future in balanced_rows[idx + 1:]:
                                f_orphan, _ = _is_orphan_row(future)
                                if not f_orphan:
                                    future_serial = _extract_serial(_first_cell_text(future))
                                    if future_serial == prev_serial + 1:
                                        merged_rows[-1] = _apply_merge(prev, cont_text)
                                        merged = True
                                    break

                    if not merged and _is_incomplete_end(prev_last):
                        merged_rows[-1] = _apply_merge(prev, cont_text)
                        merged = True

                if not merged:
                    merged_rows.append(row_html)

            balanced_rows = merged_rows

            headers = [r for r in balanced_rows if '<th' in r.lower()]
            bodies = [r for r in balanced_rows if '<th' not in r.lower()]

            table_attrs = 'border="1" style="border-collapse: collapse; width: 100%; margin: 20px 0;"'
            final_table = [f"<table {table_attrs}>"]
            if headers: final_table.append(f"<thead>{''.join(headers)}</thead>")
            if bodies: final_table.append(f"<tbody>{''.join(bodies)}</tbody>")
            final_table.append("</table>")
            return "".join(final_table)

        return re.sub(r'<table[^>]*>.*?</table>', table_fixer, html, flags=re.DOTALL | re.I)

    def _cross_table_merge(self, html_doc: str) -> str:
        def _last_row_of_table(tbl_html: str) -> str:
            rows = re.findall(r'<tr[^>]*>.*?</tr>', tbl_html, re.DOTALL | re.I)
            return rows[-1] if rows else ''
        def _first_row_of_table(tbl_html: str) -> str:
            rows = re.findall(r'<tr[^>]*>.*?</tr>', tbl_html, re.DOTALL | re.I)
            return rows[0] if rows else ''
        def _strip_first_row(tbl_html: str) -> str:
            return re.sub(r'<tr[^>]*>.*?</tr>', '', tbl_html, count=1, flags=re.DOTALL | re.I).strip()
        def _patch_last_row_last_cell(tbl_html: str, extra: str) -> str:
            rows = re.findall(r'<tr[^>]*>.*?</tr>', tbl_html, re.DOTALL | re.I)
            if not rows: return tbl_html
            last_row = rows[-1]
            patched_row = _apply_merge(last_row, extra)
            if patched_row == last_row:
                patched_row = re.sub(r'(</t[dh]>)(?!.*</t[dh]>)', lambda m: extra + m.group(1), last_row, count=1, flags=re.DOTALL | re.I)
            pos = tbl_html.rfind(last_row)
            if pos >= 0: return tbl_html[:pos] + patched_row + tbl_html[pos + len(last_row):]
            return tbl_html

        TABLE_RE = re.compile(r'(<table[^>]*>.*?</table>)', re.DOTALL | re.I)
        segments = TABLE_RE.split(html_doc)
        
        result_segs = list(segments)
        i = 1
        while i < len(result_segs) - 2:
            tbl_a = result_segs[i]
            gap   = result_segs[i + 1]
            tbl_b = result_segs[i + 2]
            
            if not tbl_a or not tbl_b:
                i += 2
                continue

            raw_gap_text = re.sub(r'<[^>]+>', ' ', gap).strip()
            if len(raw_gap_text) > 35 and 'page' not in raw_gap_text.lower():
                i += 2
                continue

            last_row_a  = _last_row_of_table(tbl_a)
            first_row_b = _first_row_of_table(tbl_b)
            if not last_row_a or not first_row_b:
                i += 2
                continue

            do_merge = False
            last_serial = _extract_serial(_first_cell_text(last_row_a))
            next_serial = _extract_serial(_first_cell_text(first_row_b))
            
            if last_serial is not None and next_serial is not None and next_serial == last_serial + 1:
                do_merge = True
            elif len(raw_gap_text) < 40:
                cols_a = len(_cells_of(last_row_a))
                cols_b = len(_cells_of(first_row_b))
                if cols_a > 1 and abs(cols_a - cols_b) <= 1:
                    do_merge = True

            orphan_extra = None
            if not do_merge:
                cells_b = _cells_of(first_row_b)
                if len(cells_b) >= 2:
                    meaningful_cells = [t for t in (_cell_text(c) for c in cells_b[:-1]) if len(t) > 3]
                    if len(meaningful_cells) <= 1 and _is_incomplete_end(_last_cell_text(last_row_a)):
                        do_merge = True
                        orphan_extra = _cell_text(cells_b[-1])

            if do_merge:
                if orphan_extra:
                    tbl_a = _patch_last_row_last_cell(tbl_a, orphan_extra)
                    tbl_b = _strip_first_row(tbl_b)
                
                b_rows = re.findall(r'<tr[^>]*>.*?</tr>', tbl_b, re.DOTALL | re.I)
                b_data = ''.join([r for r in b_rows if '<th' not in r.lower()])
                
                if b_data.strip():
                    ins_pos = tbl_a.lower().rfind('</tbody>')
                    if ins_pos == -1: ins_pos = tbl_a.lower().rfind('</table>')
                    if ins_pos >= 0:
                        tbl_a = tbl_a[:ins_pos] + b_data + tbl_a[ins_pos:]
                        result_segs[i] = tbl_a
                        result_segs[i+1] = ''
                        result_segs[i+2] = ''
                        result_segs.pop(i + 1)
                        result_segs.pop(i + 1)
                        continue
            i += 2

        merged_doc = ''.join(result_segs)

        def _balance_table(match):
            tbl = match.group(0)
            rows = re.findall(r'<tr[^>]*>.*?</tr>', tbl, re.DOTALL | re.I)
            if not rows: return tbl
            
            max_cols = 0
            row_cells = []
            for r in rows:
                cells = re.findall(r'<t[dh][^>]*>.*?</t[dh]>', r, re.DOTALL | re.I)
                row_cells.append((r, cells))
                if len(cells) > max_cols: max_cols = len(cells)
            
            patched_tbl = tbl
            for r, cells in row_cells:
                missing = max_cols - len(cells)
                if missing > 0:
                    padding = '<td style="padding: 8px;"></td>' * missing
                    patched_r = re.sub(r'(</tr>)(?!.*</tr>)', lambda m: padding + m.group(1), r, count=1, flags=re.DOTALL | re.I)
                    if patched_r != r:
                        patched_tbl = patched_tbl.replace(r, patched_r, 1)
            return patched_tbl

        return re.sub(r'<table[^>]*>.*?</table>', _balance_table, merged_doc, flags=re.DOTALL | re.I)

    def _clean_llm_output(self, raw_content: str) -> str:
        cleaned = raw_content.strip()
        cleaned = re.sub(r'^```html\s*', '', cleaned, flags=re.I)
        cleaned = re.sub(r'^```\s*', '', cleaned, flags=re.I | re.M)
        cleaned = re.sub(r'```$', '', cleaned)
        cleaned = cleaned.strip()

        cleaned = re.sub(r'<!DOCTYPE[^>]*>', '', cleaned, flags=re.I)
        cleaned = re.sub(r'<html[^>]*>', '', cleaned, flags=re.I)
        cleaned = re.sub(r'</html>', '', cleaned, flags=re.I)
        cleaned = re.sub(r'<head[^>]*>.*?</head>', '', cleaned, flags=re.DOTALL | re.I)
        cleaned = re.sub(r'<body[^>]*>', '', cleaned, flags=re.I)
        cleaned = re.sub(r'</body>', '', cleaned, flags=re.I)
        cleaned = cleaned.strip()

        if markdown and any(cleaned.startswith(c) for c in ['#', '*', '-', '1.']):
            cleaned = markdown.markdown(cleaned, extensions=['fenced_code'])

        lower_cleaned = cleaned.lower()
        has_structure = any(tag in lower_cleaned for tag in ["<p", "<table", "<ul", "<ol", "<h", "<div"])
        if not has_structure:
            lines = cleaned.split('\n')
            cleaned = "\n".join([f"<p>{l.strip()}</p>" for l in lines if l.strip()])

        cleaned = re.sub(
            r'<table(?![^>]*border=)',
            r'<table border="1" style="border-collapse: collapse; width: 100%; margin: 20px 0;"',
            cleaned
        )
        cleaned = re.sub(r'<th(?![^>]*style=)', r'<th style="background-color: #f2f2f2; padding: 8px;"', cleaned)
        cleaned = re.sub(r'<td(?![^>]*style=)', r'<td style="padding: 8px;"', cleaned)

        def table_reducer(match):
            table_html = match.group(0)
            rows_list = re.findall(r'<tr[^>]*>.*?</tr>', table_html, re.DOTALL | re.I)
            rows = len(rows_list)
            row_content = re.findall(r'<(td|th)[^>]*>(.*?)</\1>', table_html, re.DOTALL | re.I)
            cell_count = len(row_content)
            cols = cell_count / (rows or 1)

            all_colspans = re.findall(r'colspan=[\'"]?(\d+)[\'"]?', table_html, re.I)
            is_uniform_colspan = len(set(all_colspans)) == 1 and len(all_colspans) == rows

            generic_markers = [
                "particulars", "description", "value", "content", "field",
                "item", "section", "details", "remarks", "scope"
            ]
            first_row = re.search(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.I)
            is_generic = False
            if first_row:
                header_text = first_row.group(1).lower()
                if any(m in header_text for m in generic_markers):
                    is_generic = True

            if (rows < 3 and cols < 2.5) or (rows == 1) or is_uniform_colspan or (is_generic and cols <= 2):
                list_items = []
                has_long_paragraphs = False
                for row_html in rows_list:
                    cells = re.findall(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', row_html, re.DOTALL | re.I)
                    if len(cells) == 2:
                        val0 = re.sub(r'<[^>]+>', '', cells[0]).strip()
                        if re.match(r'^\s*(\d+|[a-z])[\.\)]\s*$', val0, re.I):
                            val1 = re.sub(r'<[^>]+>', '', cells[1]).strip()
                            if len(val1) > 150: has_long_paragraphs = True
                            if val1: list_items.append(val1)

                if has_long_paragraphs:
                    res = []
                    for _, content in row_content:
                        c = re.sub(r'<[^>]+>', ' ', content).strip()
                        if c: res.append(f"<p>{c}</p>")
                    return "".join(res)

                if len(list_items) >= 3:
                    li_html = "".join([f"<li>{x}</li>" for x in list_items])
                    return f"<ol>{li_html}</ol>"

                text_blocks = []
                for _, content in row_content:
                    c = re.sub(r'<[^>]+>', ' ', content).strip()
                    if c:
                        text_blocks.append(c)
                if len(text_blocks) > 4:
                    if (len(text_blocks) >= 2
                            and text_blocks[0].lower() in {"section", "item"}
                            and text_blocks[1].lower() in {"content", "description"}):
                        text_blocks = text_blocks[2:]

                    heading = ""
                    if text_blocks and (
                        "annexure" in text_blocks[0].lower()
                        or "scope of work" in text_blocks[0].lower()
                    ):
                        heading = f"<h3>{text_blocks[0]}</h3>"
                        text_blocks = text_blocks[1:]

                    li = "".join([f"<li>{t}</li>" for t in text_blocks])
                    return f"{heading}<ol>{li}</ol>"
                return "".join([f"<p>{t}</p>" for t in text_blocks])

            return table_html

        cleaned = re.sub(r'<table[^>]*>.*?</table>', table_reducer, cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned


