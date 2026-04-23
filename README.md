# Local Document Intelligence Agent

## Project Overview
This project is an advanced, privacy-first Document Intelligence pipeline that runs completely locally. It parses various document types (PDFs, images, Word documents), extracts layout-preserved structures (like tables, multi-column text, and lists), and applies an intelligence layer to structure unstructured visual data into machine-readable semantic HTML and plain text. 

Unlike traditional OCR systems that destroy spatial relationships or cloud-based LLMs that compromise data privacy, this agent processes everything on your local machine using state-of-the-art open-weights Vision-Language Models (VLMs) and intelligent structural heuristics.

---

## Problem Statement
1. **Layout Destruction:** Traditional OCR often reads top-to-bottom, destroying the visual layout of documents, particularly tables, multi-column layouts, and nested hierarchies.
2. **Table Fragmentation:** When a table spans multiple pages, most OCR engines treat them as separate, disconnected grids, losing headers and context.
3. **Privacy & Cost:** Cloud-based OCR services (like AWS Textract or OpenAI Vision) require sending highly sensitive documents over the internet and incur recurring API costs. 

**The Solution:** This agent provides deterministic, layout-preserving extraction locally. It enforces a strict architecture where the LLM handles visual blueprinting and text extraction, while deterministic Python logic handles HTML reconstruction, grid mapping, and cross-page table fusion.

---

## Key Features
- **100% Local Processing:** Powered by local models via Ollama (optimized for `glm-ocr` and `gemma3:4b`). No data leaves your machine.
- **Multi-Format Extraction:** Seamlessly extracts and unifies content from PDFs, PNGs, JPGs, and DOCX files.
- **Cross-Page Table Fusion (Memory Module):** Intelligently detects when a table continues across page boundaries. It fingerprints the table structure and merges it back into a single continuous HTML table.
- **Hierarchical Layout Preservation:** Preserves reading order, text-around-table context, and multi-row headers.
- **Deterministic HTML Normalization:** Hardened table normalization pipeline prevents the incorrect dissolution of structured forms (e.g., GST forms, Bank Guarantees) into plain text.
- **Dynamic Routing:** Intelligently routes document pages between single-pass OCR and regional OCR based on layout complexity.
- **Dual Outputs:** Outputs both raw `.txt` for simple indexing and normalized semantic `.html` for high-fidelity downstream processing (like LLM chunking/RAG pipelines).

---

## Architecture & Core Modules

The repository is modularly designed into specific functional components:

### 1. `main.py` (Orchestration Layer)
The primary entry point. It initializes the `OllamaOCREngine`, registers the `DocumentRegistry`, and spins up the `DocumentIntelligenceEngine`. It handles the batch processing of all files placed in the input directory.

### 2. `extractors.py` (File Parsing Layer)
Contains format-specific file extractors:
- **PDFExtractor:** Uses `PyMuPDF` (`fitz`) to intelligently chunk and rasterize PDF pages.
- **ImageExtractor:** Uses `Pillow` to handle standard PNG, JPEG, and TIFF files.
- **DOCXExtractor:** Parses native Microsoft Word documents utilizing `python-docx`.

### 3. `ocr_engine.py` (Vision processing)
Handles interactions with the local Ollama model (`glm-ocr:q8_0`). Features built-in resilience, automatically adjusting resolution and retrying when the vision model encounters assertion errors or complex unreadable layouts.

### 4. `memory.py` (State & Fusion Layer)
Manages the `DocumentTableMemory` and `TableFingerprint`. It tracks the structural fingerprints of tables (e.g., column count, header patterns) as pages are processed. If an incomplete table is detected at the end of a page, it holds it in memory and logically merges it with the table on the subsequent page.

### 5. `intelligence.py` (Cognitive Layer)
Acts as the cognitive orchestrator. It uses heuristics to classify pages, enforces strict left-to-right extraction rules, and ensures that wide-format financial/engineering tables remain coherent. It applies base64 image encoding for Ollama stability and limits context windows dynamically.

### 6. `utils.py` (Normalization & Utility)
Contains the `HTMLCleaner` and structural logic to sanitize LLM-generated HTML, fixing incomplete tags, merging adjacent rows, detecting orphan rows, and ensuring standard semantic structures.

---

## Tech Stack
- **Python 3.9+**
- **Ollama** (Local LLM Execution environment)
- **PyMuPDF / fitz** (High-speed PDF Rendering)
- **python-docx** (Word Document Parsing)
- **Pillow** (Image Processing and base64 encoding)
- **BeautifulSoup4** (HTML Structuring & Sanitization)
- **Markdown** (Text formatting utilities)

---

## Installation Guide

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd final_agent
```

### 2. Set up a Python Virtual Environment
It is highly recommended to isolate the dependencies.
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and Setup Ollama
1. Download and install [Ollama](https://ollama.com/) for your operating system (Windows/macOS/Linux).
2. Start the Ollama server in your background.
3. Pull the required models used by the agent:
```bash
ollama pull glm-ocr:q8_0
ollama pull gemma3:4b  # Secondary/fallback validation model
```

---

## How to Run the Pipeline

1. **Add Documents:** Place the documents you want to process inside the `input_files/` directory. Supported formats include `.pdf`, `.png`, `.jpg`, `.jpeg`, and `.docx`.
2. **Execute:** Run the main pipeline script from the root of the repository:
   ```bash
   python main.py
   ```
3. **Monitor:** The script will output logging information to the console, detailing the extraction strategy per page, memory fusion events, and OCR progression.

---

## Output Structure

Once the processing is complete, the engine generates two distinct formats for every input document:

1. **`output_html/`**: Contains `<filename>.html`. This is a rich, structural representation of your document. All tables, list items, headers, and semantic reading blocks are preserved. This format is ideal for injecting into complex RAG (Retrieval-Augmented Generation) pipelines where table structure matters.
2. **`output_text/`**: Contains `<filename>.txt`. This is a linear, flattened plain text version of the document, optimized for fast full-text search (Elasticsearch) or simple tokenization.

---

## Future Roadmap
- [ ] **OpenCV Integration:** Implement explicit table-grid normalization using OpenCV (Hough Line Transform) to assist the VLM on highly degraded scanned tables.
- [ ] **Batch Parallelization:** Support parallel asynchronous processing (e.g., `asyncio` / ThreadPoolExecutor) for faster multi-document throughput.
- [ ] **JSON Schema Export:** Add extraction endpoints to export fully typed `pydantic` JSON schemas alongside HTML.
- [ ] **Docker Orchestration:** A planned `Dockerfile` / `docker-compose.yml` to package the Python environment, GPU drivers, and Ollama into a seamless, reproducible cloud-native container.

---

*Note: For issues regarding Ollama timeouts or memory limits, ensure your hardware has adequate VRAM, or configure the memory parameters inside `intelligence.py`.*
