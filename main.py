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

from .utils import HTMLCleaner
from .ocr_engine import OllamaOCREngine
from .extractors import DocumentRegistry
from .intelligence import DocumentIntelligenceEngine

class V7Agent:
    """
    Orchestrates the V7 Architecture:
    1. V4.5 OCR Extraction -> HTML
    2. HTML -> Text Cleanup -> TXT (BeautifulSoup)
    3. Model Swap
    4. V5 Intelligence -> JSON (from TXT)
    """
    def __init__(self, input_dir: str, html_out_dir: str, txt_out_dir: str):
        self.input_path = Path(input_dir)
        self.html_out_path = Path(html_out_dir)
        self.txt_out_path = Path(txt_out_dir)
        self.ocr_engine = OllamaOCREngine(model="glm-ocr:q8_0")
        self.registry = DocumentRegistry(self.ocr_engine)
        self.intelligence_engine = DocumentIntelligenceEngine(model="gemma3:4b")
        self.master_summary: List[Dict[str, Any]] = []

    def process_all_files(self):
        self.html_out_path.mkdir(parents=True, exist_ok=True)
        self.txt_out_path.mkdir(parents=True, exist_ok=True)
        if not self.input_path.exists(): 
            logger.warning(f"Input directory did not exist. Created it at: {self.input_path}")
            self.input_path.mkdir(parents=True, exist_ok=True)
            logger.info("Please place your files in the input directory and run the script again.")
            return
            
        files = [f for f in self.input_path.iterdir() if f.is_file()]
        
        # Free memory before starting heavy OCR
        logger.info("=== INITIAL CLEANUP: Stopping any loaded models ===")
        try:
            ollama.generate(model="gemma3:4b", keep_alive=0)
            ollama.generate(model="glm-ocr:q8_0", keep_alive=0)
        except Exception as e:
            logger.debug(f"Cleanup soft-fail: {e}")
        
        # PHASE 1: OCR Extraction (V4.5 Logic)
        logger.info(f"=== PHASE 1: Starting OCR Extraction for {len(files)} files ===")
        generated_html_files = []
        for file_path in files:
            extractor = self.registry.get_extractor(file_path.suffix)
            if not extractor: continue
            try:
                logger.info(f"Extracting {file_path.name}...")
                html_content = extractor.extract(file_path)
                html_content = self.ocr_engine._cross_table_merge(html_content)
                output_file = self.html_out_path / f"{file_path.stem}.html"
                output_file.write_text(html_content, encoding='utf-8')
                logger.info(f"SUCCESS (OCR): Saved {output_file.name}")
                generated_html_files.append((file_path.name, output_file))
            except Exception as e:
                logger.error(f"OCR Error on {file_path.name}: {e}", exc_info=True)

        # PHASE 2: HTML -> Clean Text
        logger.info("=== PHASE 2: Text Cleaning & Normalization ===")
        generated_txt_files = []
        for original_name, html_path in generated_html_files:
            try:
                html_content = html_path.read_text(encoding='utf-8')
                clean_text = HTMLCleaner.clean_to_text(html_content)
                txt_output_file = self.txt_out_path / f"{html_path.stem}.txt"
                txt_output_file.write_text(clean_text, encoding='utf-8')
                logger.info(f"SUCCESS (Clean): Saved {txt_output_file.name}")
                generated_txt_files.append((original_name, txt_output_file))
            except Exception as e:
                logger.error(f"Text Cleaning Error on {original_name}: {e}", exc_info=True)

        # PHASE 3: Stop OCR Model
        logger.info("=== PHASE 3: Stopping OCR model to free RAM ===")
        self.intelligence_engine.stop_ocr_model()
        
        # PHASE 4: Intelligence (V5 Logic)
        logger.info("=== PHASE 4: Running Document Intelligence ===")
        for original_name, txt_path in generated_txt_files:
            try:
                plain_text = txt_path.read_text(encoding='utf-8')
                logger.info(f"Analyzing {original_name} from {txt_path.name}...")
                summary = self.intelligence_engine.analyze(original_name, plain_text)
                self.master_summary.append(summary.to_dict())
                logger.info(f"SUCCESS (Intelligence): Analyzed {original_name}")
            except Exception as e:
                logger.error(f"Intelligence Error on {original_name}: {e}", exc_info=True)
                
        # PHASE 5: Save Summary
        logger.info("=== PHASE 5: Saving master summary ===")
        summary_path = self.input_path.parent / "document_summary.json"
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(self.master_summary, f, indent=2)
            logger.info(f"Done! Master summary saved to {summary_path.absolute()}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")


if __name__ == "__main__":
    # Ensure paths are always relative to the final_agent directory
    AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(AGENT_DIR, "input_files")
    HTML_OUT_DIR = os.path.join(AGENT_DIR, "output_html")
    TXT_OUT_DIR = os.path.join(AGENT_DIR, "output_text")
    
    agent = V7Agent(input_dir=INPUT_DIR, html_out_dir=HTML_OUT_DIR, txt_out_dir=TXT_OUT_DIR)
    agent.process_all_files()
