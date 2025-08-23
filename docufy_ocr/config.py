from pathlib import Path
import os
import shutil
import pytesseract

DEFAULT_PDF = Path("./data/dod-mandatory-CUI-traning-cert.pdf")
OUT_TXT     = Path("./data/ocr_raw.txt")
OUT_WORDS   = Path("./data/ocr_words.json")

# Locate Tesseract
TESSERACT_EXE = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
else:
    raise FileNotFoundError(
        "Tesseract not found. Install it (e.g., winget install -e --id UB-Mannheim.TesseractOCR) "
        "or adjust TESSERACT_EXE in config.py."
    )
