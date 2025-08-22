# src/main.py
from __future__ import annotations

import io
import os
import re
import json
import shutil
import statistics
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from pytesseract import Output

try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


DEFAULT_PDF = Path("./data/dod-mandatory-CUI-traning-cert.pdf")
OUT_TXT     = Path("./data/ocr_raw.txt")
OUT_WORDS   = Path("./data/ocr_words.json")

TESSERACT_EXE = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
else:
    raise FileNotFoundError(
        "Tesseract not found. Install it (e.g., winget install -e --id UB-Mannheim.TesseractOCR) "
        "or adjust TESSERACT_EXE in this file."
    )


def render_page_to_pil(page: fitz.Page, dpi: int = 360) -> Image.Image:
    """Render a PDF page to a PIL image at the requested DPI."""
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def auto_rotate_osd(pil_img: Image.Image) -> Image.Image:
    """Use Tesseract OSD to detect orientation and rotate accordingly."""
    try:
        osd = pytesseract.image_to_osd(pil_img)
        m = re.search(r"Rotate:\s+(\d+)", osd)
        if m:
            angle = int(m.group(1)) % 360
            if angle:
                pil_img = pil_img.rotate(360 - angle, expand=True)
    except Exception:
        pass
    return pil_img


def deskew_adaptive(pil_img: Image.Image) -> Image.Image:
    """
    Deskew and adaptively binarize using OpenCV when available.
    Falls back to the original image if OpenCV is not installed.
    """
    if not HAVE_CV2:
        return pil_img

    img = np.array(pil_img)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thr == 0))
    if coords.size < 10:
        rotated = gray
    else:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    adap = cv2.adaptiveThreshold(
        rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return Image.fromarray(adap)


def ocr_with_variants(pil_img: Image.Image) -> dict:
    """
    Try several (OEM, PSM) combos, return the best by mean word confidence.
    Returns: {"mean_conf": float, "text": str, "data": pytesseract DICT}
    """
    candidates = [
        ("3", "4"), 
        ("3", "6"), 
        ("1", "4"), 
        ("1", "6"), 
    ]
    best = {"mean_conf": -1.0, "text": "", "data": None}

    for oem, psm in candidates:
        cfg = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1 -c user_defined_dpi=300"
        text = pytesseract.image_to_string(pil_img, lang="eng", config=cfg)
        data = pytesseract.image_to_data(pil_img, lang="eng", config=cfg, output_type=Output.DICT)

        confs = []
        for c in data.get("conf", []):
            try:
                fc = float(c)
                if fc >= 0:
                    confs.append(fc)
            except Exception:
                pass

        mean_conf = statistics.mean(confs) if confs else -1.0
        if mean_conf > best["mean_conf"]:
            best = {"mean_conf": mean_conf, "text": text, "data": data}

    return best


def pack_words(data: dict) -> list[dict]:
    """Convert pytesseract image_to_data DICT to a compact word list with boxes/conf."""
    words = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        conf = data["conf"][i]
        try:
            conf_val = float(conf)
        except Exception:
            continue
        if conf_val < 0:
            continue
        words.append({
            "text": txt,
            "conf": conf_val,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "block": int(data["block_num"][i]),
            "para": int(data["par_num"][i]),
            "line": int(data["line_num"][i]),
            "word": int(data["word_num"][i]),
        })
    return words


def ocr_pdf_generic(pdf_path: Path, dpi: int = 360) -> tuple[str, list[dict]]:
    """
    Generic OCR:
      1) If PDF has a text layer, use it (no OCR).
      2) Otherwise render full pages -> rotate -> (deskew+adaptive) -> OCR with variants.
    Returns:
      raw_text: concatenated page text
      words_per_page: [{"page": i, "words": [ ... word dicts ... ]}, ...]
    """
    with fitz.open(pdf_path) as doc:
        has_text = any(page.get_text("text").strip() for page in doc)
        if has_text:
            raw_pages = []
            words_stub = []
            for i, page in enumerate(doc, 1):
                raw_pages.append(f"\n=== Page {i} ===\n{page.get_text('text')}")
                words_stub.append({"page": i, "words": []})
            return "".join(raw_pages), words_stub

    raw_pages = []
    words_per_page = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, 1):
            pil = render_page_to_pil(page, dpi=dpi)

            pil = auto_rotate_osd(pil)
            pil = deskew_adaptive(pil)

            result = ocr_with_variants(pil)

            raw_pages.append(f"\n=== Page {i} (conf≈{result['mean_conf']:.1f}) ===\n{result['text']}")
            words_per_page.append({"page": i, "words": pack_words(result["data"]) if result["data"] else []})

    return "".join(raw_pages), words_per_page


def main() -> None:
    
    import sys

    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_WORDS.parent.mkdir(parents=True, exist_ok=True)

    raw_text, words = ocr_pdf_generic(pdf_path, dpi=360)

    OUT_TXT.write_text(raw_text, encoding="utf-8")
    OUT_WORDS.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OCR complete.")
    print(f"  Raw text  -> {OUT_TXT}")
    print(f"  Word boxes-> {OUT_WORDS}")


if __name__ == "__main__":
    main()
