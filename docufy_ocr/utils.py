import io
import re
import statistics
import fitz
from PIL import Image
import pytesseract
from pytesseract import Output

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

def ocr_with_variants(pil_img: Image.Image) -> dict:
    """
    Try several (OEM, PSM) combos, return the best by mean word confidence.
    Returns: {"mean_conf": float, "text": str, "data": pytesseract DICT}
    """
    candidates = [("3", "4"), ("3", "6"), ("1", "4"), ("1", "6")]
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
