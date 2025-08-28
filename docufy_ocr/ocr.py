import json
from pathlib import Path
import fitz
from PIL import Image
from typing import Tuple
from .utils import render_page_to_pil, auto_rotate_osd, ocr_with_variants, pack_words


class DocuOCR:
    def __init__(self, dpi: int = 360, lang: str = "eng"):
        self.dpi = dpi
        self.lang = lang

    def process_page(self, page: fitz.Page) -> tuple[str, list[dict]]:
        pil = render_page_to_pil(page, dpi=self.dpi)
        pil = auto_rotate_osd(pil)
        result = ocr_with_variants(pil, lang=self.lang, dpi=self.dpi)
        words = pack_words(result["data"]) if result["data"] else []
        return result["text"], words

    def process_pdf(self, pdf_path: Path) -> tuple[str, list[dict]]:
        with fitz.open(pdf_path) as doc:
            has_text = any(page.get_text("text").strip() for page in doc)
            if has_text:
                raw_pages = []
                words_stub = []
                for i, page in enumerate(doc, 1):
                    raw_pages.append(
                        f"\n=== Page {i} ===\n{page.get_text('text')}")
                    words_stub.append({"page": i, "words": []})
                return "".join(raw_pages), words_stub

        raw_pages = []
        words_per_page = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, 1):
                text, words = self.process_page(page)
                raw_pages.append(f"\n=== Page {i} ===\n{text}")
                words_per_page.append({"page": i, "words": words})
        return "".join(raw_pages), words_per_page

    def save_results(self, pdf_path: Path, outdir: Path):
        raw_text, words = self.process_pdf(pdf_path)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{pdf_path.stem}_ocr.txt").write_text(raw_text,
                                                         encoding="utf-8")
        (outdir / f"{pdf_path.stem}_ocr_words.json").write_text(
            json.dumps(words, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    @staticmethod
    def _clamp01(v: float) -> float:
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    @staticmethod
    def _rect_unrotated_from_view_rotated(
        x: float, y: float, w: float, h: float, rotation: int
    ) -> Tuple[float, float, float, float]:
        rotation = rotation % 360
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

        def inv_map(px: float, py: float) -> Tuple[float, float]:
            if rotation == 0:
                return px, py
            if rotation == 90:
                return py, 1.0 - px
            if rotation == 180:
                return 1.0 - px, 1.0 - py
            if rotation == 270:
                return 1.0 - py, px
            return px, py

        mapped = [inv_map(px, py) for (px, py) in corners]
        xs = [DocuOCR._clamp01(mx) for mx, _ in mapped]
        ys = [DocuOCR._clamp01(my) for _, my in mapped]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)

    def _render_region_to_pil(
        self, page: fitz.Page, rect_norm_unrotated: Tuple[float, float, float, float]
    ) -> Image.Image:
        x, y, w, h = rect_norm_unrotated
        if w <= 0 or h <= 0:
            raise ValueError("Selection has zero area.")
        pr = page.rect
        clip = fitz.Rect(
            pr.x0 + x * pr.width,
            pr.y0 + y * pr.height,
            pr.x0 + (x + w) * pr.width,
            pr.y0 + (y + h) * pr.height,
        )
        mat = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if mode == "RGBA":
            img = img.convert("RGB")
        return img

    def process_page_region(
        self,
        page: fitz.Page,
        rect_norm_view: Tuple[float, float, float, float],
        rotation_view: int = 0,
    ) -> tuple[str, list[dict]]:
        rotation_view = rotation_view % 360
        x, y, w, h = rect_norm_view
        x = self._clamp01(x)
        y = self._clamp01(y)
        w = self._clamp01(w)
        h = self._clamp01(h)
        xr, yr, wr, hr = self._rect_unrotated_from_view_rotated(
            x, y, w, h, rotation_view)
        if wr == 0 or hr == 0:
            return "", []
        pil = self._render_region_to_pil(page, (xr, yr, wr, hr))
        if rotation_view:
            pil = pil.rotate(-rotation_view, expand=True)
        result = ocr_with_variants(pil, lang=self.lang, dpi=self.dpi)
        words = pack_words(result["data"]) if result["data"] else []
        return result["text"], words

    def process_pdf_region(
        self,
        pdf_path: Path,
        page_number: int,
        rect_norm_view: Tuple[float, float, float, float],
        rotation_view: int = 0,
    ) -> tuple[str, list[dict]]:
        rotation_view = rotation_view % 360
        if page_number < 1:
            raise ValueError("page_number must be 1-based.")
        with fitz.open(pdf_path) as doc:
            idx = page_number - 1
            if idx < 0 or idx >= len(doc):
                raise ValueError("Page index out of range.")
            page = doc[idx]
            return self.process_page_region(page, rect_norm_view, rotation_view)
