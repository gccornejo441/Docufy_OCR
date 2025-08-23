import json
from pathlib import Path
import fitz
from .utils import render_page_to_pil, auto_rotate_osd, ocr_with_variants, pack_words

class DocuOCR:
    def __init__(self, dpi: int = 360, lang: str = "eng"):
        self.dpi = dpi
        self.lang = lang

    def process_page(self, page: fitz.Page) -> tuple[str, list[dict]]:
        pil = render_page_to_pil(page, dpi=self.dpi)
        pil = auto_rotate_osd(pil)
        result = ocr_with_variants(pil)
        words = pack_words(result["data"]) if result["data"] else []
        return result["text"], words

    def process_pdf(self, pdf_path: Path) -> tuple[str, list[dict]]:
        """OCR entire PDF, return raw text + words per page."""
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
                text, words = self.process_page(page)
                raw_pages.append(f"\n=== Page {i} ===\n{text}")
                words_per_page.append({"page": i, "words": words})
        return "".join(raw_pages), words_per_page

    def save_results(self, pdf_path: Path, outdir: Path):
        """Run OCR and save results to TXT + JSON files."""
        raw_text, words = self.process_pdf(pdf_path)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{pdf_path.stem}_ocr.txt").write_text(raw_text, encoding="utf-8")
        (outdir / f"{pdf_path.stem}_ocr_words.json").write_text(
            json.dumps(words, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
