FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    libtesseract-dev \
    curl \
  && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN uv pip install --system \
    pdfplumber==0.11.7 \
    pytesseract==0.3.13 \
    pillow==11.3.0 \
    pypdfium2==4.30.0

RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

COPY --chown=appuser:appuser . .

CMD ["python", "main.py", "data/dod-mandatory-CUI-traning-cert.pdf"]

