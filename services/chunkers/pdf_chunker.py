from typing import List, Dict, Any
import pdfplumber
from io import BytesIO
from utils.text_utils import normalize_text, split_into_chunks

def parse_pdf(content_bytes: bytes, doc_title: str) -> List[Dict[str, Any]]:
    """
    Extrae texto por páginas y luego lo parte en chunks.
    (No hace OCR; si el PDF es escaneado, esto no leerá texto.)
    """
    items: List[Dict[str, Any]] = []
    with pdfplumber.open(BytesIO(content_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            t = page.extract_text() or ""
            t = normalize_text(t)
            if not t:
                continue
            for chunk in split_into_chunks(t, max_chars=2800, overlap=350):
                items.append({
                    "text": chunk,
                    "metadata": {"doc_title": doc_title, "source_type": "pdf", "page": i+1}
                })
    return items
