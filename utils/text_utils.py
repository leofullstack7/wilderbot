import re
import hashlib
from typing import List

def normalize_text(t: str) -> str:
    t = t or ""
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    return t.strip()

def make_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

def split_into_chunks(text: str, max_chars: int = 3000, overlap: int = 400) -> List[str]:
    """
    Divide un texto largo en bloques ~max_chars con un pequeño solape.
    Tamaños pensados para ~800–1200 tokens aprox.
    """
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == n:
            break
        start = end - overlap  # solape
        if start < 0:
            start = 0
    return [c for c in chunks if c]
