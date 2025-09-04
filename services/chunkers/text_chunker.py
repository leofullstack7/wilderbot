from typing import List, Dict, Any
from utils.text_utils import normalize_text, split_into_chunks

def parse_text_note(title: str, content: str) -> List[Dict[str, Any]]:
    content = normalize_text(content)
    items: List[Dict[str, Any]] = []
    for chunk in split_into_chunks(content, max_chars=3000, overlap=400):
        items.append({
            "text": chunk,
            "metadata": {"doc_title": title or "Nota", "source_type": "note"}
        })
    return items
