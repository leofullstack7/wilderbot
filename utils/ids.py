import uuid

def new_doc_id(prefix="doc") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def chunk_id(doc_id: str, i: int) -> str:
    return f"{doc_id}-c{str(i+1).zfill(3)}"
