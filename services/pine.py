# services/pine.py

from typing import List, Dict, Any, Optional
from services.clients import openai_client, EMBEDDING_MODEL
from pinecone import Pinecone
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "wilder-frases")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)  # exportado

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def upsert_chunks(items: List[Dict[str, Any]], namespace: Optional[str] = None):
    """
    items: [{id, text, metadata}]
    """
    if not items:
        return {"upserted": 0}
    vectors = []
    texts = [it["text"] for it in items]
    embs = embed_texts(texts)
    for it, v in zip(items, embs):
        vectors.append({
            "id": it["id"],
            "values": v,
            "metadata": {**(it.get("metadata") or {}), "texto": it["text"]}
        })
    if namespace:
        index.upsert(vectors=vectors, namespace=namespace)
    else:
        index.upsert(vectors=vectors)
    return {"upserted": len(vectors)}

def delete_by_doc_id(doc_id: str, namespace: Optional[str] = None) -> Dict[str, Any]:
    """
    Elimina TODOS los vectores cuyo metadata.doc_id == doc_id.
    """
    if not doc_id:
        return {"deleted": 0, "error": "doc_id vacío"}

    flt = {"doc_id": doc_id}  # igualdad simple es suficiente
    if namespace:
        index.delete(filter=flt, namespace=namespace)
    else:
        index.delete(filter=flt)
    return {"ok": True, "doc_id": doc_id}

def delete_by_ids(ids: List[str], namespace: Optional[str] = None) -> Dict[str, Any]:
    ids = [i for i in (ids or []) if i]
    if not ids:
        return {"deleted": 0, "error": "ids vacío"}
    if namespace:
        index.delete(ids=ids, namespace=namespace)
    else:
        index.delete(ids=ids)
    return {"ok": True, "count": len(ids)}
