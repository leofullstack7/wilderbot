from typing import List, Dict, Any, Optional
from services.clients import pine_index, openai_client, EMBEDDING_MODEL

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
    # Nota: sin namespace => default; así /responder actual los encontrará
    if namespace:
        pine_index.upsert(vectors=vectors, namespace=namespace)
    else:
        pine_index.upsert(vectors=vectors)
    return {"upserted": len(vectors)}


def delete_by_doc_id(doc_id: str, namespace: Optional[str] = None) -> Dict[str, Any]:
    """
    Elimina TODOS los vectores cuyo metadata.doc_id == doc_id.
    """
    if not doc_id:
        return {"deleted": 0, "error": "doc_id vacío"}

    flt = {"doc_id": {"$eq": doc_id}}
    if namespace:
        pine_index.delete(filter=flt, namespace=namespace)
    else:
        pine_index.delete(filter=flt)
    # Pinecone no devuelve conteo exacto; devolvemos ack.
    return {"ok": True, "doc_id": doc_id}


def delete_by_ids(ids: List[str], namespace: Optional[str] = None) -> Dict[str, Any]:
    """
    Borra por IDs exactos de vectores (útil si los tienes guardados).
    """
    ids = [i for i in (ids or []) if i]
    if not ids:
        return {"deleted": 0, "error": "ids vacío"}
    if namespace:
        pine_index.delete(ids=ids, namespace=namespace)
    else:
        pine_index.delete(ids=ids)
    return {"ok": True, "count": len(ids)}
