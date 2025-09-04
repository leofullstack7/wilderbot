from typing import List, Dict, Any, Optional
from services.clients import pine_index, openai_client, EMBEDDING_MODEL

def rag_search(query: str, top_k: int = 5, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
    emb = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=query).data[0].embedding
    if namespace:
        res = pine_index.query(vector=emb, top_k=top_k, include_metadata=True, namespace=namespace)
    else:
        res = pine_index.query(vector=emb, top_k=top_k, include_metadata=True)
    hits = []
    for m in res.matches or []:
        hits.append({
            "id": m.id,
            "texto": (m.metadata or {}).get("texto", ""),
            "score": float(m.score)
        })
    return hits
