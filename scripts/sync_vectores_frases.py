# scripts/sync_vectores_frases.py
import os
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()  # útil en local (.env); en Render usas env vars y secret files

import firebase_admin
from firebase_admin import credentials, firestore

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ---------------- Config ----------------
OPENAI_MODEL_EMB = "text-embedding-3-small"   # 1536 dims, económico
EMBED_DIM = 1536
INDEX_NAME = os.getenv("PINECONE_INDEX", "wilder-frases")

# ---------------- Helpers ----------------
def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Falta la variable de entorno: {name}")
    return val

def chunk_text(text: str, max_tokens: int = 500) -> List[str]:
    """Chunk sencillo por longitud aprox. (~4 chars ≈ 1 token)."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        cut = text.rfind(".", start, end)
        if cut == -1 or cut - start < max_chars * 0.5:
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut
    return [c for c in chunks if c]

# ---------------- Init servicios ----------------
# OpenAI
OPENAI_API_KEY = require_env("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Firebase Admin (Render: /etc/secrets/firebase.json; local: credenciales/serviceAccountKey.json)
firebase_cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not firebase_cred_path:
    # Fallback local
    firebase_cred_path = "/etc/secrets/firebase.json"

cred = credentials.Certificate(firebase_cred_path)
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Pinecone
PINECONE_API_KEY = require_env("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Crear índice si no existe (SDK nuevo)
# Nota: en el SDK nuevo, list_indexes() devuelve un objeto con .indexes
existing = [idx.name for idx in pc.list_indexes().indexes]
if INDEX_NAME not in existing:
    print(f"Creando índice '{INDEX_NAME}' en Pinecone...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=OPENAI_MODEL_EMB, input=texts)
    return [d.embedding for d in resp.data]

def sync_frases_wilder():
    print("Descargando frases desde Firestore...")
    docs = db.collection("frases_wilder").stream()

    upserts = []
    count = 0

    for doc in docs:
        data: Dict = doc.to_dict() or {}
        frase_id = doc.id                # ya usas IDs numéricos
        texto = (data.get("texto") or "").strip()
        if not texto:
            continue

        chunks = chunk_text(texto, max_tokens=400)
        vectors = embed_texts(chunks)

        for i, (chunk, emb) in enumerate(zip(chunks, vectors)):
            vector_id = f"{frase_id}-{i}"
            metadata = {
                "frase_id": frase_id,
                "texto": chunk,
                "tono": data.get("tono", ""),
                "contexto": data.get("contexto", ""),
                "palabras_clave": data.get("palabras_clave", []),
                "tipo": "frase_wilder",
            }
            upserts.append({"id": vector_id, "values": emb, "metadata": metadata})
            count += 1

        # sube en lotes
        if len(upserts) >= 100:
            print(f"Upserting {len(upserts)} vectores...")
            index.upsert(vectors=upserts)
            upserts = []

    if upserts:
        print(f"Upserting {len(upserts)} vectores (final)...")
        index.upsert(vectors=upserts)

    print(f"✅ Sincronización completa. {count} vectores indexados/actualizados en '{INDEX_NAME}'.")

if __name__ == "__main__":
    sync_frases_wilder()
