from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# === ENV ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # puedes poner "gpt-3.5-turbo" si prefieres
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "wilder-frases")

# === Clientes ===
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# === Esquemas (compatibles con tu n8n actual) ===
class Entrada(BaseModel):
    mensaje: str
    usuario: str | None = None
    # Campos futuros (no obligatorios todavía)
    chat_id: str | None = None
    canal: str | None = None
    faq_origen: str | None = None
    nombre: str | None = None
    celular: str | None = None

@app.get("/health")
async def health():
    return {"status": "ok"}

def rag_search(query: str, top_k: int = 5):
    """
    Busca en Pinecone usando embeddings de OpenAI.
    Devuelve lista de dicts: [{id, texto, score}]
    """
    emb = client.embeddings.create(model=EMBEDDING_MODEL, input=query).data[0].embedding
    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    hits = []
    for m in res.matches:
        hits.append({
            "id": m.id,
            "texto": (m.metadata or {}).get("texto", ""),
            "score": float(m.score)
        })
    return hits

def build_messages_wilder(user_text: str, rag_snippets: list[str]):
    contexto = "\n".join([f"- {s}" for s in rag_snippets if s.strip()])
    prompt_sistema = (
        "Actúa como Wilder Escobar, Representante a la Cámara en Colombia. "
        "Tono: cercano, empático, concreto y propositivo; reconoce la iniciativa ciudadana y sugiere un siguiente paso realista. "
        "Evita prometer lo imposible y mantén enfoque público/ciudadano. "
        "Usa el contexto recuperado solo para enriquecer el estilo y coherencia."
    )
    contexto_msg = (
        "Contexto recuperado (frases reales de Wilder):\n"
        f"{contexto if contexto else '(sin coincidencias relevantes)'}"
    )
    return [
        {"role": "system", "content": prompt_sistema},
        {"role": "user", "content": f"{contexto_msg}\n\nMensaje del ciudadano:\n{user_text}"}
    ]

@app.post("/responder")
async def responder(data: Entrada):
    try:
        # 1) Buscar contexto en Pinecone
        hits = rag_search(data.mensaje, top_k=5)
        snippets = [h["texto"] for h in hits]

        # 2) Armar mensajes y llamar a OpenAI
        messages = build_messages_wilder(data.mensaje, snippets)
        chat_completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=500
        )

        texto = chat_completion.choices[0].message.content.strip()
        return {
            "respuesta": texto,
            "fuentes": hits  # útil si luego quieres mostrarlas o auditarlas
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
