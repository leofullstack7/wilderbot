from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv

# Firestore
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()
app = FastAPI()

# === ENV ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "wilder-frases")
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "/etc/secrets/firebase.json"

# === Clients ===
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# === Firestore init ===
try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate(GOOGLE_CREDS)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ===== Schemas =====
class Entrada(BaseModel):
    mensaje: str
    usuario: Optional[str] = None
    chat_id: Optional[str] = None     # ej: tg_12345
    canal: Optional[str] = None       # telegram|whatsapp|web
    faq_origen: Optional[str] = None
    nombre: Optional[str] = None
    celular: Optional[str] = None     # "telefono" en tu BD

@app.get("/health")
async def health():
    return {"status": "ok"}

# ===== Helpers de BD (ajustados al esquema) =====
def upsert_usuario_o_anon(chat_id: str, nombre: Optional[str], telefono: Optional[str], canal: Optional[str]) -> str:
    """Devuelve usuario_id (usamos el mismo chat_id como identificador estable)."""
    usuario_id = chat_id  # simplificación útil multi-canal
    if telefono:  # va a 'usuarios'
        ref = db.collection("usuarios").document(usuario_id)
        doc = ref.get()
        if not doc.exists:
            ref.set({
                "nombre": nombre or "",
                "telefono": telefono,
                "barrio": None,
                "fecha_registro": firestore.SERVER_TIMESTAMP,
                "chats": [chat_id]
            })
        else:
            ref.update({
                "nombre": nombre or doc.to_dict().get("nombre", ""),
                "telefono": telefono,
                "chats": firestore.ArrayUnion([chat_id])
            })
    else:  # va a 'anonimos'
        ref = db.collection("anonimos").document(usuario_id)
        doc = ref.get()
        if not doc.exists:
            ref.set({
                "nombre": nombre or None,
                "fecha_registro": firestore.SERVER_TIMESTAMP,
                "chats": [chat_id]
            })
        else:
            ref.update({
                "nombre": nombre or doc.to_dict().get("nombre", None),
                "chats": firestore.ArrayUnion([chat_id])
            })
    return usuario_id

def ensure_conversacion(chat_id: str, usuario_id: str, faq_origen: Optional[str]):
    """Crea doc en conversaciones si no existe, con campos del esquema."""
    conv_ref = db.collection("conversaciones").document(chat_id)
    if not conv_ref.get().exists:
        conv_ref.set({
            "usuario_id": usuario_id,
            "faq_origen": faq_origen or None,
            "categoria_general": None,
            "titulo_propuesta": None,
            "mensajes": [],  # arreglo en el documento (según tu modelo)
            "fecha_inicio": firestore.SERVER_TIMESTAMP,
            "ultima_fecha": firestore.SERVER_TIMESTAMP,
            "tono_detectado": None
        })
    else:
        conv_ref.update({"ultima_fecha": firestore.SERVER_TIMESTAMP})
    return conv_ref

def append_mensajes(conv_ref, nuevos: List[Dict[str, Any]]):
    """Añade mensajes (role/content) al arreglo 'mensajes' del doc."""
    # Evitamos ArrayUnion porque no preserva duplicados/orden; leemos y reescribimos.
    doc = conv_ref.get()
    data = doc.to_dict() or {}
    arr = data.get("mensajes", [])
    arr.extend(nuevos)
    conv_ref.update({"mensajes": arr, "ultima_fecha": firestore.SERVER_TIMESTAMP})

# ===== RAG =====
def rag_search(query: str, top_k: int = 5):
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

def build_messages(user_text: str, rag_snippets: List[str], historial: List[Dict[str, str]]):
    contexto = "\n".join([f"- {s}" for s in rag_snippets if s.strip()])
    system_msg = (
        "Actúa como Wilder Escobar, Representante a la Cámara en Colombia.\n"
        "Tono: cercano, claro y humano. Responde en **máximo 4 frases** (sin párrafos largos).\n"
        "Si pides datos, haz **1 pregunta puntual**. Si el usuario cambia de tema, respóndelo sin perder cortesía.\n"
        "Usa el contexto recuperado para estilo y coherencia; no inventes hechos."
    )
    contexto_msg = "Contexto recuperado (frases reales de Wilder):\n" + (contexto if contexto else "(sin coincidencias relevantes)")

    msgs = [{"role": "system", "content": system_msg}]
    # historial previo ya está en formato [{'role': 'user'|'assistant', 'content': '...'}]
    if historial:
        msgs.extend(historial[-8:])  # usa últimos 8
    msgs.append({"role": "user", "content": f"{contexto_msg}\n\nMensaje del ciudadano:\n{user_text}"})
    return msgs

def load_historial_para_prompt(conv_ref) -> List[Dict[str, str]]:
    """Lee el arreglo 'mensajes' y lo devuelve tal cual para el prompt."""
    doc = conv_ref.get()
    if doc.exists:
        data = doc.to_dict() or {}
        msgs = data.get("mensajes", [])
        # Sanitizado mínimo
        out = []
        for m in msgs[-8:]:
            role = m.get("role")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": content})
        return out
    return []

# ===== Endpoint principal =====
@app.post("/responder")
async def responder(data: Entrada):
    try:
        chat_id = data.chat_id or f"web_{os.urandom(4).hex()}"
        # 1) Registrar usuario vs anónimo
        usuario_id = upsert_usuario_o_anon(chat_id, data.nombre or data.usuario, data.celular, data.canal)

        # 2) Asegurar conversacion
        conv_ref = ensure_conversacion(chat_id, usuario_id, data.faq_origen)

        # 3) RAG + historial
        hits = rag_search(data.mensaje, top_k=5)
        historial = load_historial_para_prompt(conv_ref)
        messages = build_messages(data.mensaje, [h["texto"] for h in hits], historial)

        # 4) LLM
        completion = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, temperature=0.5, max_tokens=350
        )
        texto = completion.choices[0].message.content.strip()

        # 5) Guardar turnos en arreglo 'mensajes'
        append_mensajes(conv_ref, [
            {"role": "user", "content": data.mensaje},
            {"role": "assistant", "content": texto}
        ])

        return {"respuesta": texto, "fuentes": hits, "chat_id": chat_id}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
