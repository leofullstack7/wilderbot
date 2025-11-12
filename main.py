# ============================
#  WilderBot API - VERSIÓN REFACTORIZADA PROFESIONAL
#  Mantiene 100% funcionalidad con arquitectura limpia
# ============================

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from typing import Optional, List, Dict, Any, Tuple
from google.cloud.firestore_v1 import Increment

import json
import os
from dotenv import load_dotenv
import re
import math
import time

from api.ingest import router as ingest_router
from fastapi.middleware.cors import CORSMiddleware

import firebase_admin
from firebase_admin import credentials, firestore

# =========================================================
#  Config
# =========================================================

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(ingest_router)

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "wilder-frases")
GOOGLE_CREDS     = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "/etc/secrets/firebase.json"
OPENAI_MODEL_SUMMARY = os.getenv("OPENAI_MODEL_SUMMARY", "gpt-4o-mini")

BOT_INTRO_TEXT = os.getenv(
    "BOT_INTRO_TEXT",
    "¡Hola! Soy la mano derecha de Wilder Escobar. Estoy aquí para escuchar y canalizar tus "
    "problemas, propuestas o reconocimientos. ¿Qué te gustaría contarme hoy?"
)

PRIVACY_REPLY = (
    "Entiendo perfectamente que quieras proteger tu información personal. "
    "Queremos que sepas que tus datos están completamente seguros con nosotros. "
    "Cumplimos con todas las normativas de protección de datos y solo usamos tu información "
    "para escalar tu propuesta y ayudar a que pueda hacerse realidad."
)

# Títulos permitidos para consultas
CONSULTA_TITULOS = [
    "Vida Personal Wilder", "General", "Leyes", "Movilidad",
    "Educación", "Salud", "Seguridad", "Vivienda", "Empleo"
]

# === Clientes ===
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# === Firestore Admin ===
try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate(GOOGLE_CREDS)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# =========================================================
#  Esquemas
# =========================================================

class Entrada(BaseModel):
    mensaje: str
    usuario: Optional[str] = None
    chat_id: Optional[str] = None
    canal: Optional[str] = None
    faq_origen: Optional[str] = None
    nombre: Optional[str] = None
    celular: Optional[str] = None

class ClasificarIn(BaseModel):
    chat_id: str
    contabilizar: Optional[bool] = None

@app.get("/health")
async def health():
    return {"status": "ok"}

# ═══════════════════════════════════════════════════════════════════════
# SECCIÓN 1: UTILIDADES DE TEXTO
# ═══════════════════════════════════════════════════════════════════════

def _normalize_text(t: str) -> str:
    """Normaliza texto para comparaciones."""
    t = t.lower()
    t = (t.replace("á","a").replace("é","e").replace("í","i")
           .replace("ó","o").replace("ú","u").replace("ü","u"))
    t = re.sub(r"[^a-zñ0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _titlecase(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).title()

def _clean_barrio_fragment(s: str) -> str:
    s = re.split(r"\s+(?:para|por|que|donde|con)\b|[,.;]|$", s, maxsplit=1, flags=re.IGNORECASE)[0]
    return _titlecase(s)

def limit_sentences(text: str, max_sentences: int = 3) -> str:
    """Limita respuesta a N oraciones."""
    parts = re.split(r'(?<=[\.\?!])\s+', text.strip())
    out = " ".join([p for p in parts if p][:max_sentences]).strip()
    return out or text

# Palabras de discurso que NO son nombres
DISCOURSE_START_WORDS = {
    _normalize_text(w) for w in [
        "hola", "holaa", "holaaa", "buenas", "buenos días", "saludos",
        "gracias", "ok", "okay", "vale", "listo", "perfecto", "claro", "sí", "si"
    ]
}

INTERROGATIVOS = [
    "que", "qué", "como", "cómo", "cuando", "cuándo", "donde", "dónde",
    "por que", "por qué", "cual", "cuál", "quien", "quién",
    "me gustaria saber", "quisiera saber", "podria decirme",
    "puedes explicarme", "informacion", "información"
]

# ═══════════════════════════════════════════════════════════════════════
# SECCIÓN 2: FUNCIONES DE BASE DE DATOS
# ═══════════════════════════════════════════════════════════════════════

def upsert_usuario_o_anon(
    chat_id: str,
    nombre: Optional[str],
    telefono: Optional[str],
    canal: Optional[str],
    barrio: Optional[str] = None
) -> str:
    """Crea o actualiza usuario/anónimo en BD."""
    usuario_id = chat_id

    if telefono:
        ref = db.collection("usuarios").document(usuario_id)
        doc = ref.get()
        if not doc.exists:
            ref.set({
                "nombre": nombre or "",
                "telefono": telefono,
                "barrio": barrio or None,
                "fecha_registro": firestore.SERVER_TIMESTAMP,
                "chats": [chat_id],
                "canal": canal or "web",
            })
        else:
            prev = doc.to_dict() or {}
            ref.update({
                "nombre": nombre or prev.get("nombre", ""),
                "telefono": telefono,
                "barrio": barrio or prev.get("barrio"),
                "chats": firestore.ArrayUnion([chat_id]),
                "canal": canal or prev.get("canal", "web"),
            })
    else:
        ref = db.collection("anonimos").document(usuario_id)
        doc = ref.get()
        if not doc.exists:
            ref.set({
                "nombre": nombre or None,
                "fecha_registro": firestore.SERVER_TIMESTAMP,
                "chats": [chat_id],
                "canal": canal or "web",
            })
        else:
            prev = doc.to_dict() or {}
            ref.update({
                "nombre": nombre or prev.get("nombre", None),
                "chats": firestore.ArrayUnion([chat_id]),
                "canal": canal or prev.get("canal", "web"),
            })
    return usuario_id

def ensure_conversacion(chat_id: str, usuario_id: str, faq_origen: Optional[str], canal: Optional[str]):
    """Crea conversación si no existe."""
    conv_ref = db.collection("conversaciones").document(chat_id)
    if not conv_ref.get().exists:
        conv_ref.set({
            "usuario_id": usuario_id,
            "faq_origen": faq_origen or None,
            "canal": canal or "web",
            "categoria_general": [],
            "titulo_propuesta": [],
            "mensajes": [],
            "fecha_inicio": firestore.SERVER_TIMESTAMP,
            "ultima_fecha": firestore.SERVER_TIMESTAMP,
            "tono_detectado": None,
            "last_topic_vec": None,
            "last_topic_summary": None,
            "awaiting_topic_confirm": False,
            "candidate_new_topic_summary": None,
            "candidate_new_topic_vec": None,
            "topics_history": [],
            "argument_requested": False,
            "argument_collected": False,
            "proposal_requested": False,
            "proposal_collected": False,
            "proposal_nudge_count": 0,
            "current_proposal": None,
            "contact_intent": None,
            "contact_requested": False,
            "contact_collected": False,
            "contact_refused": False,
            "contact_refusal_count": 0,
            "contact_info": {"nombre": None, "barrio": None, "telefono": None},
            "project_location": None,
            "location_note_sent": False,
        })
    else:
        if canal:
            conv_ref.update({"ultima_fecha": firestore.SERVER_TIMESTAMP, "canal": canal})
        else:
            conv_ref.update({"ultima_fecha": firestore.SERVER_TIMESTAMP})
    return conv_ref

def append_mensajes(conv_ref, nuevos: List[Dict[str, Any]]):
    """Guarda mensajes y actualiza resumen automáticamente."""
    snap = conv_ref.get()
    data = snap.to_dict() or {}
    arr = data.get("mensajes", [])
    arr.extend(nuevos)
    conv_ref.update({"mensajes": arr, "ultima_fecha": firestore.SERVER_TIMESTAMP})

    # Resumen corto
    try:
        resumen = summarize_conversation_brief(arr, max_chars=100)
        conv_ref.update({"resumen": resumen})
    except:
        pass
    
    # Resumen en caliente
    try:
        update_conversation_summary(conv_ref)
    except:
        pass

def load_historial_para_prompt(conv_ref) -> List[Dict[str, str]]:
    """Carga últimos 8 mensajes para contexto."""
    snap = conv_ref.get()
    if snap.exists:
        data = snap.to_dict() or {}
        msgs = data.get("mensajes", [])
        out = []
        for m in msgs[-8:]:
            role = m.get("role")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": content})
        return out
    return []

# ═══════════════════════════════════════════════════════════════════════
# SECCIÓN 3: MEMORIA CONVERSACIONAL
# ═══════════════════════════════════════════════════════════════════════

def update_conversation_summary(conv_ref, force: bool = False):
    """Mantiene resumen actualizado cada 4 mensajes."""
    snap = conv_ref.get()
    data = snap.to_dict() or {}
    
    msg_count = int(data.get("messages_since_summary", 0)) + 1
    
    if msg_count < 4 and not force:
        conv_ref.update({"messages_since_summary": msg_count})
        return
    
    all_messages = data.get("mensajes", [])
    
    if len(all_messages) < 4:
        return
    
    recent = all_messages[-12:]
    transcript = []
    for m in recent:
        role = "U" if m["role"] == "user" else "B"
        content = (m.get("content", "") or "")[:180]
        transcript.append(f"{role}: {content}")
    
    transcript_text = "\n".join(transcript)
    
    sys = (
        "Resume en 140 caracteres: tema, propuesta (si hay), fase.\n"
        "Sin nombres ni teléfonos."
    )
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_SUMMARY,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": transcript_text}
            ],
            temperature=0.2,
            max_tokens=50,
            timeout=3
        )
        
        summary = response.choices[0].message.content.strip()[:140]
        
        conv_ref.update({
            "conversacion_resumida": summary,
            "messages_since_summary": 0
        })
        
    except:
        pass

def summarize_conversation_brief(mensajes: List[Dict[str, str]], max_chars: int = 100) -> str:
    """Resume conversación en ≤100 caracteres."""
    parts = []
    for m in mensajes[-40:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        content = re.sub(r"\+?\d[\d\s\-]{6,}", "[número]", content)
        tag = "C" if role == "user" else "B"
        parts.append(f"{tag}: {content}")
    transcript = "\n".join(parts) if parts else ""

    try:
        out = client.chat.completions.create(
            model=OPENAI_MODEL_SUMMARY,
            messages=[
                {"role": "system", "content": "Resume en 100 chars."},
                {"role": "user", "content": transcript}
            ],
            temperature=0.2,
            max_tokens=60,
        ).choices[0].message.content
    except:
        out = (mensajes[0].get("content") if mensajes else "")[:max_chars]

    return out[:max_chars]

# ═══════════════════════════════════════════════════════════════════════
# SECCIÓN 4: DETECCIÓN DE INTENCIONES
# ═══════════════════════════════════════════════════════════════════════

def is_plain_greeting(text: str) -> bool:
    """Detecta saludos simples sin tema."""
    if not text:
        return False
    t = _normalize_text(text)
    kws = ("hola","holaa","buenas","buenos dias","como estas","que mas","saludos")
    short = len(t) <= 40
    has_kw = any(k in t for k in kws)
    topicish = any(w in t for w in (
        "arregl","propuesta","proponer","hueco","parque","colegio",
        "salud","seguridad","ayuda","necesito","quiero"
    ))
    return short and has_kw and not topicish

def is_proposal_intent_heuristic(text: str) -> bool:
    """Heurística rápida para detectar intención de propuesta."""
    t = _normalize_text(text)
    kw = ["propongo", "propuesta", "sugerencia", "mi idea", "quiero proponer"]
    return any(k in t for k in kw)

def is_question_like(text: str) -> bool:
    """Detecta si el mensaje es una pregunta (no una propuesta)."""
    t = _normalize_text(text)
    
    if "?" in text:
        return True
    
    interrogativos = [
        "que", "qué", "como", "cómo", "cuando", "cuándo",
        "donde", "dónde", "por que", "por qué", "cual", "cuál",
        "quien", "quién"
    ]
    
    tiene_interrogativo = any(kw in t for kw in interrogativos)
    
    if is_proposal_intent_heuristic(text):
        return False
    
    return tiene_interrogativo

def is_proposal_denial(text: str) -> bool:
    """Detecta si el usuario rechaza dar propuesta."""
    t = _normalize_text(text)
    pats = [
        r'\b(aun|aún|todavia)\s+no\b.*\b(propuest|idea)',
        r'\b(no\s+tengo|no\s+he\s+hecho)\b.*\b(propuest|idea)',
        r'\b(olvidalo|mejor\s+no|mas\s+tarde)\b'
    ]
    return any(re.search(p, t) for p in pats)

def looks_like_proposal_content(text: str) -> bool:
    """
    Detecta si el texto contiene una propuesta CON CONTENIDO REAL.
    """
    if is_proposal_denial(text):
        return False
    
    t = _normalize_text(text)
    
    # Rechazar intención pura sin contenido
    intencion_pura = re.match(
        r'^\s*(?:quiero|quisiera|me gustaria)\s+(?:proponer|hacer\s+una\s+propuesta)\s*$',
        t
    )
    if intencion_pura:
        return False
    
    # Debe tener VERBO DE ACCIÓN + OBJETO
    tiene_verbo = bool(re.search(
        r'\b(arregl|mejor|constru|instal|crear|paviment|ilumin|repar|pint|adecu|señaliz)\w*',
        t
    ))
    
    tiene_objeto = bool(re.search(
        r'\b(alumbrado|luminaria|parque|semaforo|cancha|via|calle|anden|acera|juegos|polideportivo|hospital|colegio|puesto|centro)\b',
        t
    ))
    
    tiene_ubicacion = bool(re.search(
        r'\b(barrio|sector|comuna|vereda|en\s+[A-Z])\b',
        text  # Original sin normalizar para detectar mayúsculas
    ))
    
    # Criterios de aceptación
    if tiene_verbo and tiene_objeto:
        return True
    
    if tiene_verbo and tiene_ubicacion and len(t) >= 15:
        return True
    
    if tiene_objeto and tiene_ubicacion and len(t) >= 15:
        return True
    
    if len(t) >= 40 and (tiene_verbo or tiene_objeto):
        return True
    
    return False

def has_argument_text(t: str) -> bool:
    """Detecta si el texto parece un argumento."""
    t = _normalize_text(t)
    if any(k in t for k in ["porque", "ya que", "debido", "es importante"]):
        return True
    if len(t) >= 30:
        return True
    return False

def detect_contact_refusal(text: str) -> bool:
    """Detecta si el usuario rechaza dar datos personales."""
    t = _normalize_text(text)
    return any(p in t for p in [
        "no me gusta dar mis datos",
        "no quiero compartir mis datos",
        "no doy mi celular"
    ])

# ═══════════════════════════════════════════════════════════════════════
# SECCIÓN 5: EXTRACCIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════════

def extract_user_name(text: str) -> Optional[str]:
    """Extrae nombre del usuario."""
    m = re.search(r'\b(?:soy|me llamo|mi nombre es)\s+([A-Za-záéíóúñ ]{2,40})', text, flags=re.IGNORECASE)
    if m:
        nombre = m.group(1).strip(" .,")
        return nombre if _normalize_text(nombre) not in DISCOURSE_START_WORDS else None
    return None

def extract_phone(text: str) -> Optional[str]:
    """Extrae teléfono."""
    m = re.search(r'(\+?\d[\d\s\-]{7,16}\d)', text)
    if not m:
        return None
    tel = re.sub(r'\D', '', m.group(1))
    tel = re.sub(r'^(?:00)?57', '', tel)
    return tel if 8 <= len(tel) <= 12 else None

def extract_user_barrio(text: str) -> Optional[str]:
    """
    Extrae barrio de residencia con detección FLEXIBLE.
    Acepta:
      - "vivo en Aranjuez"
      - "Barrio Milan" (directo)
      - "Milan" (si está después de pedirlo)
    """
    
    # Patrón 1: "vivo/resido en (el barrio)? X"
    m = re.search(
        r'\b(?:vivo|resido)\s+en\s+(?:el\s+)?(?:barrio\s+)?'
        r'([A-Záéíóúñ][A-Za-záéíóúñ0-9 \-]{1,49}?)'
        r'(?=(?:\s+(?:y|mi|número|teléfono|celular|desde|de|del|con|para|por)\b|[,.;]|$))',
        text, flags=re.IGNORECASE
    )
    if m:
        return _clean_barrio_fragment(m.group(1))
    
    # Patrón 2: "Barrio X" (explícito con la palabra "barrio")
    m = re.search(
        r'\bbarrio\s+([A-Záéíóúñ][A-Za-záéíóúñ0-9 \-]{2,30})',
        text, flags=re.IGNORECASE
    )
    if m:
        return _clean_barrio_fragment(m.group(1))
    
    # Patrón 3: "soy del barrio X"
    m = re.search(r'\bsoy\s+del\s+barrio\s+([A-Za-záéíóúñ0-9 \-]{2,50})', text, flags=re.IGNORECASE)
    if m:
        return _clean_barrio_fragment(m.group(1))
    
    # Patrón 4: "mi barrio es X" o "mi barrio X"
    m = re.search(r'\bmi\s+barrio\s+(?:es\s+)?([A-Za-záéíóúñ0-9 \-]{2,50})', text, flags=re.IGNORECASE)
    if m:
        return _clean_barrio_fragment(m.group(1))
    
    # ──────────────────────────────────────────────────────────
    # NUEVO: Patrón 5 - Nombre de barrio suelto con mayúscula inicial
    # (Solo si tiene entre 2-4 palabras y empieza con mayúscula)
    # ──────────────────────────────────────────────────────────
    
    # Buscar palabras que empiecen con mayúscula (posibles nombres de barrio)
    palabras_mayuscula = re.findall(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})\b', text)
    
    for posible_barrio in palabras_mayuscula:
        # Validar que NO sea una palabra común de discurso
        normalized = _normalize_text(posible_barrio)
        
        if normalized in DISCOURSE_START_WORDS:
            continue
        
        # Validar que tenga 2-4 palabras (barrios típicos)
        num_palabras = len(posible_barrio.split())
        if 1 <= num_palabras <= 4:
            return _titlecase(posible_barrio)
    
    return None

def extract_project_location(text: str) -> Optional[str]:
    """Extrae barrio del proyecto (no de residencia)."""
    m = re.search(r'\ben\s+el\s+barrio\s+([A-Za-záéíóúñ0-9 \-]{2,50})', text, flags=re.IGNORECASE)
    if m:
        return _clean_barrio_fragment(m.group(1))
    return None

def extract_proposal_text(text: str) -> str:
    """Extrae texto limpio de la propuesta."""
    t = text.strip()
    t = re.sub(r'^\s*(?:hola|buenas)[,!\s\-]*', '', t, flags=re.IGNORECASE)
    return limit_sentences(t, 2)

def llm_extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """Usa LLM para extraer contacto cuando formato no es claro."""
    sys = "Extrae: nombre, barrio, telefono. JSON."
    usr = f"Mensaje: {text}"
    
    try:
        out = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            temperature=0.0,
            max_tokens=150
        ).choices[0].message.content.strip()
        
        return json.loads(out)
    except:
        return {"nombre": None, "barrio": None, "telefono": None}

# ═══════════════════════════════════════════════════════════════════════
# SECCIÓN 6: CLASIFICACIÓN INTELIGENTE
# ═══════════════════════════════════════════════════════════════════════

def smart_classify_with_context(mensaje: str, resumen: str, conv_data: dict) -> Dict[str, Any]:
    """
    Clasificación híbrida: heurísticas + LLM con resumen.
    """
    t = _normalize_text(mensaje)
    
    # NIVEL 1: Heurísticas (gratis, instantáneo)
    propuesta_keywords = [
        "propongo", "mi propuesta es", "quisiera que construyan",
        "me gustaria que arreglen", "sugiero que", "deberían hacer"
    ]
    if any(kw in t for kw in propuesta_keywords):
        return {"tipo": "propuesta", "confianza": "alta", "metodo": "heuristica_propuesta"}
    
    # Consultas obvias
    if "?" in mensaje and not any(kw in t for kw in ["propongo", "sugiero", "me gustaria construir"]):
        if any(kw in t for kw in ["wilder", "ley", "proyecto de ley", "que propone", "que apoya"]):
            return {"tipo": "consulta", "confianza": "alta", "metodo": "heuristica_consulta"}
    
    # NIVEL 2: LLM con resumen (casos ambiguos)
    contexto = resumen if resumen else "Inicio de conversación (sin contexto previo)"
    
    sys = (
        f"Contexto: {contexto}\n"
        f"Nuevo mensaje: {mensaje}\n\n"
        "¿Es 'consulta' (pide información/explicación) o 'propuesta' (quiere construir/mejorar algo físico)?\n"
        "Responde UNA palabra: consulta|propuesta"
    )
    
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys}],
            temperature=0,
            max_tokens=5,
            timeout=2
        )
        elapsed = int((time.time() - start) * 1000)
        
        answer = response.choices[0].message.content.strip().lower()
        tipo = "propuesta" if "propuesta" in answer else "consulta"
        
        print(f"[PERF] LLM classify: {elapsed}ms")
        return {"tipo": tipo, "confianza": "media", "metodo": "llm_resumido"}
        
    except Exception as e:
        print(f"[WARN] LLM classify failed: {e}")
        return {"tipo": "consulta", "confianza": "baja", "metodo": "fallback"}

# ═══════════════════════════════════════════════════════════════════════
# SECCIÓN 7: RAG
# ═══════════════════════════════════════════════════════════════════════

def rag_search(query: str, top_k: int = 5):
    """Busca en Pinecone."""
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

def reformulate_query_with_context(
    current_query: str, 
    conversation_history: List[Dict[str, str]],
    max_history: int = 6
) -> str:
    """
    Reformula query con contexto COMPLETO de múltiples temas mencionados.
    """
    
    # Si la query ya es específica y larga, no reformular
    if len(current_query.split()) > 8:
        return current_query
    
    recent = conversation_history[-max_history:] if conversation_history else []
    
    if not recent:
        return current_query
    
    # ──────────────────────────────────────────────────────────
    # NUEVO: Extraer TODAS las leyes/temas mencionados (no solo el último)
    # ──────────────────────────────────────────────────────────
    temas_mencionados = []
    
    for msg in recent:
        if msg["role"] == "assistant":
            content = msg.get("content", "")
            
            # Extraer TODAS las leyes mencionadas
            leyes = re.findall(r'Ley\s+(\d+)\s+de\s+(\d+)', content, re.IGNORECASE)
            for num_ley, año in leyes:
                temas_mencionados.append(f"Ley {num_ley} de {año}")
            
            # Extraer temas generales (ej: "nivelación educativa", "cuidado ambiental")
            # Buscar frases clave después de "sobre", "para", "de"
            temas_generales = re.findall(
                r'(?:sobre|para|de)\s+([a-záéíóúñ\s]{10,50}?)(?:\.|,|$)',
                content.lower()
            )
            temas_mencionados.extend([t.strip() for t in temas_generales if len(t.strip()) > 10])
    
    # ──────────────────────────────────────────────────────────
    # Detectar referencias plurales ("las dos", "ambas", "esas")
    # ──────────────────────────────────────────────────────────
    referencias_plurales = ["las dos", "ambas", "esas", "esos", "las tres", "todos"]
    query_lower = current_query.lower()
    
    if any(ref in query_lower for ref in referencias_plurales):
        if len(temas_mencionados) >= 2:
            # Tomar las últimas 2-3 leyes mencionadas
            ultimos_temas = temas_mencionados[-3:] if len(temas_mencionados) >= 3 else temas_mencionados[-2:]
            reformulated = " y ".join(ultimos_temas) + " detalles"
            print(f"[QUERY] Reformulada (plural): '{reformulated}'")
            return reformulated
    
    # ──────────────────────────────────────────────────────────
    # Detectar referencias singulares ("eso", "esa", "la primera")
    # ──────────────────────────────────────────────────────────
    referencias_vagas = ["eso", "esa", "ese", "sobre eso", "de eso", "aquello"]
    
    if any(ref in query_lower for ref in referencias_vagas):
        if temas_mencionados:
            # Tomar el último tema mencionado
            ultimo_tema = temas_mencionados[-1]
            reformulated = f"{ultimo_tema} detalles"
            print(f"[QUERY] Reformulada (singular): '{reformulated}'")
            return reformulated
    
    return current_query


# ═══════════════════════════════════════════════════════════════════════
# SECCIÓN 8: HELPERS DE CONTACTO
# ═══════════════════════════════════════════════════════════════════════

def build_contact_request(missing: List[str]) -> str:
    """Genera mensaje pidiendo datos faltantes."""
    etiquetas = {"nombre": "tu nombre", "barrio": "tu barrio", "celular": "celular"}
    pedir = [etiquetas[m] for m in missing if m in etiquetas]
    frase = pedir[0] if len(pedir) == 1 else (", ".join(pedir[:-1]) + " y " + pedir[-1])
    return f"¿Me compartes {frase}?"

def build_project_location_request() -> str:
    """Pide ubicación del proyecto."""
    return "Para ubicar el caso en el mapa: ¿en qué barrio sería exactamente el proyecto?"

def craft_argument_question(name: Optional[str], project_location: Optional[str] = None) -> str:
    """Pregunta por argumentos."""
    saludo = f"{name}, " if name else ""
    return f"{saludo}¿Por qué es importante?"

def positive_ack_and_request_argument(name: Optional[str], project_location: Optional[str] = None) -> str:
    """Reconoce propuesta y pide argumento."""
    return "Excelente idea. ¿Por qué sería importante?"

def strip_contact_requests(texto: str) -> str:
    """Elimina pedidos de contacto del texto."""
    return texto

# ═══════════════════════════════════════════════════════════════════════
# SECCIÓN 9: GENERACIÓN DE RESPUESTAS
# ═══════════════════════════════════════════════════════════════════════

def build_messages(
    user_text: str,
    rag_snippets: List[str],
    historial: List[Dict[str, str]],
    emphasize_contact: bool = False,
    emphasize_argument: bool = False,
    intent: str = "otro",
):
    """Construye mensajes para el LLM."""
    contexto = "\n".join([f"- {s}" for s in rag_snippets if s.strip()])

    system_msg = (
        "Actúa como Wilder Escobar, Representante a la Cámara en Colombia.\n"
        "Tono cercano y claro. **Máximo 3–4 frases, sin párrafos largos.**\n"
        "No saludes de nuevo. No pidas contacto si no toca."
    )

    if emphasize_argument:
        system_msg += "\nPrioriza UNA pregunta breve para entender motivo/impacto."

    if emphasize_contact and intent in ("propuesta", "problema"):
        system_msg += "\nAhora pide contacto con suavidad (nombre, barrio y celular)."

    contexto_msg = "Contexto recuperado:\n" + (contexto if contexto else "(sin coincidencias)")

    msgs = [{"role": "system", "content": system_msg}]
    if historial:
        msgs.extend(historial[-8:])
    msgs.append({"role": "user", "content": f"{contexto_msg}\n\nMensaje:\n{user_text}"})
    return msgs

# ═══════════════════════════════════════════════════════════════════════
# SECCIÓN 10: ENDPOINT PRINCIPAL /responder
# ═══════════════════════════════════════════════════════════════════════

@app.post("/responder")
async def responder(data: Entrada):
    try:
        chat_id = data.chat_id or f"web_{os.urandom(4).hex()}"
        usuario_id = upsert_usuario_o_anon(chat_id, data.nombre or data.usuario, data.celular, data.canal)
        conv_ref = ensure_conversacion(chat_id, usuario_id, data.faq_origen, data.canal)
        conv_data = conv_ref.get().to_dict() or {}
        
        # ═══════════════════════════════════════════════════════════════
        # CAPA 0: Fast Path (respuestas instantáneas)
        # ═══════════════════════════════════════════════════════════════
        
        # Saludo inicial
        if not conv_data.get("mensajes") and is_plain_greeting(data.mensaje):
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": BOT_INTRO_TEXT}
            ])
            return {"respuesta": BOT_INTRO_TEXT, "fuentes": [], "chat_id": chat_id}
        
        # Rechazo de datos cuando ya se pidieron
        if conv_data.get("contact_requested") and not conv_data.get("contact_collected"):
            if detect_contact_refusal(data.mensaje):
                refusal_count = int(conv_data.get("contact_refusal_count", 0))
                if refusal_count == 0:
                    conv_ref.update({"contact_refusal_count": 1})
                    texto = PRIVACY_REPLY + " ¿Me compartes tus datos?"
                else:
                    conv_ref.update({"contact_refused": True})
                    texto = "Entiendo tu decisión. ¡Que tengas buen día!"
                
                append_mensajes(conv_ref, [
                    {"role": "user", "content": data.mensaje},
                    {"role": "assistant", "content": texto}
                ])
                return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
        
        # ═══════════════════════════════════════════════════════════════
        # CAPA 1: Extracción de Datos
        # ═══════════════════════════════════════════════════════════════
        
        name = extract_user_name(data.mensaje)
        phone = extract_phone(data.mensaje)
        user_barrio = extract_user_barrio(data.mensaje)
        proj_loc = extract_project_location(data.mensaje)
        
        # Si ya pedimos contacto y no hay formato claro, usar LLM
        if conv_data.get("contact_requested") and not (name or phone or user_barrio):
            llm_data = llm_extract_contact_info(data.mensaje)
            if llm_data.get("nombre"):
                name = llm_data["nombre"]
            if llm_data.get("telefono"):
                phone = llm_data["telefono"]
            if llm_data.get("barrio"):
                user_barrio = llm_data["barrio"]
        
        # Actualizar info de contacto
        if name or phone or user_barrio:
            current_info = conv_data.get("contact_info") or {}
            new_info = dict(current_info)
            
            if name and not current_info.get("nombre"):
                new_info["nombre"] = name
            if user_barrio:
                new_info["barrio"] = user_barrio
            if phone:
                new_info["telefono"] = phone
            
            conv_ref.update({"contact_info": new_info})
            if phone:
                conv_ref.update({"contact_collected": True})
                upsert_usuario_o_anon(
                    chat_id,
                    new_info.get("nombre") or data.nombre or data.usuario,
                    phone,
                    data.canal,
                    new_info.get("barrio") or user_barrio
                )
        
        if proj_loc:
            conv_ref.update({"project_location": proj_loc})
        
        # ═══════════════════════════════════════════════════════════════
        # CAPA 2: Clasificación Inteligente
        # ═══════════════════════════════════════════════════════════════
        
        resumen_contexto = conv_data.get("conversacion_resumida", "")
        clasificacion = smart_classify_with_context(data.mensaje, resumen_contexto, conv_data)
        
        # Si ya estamos en flujo de propuesta, mantenerlo
        already_in_proposal_flow = bool(
            conv_data.get("proposal_collected") or
            conv_data.get("proposal_requested") or
            conv_data.get("argument_requested") or
            conv_data.get("contact_intent") == "propuesta"
        )
        
        if already_in_proposal_flow:
            clasificacion["tipo"] = "propuesta"
            clasificacion["confianza"] = "alta"
        
        print(f"[CLASSIFY] {clasificacion}")
        
        # ═══════════════════════════════════════════════════════════════
        # CAPA 3: Flujo de Propuestas (Determinista)
        # ═══════════════════════════════════════════════════════════════
        
        # Negación de propuesta
        if is_proposal_denial(data.mensaje):
            conv_ref.update({
                "proposal_requested": False,
                "proposal_collected": False,
                "argument_requested": False,
                "argument_collected": False,
            })
            texto = "Perfecto. Cuando la tengas, cuéntamela."
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto}
            ])
            return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
        
        is_proposal_flow = already_in_proposal_flow or clasificacion["tipo"] == "propuesta"
        
        if is_proposal_flow:
            # FASE 1: Recopilar propuesta
            if not conv_data.get("proposal_collected"):
                # Caso 1A: Solo intención
                if is_proposal_intent_heuristic(data.mensaje) and not looks_like_proposal_content(data.mensaje):
                    conv_ref.update({
                        "proposal_requested": True,
                        "proposal_nudge_count": 0
                    })
                    texto = "¡Perfecto! ¿Cuál es tu propuesta? Cuéntamela en 1-2 frases."
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                # Caso 1B: Contenido completo
                if looks_like_proposal_content(data.mensaje):
                    propuesta_extraida = extract_proposal_text(data.mensaje)
                    
                    if len(_normalize_text(propuesta_extraida)) < 10:
                        conv_ref.update({"proposal_requested": True, "proposal_nudge_count": 1})
                        texto = "Cuéntame más: ¿qué te gustaría que se hiciera y en qué barrio?"
                        append_mensajes(conv_ref, [
                            {"role": "user", "content": data.mensaje},
                            {"role": "assistant", "content": texto}
                        ])
                        return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                    
                    conv_ref.update({
                        "current_proposal": propuesta_extraida,
                        "proposal_collected": True,
                        "argument_requested": True,
                        "proposal_nudge_count": 0
                    })
                    texto = positive_ack_and_request_argument(name, proj_loc)
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                # Caso 1C: Ya pedimos pero no llegó (NUDGES)
                if conv_data.get("proposal_requested"):
                    nudges = int(conv_data.get("proposal_nudge_count", 0))
                    
                    if is_proposal_denial(data.mensaje):
                        conv_ref.update({
                            "proposal_requested": False,
                            "proposal_nudge_count": 0,
                            "contact_intent": None
                        })
                        texto = "Perfecto. Cuando la tengas, cuéntamela."
                        append_mensajes(conv_ref, [
                            {"role": "user", "content": data.mensaje},
                            {"role": "assistant", "content": texto}
                        ])
                        return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                    
                    if is_question_like(data.mensaje):
                        conv_ref.update({
                            "proposal_requested": False,
                            "proposal_nudge_count": 0,
                            "contact_intent": None
                        })
                        texto = "¿Prefieres que responda tu pregunta o seguimos con tu propuesta?"
                        append_mensajes(conv_ref, [
                            {"role": "user", "content": data.mensaje},
                            {"role": "assistant", "content": texto}
                        ])
                        return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                    
                    nudges += 1
                    conv_ref.update({"proposal_nudge_count": nudges})
                    
                    if nudges == 1:
                        texto = "¿Cuál es tu propuesta? Dímela en 1-2 frases."
                    elif nudges == 2:
                        texto = "Escribe la propuesta en 1-2 frases (ej: 'Arreglar luminarias del parque')."
                    else:
                        conv_ref.update({
                            "proposal_requested": False,
                            "proposal_nudge_count": 0,
                            "contact_intent": None
                        })
                        texto = "Si prefieres, dime tu pregunta y te ayudo."
                    
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
            
            # FASE 2: Recopilar argumento
            if conv_data.get("proposal_collected") and not conv_data.get("argument_collected"):
                es_argumento = has_argument_text(data.mensaje) or len(_normalize_text(data.mensaje)) >= 20
                
                if es_argumento:
                    conv_ref.update({
                        "argument_collected": True,
                        "contact_requested": True,
                        "contact_intent": "propuesta"
                    })
                    
                    current_info = conv_data.get("contact_info") or {}
                    missing = []
                    if not current_info.get("nombre"): missing.append("nombre")
                    if not current_info.get("barrio"): missing.append("barrio")
                    if not current_info.get("telefono"): missing.append("celular")
                    
                    if missing:
                        texto = build_contact_request(missing)
                    else:
                        texto = "Perfecto, con estos datos escalamos tu propuesta."
                    
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                else:
                    texto = craft_argument_question(name, proj_loc)
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
            
            # FASE 3: Recopilar contacto
            if conv_data.get("argument_collected") and conv_data.get("contact_requested"):
                current_info = conv_data.get("contact_info") or {}
                missing = []
                if not (current_info.get("nombre") or name): missing.append("nombre")
                if not (current_info.get("barrio") or user_barrio): missing.append("barrio")
                if not (current_info.get("telefono") or phone): missing.append("celular")
                
                if missing:
                    texto = build_contact_request(missing)
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                else:
                    nombre_txt = current_info.get("nombre", "")
                    texto = (f"Gracias, {nombre_txt}. " if nombre_txt else "Gracias. ")
                    texto += "Con estos datos escalamos el caso."
                    
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
        
        # ═══════════════════════════════════════════════════════════════
        # CAPA 4: RAG + Generación
        # ═══════════════════════════════════════════════════════════════
        
        historial = load_historial_para_prompt(conv_ref)
        
        # Reformular query si hace referencia a contexto previo
        query_for_search = data.mensaje
        referencias = ["eso", "esa", "ese", "lo", "la", "el"]
        if any(ref in data.mensaje.lower() for ref in referencias):
            query_for_search = reformulate_query_with_context(data.mensaje, historial)
        
        hits = rag_search(query_for_search, top_k=5)
        
        messages = build_messages(
            data.mensaje,
            [h["texto"] for h in hits],
            historial,
            emphasize_contact=False,
            emphasize_argument=False,
            intent=clasificacion["tipo"],
        )
        
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=280
        )
        texto = completion.choices[0].message.content.strip()
        texto = limit_sentences(texto, 3)
        
        append_mensajes(conv_ref, [
            {"role": "user", "content": data.mensaje},
            {"role": "assistant", "content": texto}
        ])
        
        return {"respuesta": texto, "fuentes": hits, "chat_id": chat_id}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT /clasificar
# ═══════════════════════════════════════════════════════════════════════

@app.post("/clasificar")
async def clasificar(body: ClasificarIn):
    """Clasifica conversación y actualiza panel."""
    try:
        chat_id = body.chat_id
        conv_ref = db.collection("conversaciones").document(chat_id)
        snap = conv_ref.get()
        
        if not snap.exists:
            return {"ok": False, "mensaje": "Conversación no encontrada"}
        
        conv_data = snap.to_dict() or {}
        mensajes = conv_data.get("mensajes", [])
        
        ultimo_usuario = ""
        for m in reversed(mensajes):
            if m.get("role") == "user":
                ultimo_usuario = m.get("content", "")
                break
        
        if not ultimo_usuario:
            return {"ok": False, "mensaje": "No hay mensajes"}
        
        # Detectar tipo
        propuesta = conv_data.get("current_proposal") or ""
        
        if propuesta:
            tipo = "propuesta"
            texto_a_clasificar = propuesta
        else:
            es_consulta = (
                ("?" in ultimo_usuario) or
                any(kw in _normalize_text(ultimo_usuario) 
                    for kw in ["que", "qué", "como", "cómo", "wilder", "ley"])
            )
            
            if es_consulta:
                tipo = "consulta"
                texto_a_clasificar = ultimo_usuario
            else:
                return {"ok": True, "skipped": True, "reason": "ni_consulta_ni_propuesta"}
        
        # Clasificar con LLM
        if tipo == "consulta":
            sys = (
                "Clasifica esta CONSULTA.\n"
                "Devuelve JSON: {\"categoria_general\": \"Consulta\", \"titulo_propuesta\": \"[5-8 palabras]\"}"
            )
            usr = f"Consulta: {texto_a_clasificar}"
        else:
            ubicacion = conv_data.get("project_location") or ""
            sys = (
                "Clasifica esta PROPUESTA.\n"
                "Devuelve JSON: {\"categoria_general\": \"[Infraestructura|Salud|...]\", \"titulo_propuesta\": \"[Acción + Qué + Dónde]\"}"
            )
            usr = f"Propuesta: {texto_a_clasificar}\nUbicación: {ubicacion}"
        
        try:
            out = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr}
                ],
                temperature=0.2,
                max_tokens=150
            ).choices[0].message.content.strip()
            
            out = out.replace("```json", "").replace("```", "").strip()
            data = json.loads(out)
            
        except Exception as e:
            if tipo == "consulta":
                data = {"categoria_general": "Consulta", "titulo_propuesta": "Consulta ciudadana"}
            else:
                data = {"categoria_general": "General", "titulo_propuesta": "Propuesta ciudadana"}
        
        categoria = data.get("categoria_general", "General")
        titulo = data.get("titulo_propuesta", "Sin título")
        
        # Guardar (acumulativo)
        categorias_existentes = conv_data.get("categoria_general") or []
        titulos_existentes = conv_data.get("titulo_propuesta") or []
        
        if isinstance(categorias_existentes, str):
            categorias_existentes = [categorias_existentes]
        if isinstance(titulos_existentes, str):
            titulos_existentes = [titulos_existentes]
        
        if categoria not in categorias_existentes:
            categorias_existentes.append(categoria)
        
        titulo_normalizado = _normalize_text(titulo)
        titulos_normalizados_existentes = [_normalize_text(t) for t in titulos_existentes]
        
        if titulo_normalizado not in titulos_normalizados_existentes:
            titulos_existentes.append(titulo)
        
        conv_ref.update({
            "categoria_general": categorias_existentes,
            "titulo_propuesta": titulos_existentes,
            "ultima_fecha": firestore.SERVER_TIMESTAMP
        })
        
        return {
            "ok": True,
            "clasificacion": {
                "tipo": tipo,
                "categoria_general": categoria,
                "titulo_propuesta": titulo,
                "todas_categorias": categorias_existentes,
                "todos_titulos": titulos_existentes
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)