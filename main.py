# ============================
#  WilderBot API (FastAPI) - VERSIÓN REFACTORIZADA CON SISTEMA DE CAPAS
#  - Sistema híbrido multicapa para optimización de velocidad y costos
#  - Memoria conversacional real con contexto inteligente
#  - Extracción de datos por capas (Regex → Heurísticas → LLM)
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

CONSULTA_TITULOS = [
    "Vida Personal Wilder", "General", "Leyes", "Movilidad",
    "Educación", "Salud", "Seguridad", "Vivienda", "Empleo"
]

PRIVACY_REPLY = (
    "Entiendo perfectamente que quieras proteger tu información personal. "
    "Queremos que sepas que tus datos están completamente seguros con nosotros. "
    "Cumplimos con todas las normativas de protección de datos y solo usamos tu información "
    "para escalar tu propuesta y ayudar a que pueda hacerse realidad."
)

# === Clientes ===
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

print(f"[RAG] Pinecone index en uso: {PINECONE_INDEX}")
print(f"[RAG] Modelo de embeddings: {EMBEDDING_MODEL}")

# === Firestore Admin ===
try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate(GOOGLE_CREDS)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# =========================================================
#  Esquemas de entrada
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

# =========================================================
#  Health
# =========================================================

@app.get("/health")
async def health():
    return {"status": "ok"}

# =========================================================
#  Text Utils
# =========================================================

def _normalize_text(t: str) -> str:
    t = t.lower()
    t = (t.replace("á","a").replace("é","e").replace("í","i")
           .replace("ó","o").replace("ú","u").replace("ü","u"))
    t = re.sub(r"[^a-zñ0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

DISCOURSE_START_WORDS = {
    _normalize_text(w) for w in [
        "hola", "holaa", "holaaa", "buenas", "buenos días", "buen día", "saludos",
        "gracias", "ok", "okay", "oki", "vale", "de acuerdo", "listo", "listos",
        "bueno", "entendido", "hecho", "claro", "claro que sí", "sí", "si",
        "perfecto", "hey", "dale", "de una", "genial", "súper", "super"
    ]
}

INTERROGATIVOS = [
    "que", "qué", "como", "cómo", "cuando", "cuándo", "donde", "dónde",
    "por que", "por qué", "cual", "cuál", "quien", "quién",
    "me gustaria saber", "quisiera saber", "podria decirme",
    "puedes explicarme", "informacion", "información"
]

def limit_sentences(text: str, max_sentences: int = 3) -> str:
    parts = re.split(r'(?<=[\.\?!])\s+', text.strip())
    out = " ".join([p for p in parts if p][:max_sentences]).strip()
    return out or text

def _titlecase(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).title()

def _clean_barrio_fragment(s: str) -> str:
    s = re.split(r"\s+(?:para|por|que|donde|con)\b|[,.;]|$", s, maxsplit=1, flags=re.IGNORECASE)[0]
    return _titlecase(s)

# =========================================================
#  Helpers de BD
# =========================================================

def upsert_usuario_o_anon(
    chat_id: str,
    nombre: Optional[str],
    telefono: Optional[str],
    canal: Optional[str],
    barrio: Optional[str] = None
) -> str:
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
            "current_proposal": None,
            "contact_intent": None,
            "contact_requested": False,
            "contact_collected": False,
            "contact_refused": False,
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
    snap = conv_ref.get()
    data = snap.to_dict() or {}
    arr = data.get("mensajes", [])
    arr.extend(nuevos)
    conv_ref.update({"mensajes": arr, "ultima_fecha": firestore.SERVER_TIMESTAMP})

    try:
        resumen = summarize_conversation_brief(arr, max_chars=100)
        conv_ref.update({"resumen": resumen, "resumen_updated_at": firestore.SERVER_TIMESTAMP})
    except Exception:
        pass
    
    try:
        update_conversation_summary(conv_ref)
    except Exception as e:
        print(f"[WARN] Conversation summary update failed: {e}")

def load_historial_para_prompt(conv_ref) -> List[Dict[str, str]]:
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

def update_conversation_summary(conv_ref, force: bool = False):
    snap = conv_ref.get()
    data = snap.to_dict() or {}
    
    msg_count = int(data.get("messages_since_summary", 0)) + 1
    
    if msg_count < 4 and not force:
        conv_ref.update({"messages_since_summary": msg_count})
        return
    
    all_messages = data.get("mensajes", [])
    
    if len(all_messages) < 4:
        conv_ref.update({"messages_since_summary": msg_count})
        return
    
    recent = all_messages[-12:]
    transcript = []
    for m in recent:
        role = "U" if m["role"] == "user" else "B"
        content = (m.get("content", "") or "")[:180]
        transcript.append(f"{role}: {content}")
    
    transcript_text = "\n".join(transcript)
    
    sys = (
        "Resume en MÁXIMO 140 caracteres: tema, propuesta (si hay), fase actual.\n"
        "Ejemplo: 'Consultó ley fauna. Propuso mejorar alumbrado Milán. Dio argumento seguridad.'\n"
        "Sin nombres ni teléfonos."
    )
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
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
            "summary_updated_at": firestore.SERVER_TIMESTAMP,
            "messages_since_summary": 0
        })
        
    except Exception as e:
        conv_ref.update({"messages_since_summary": msg_count})
        print(f"[WARN] Summary update failed: {e}")

def summarize_conversation_brief(mensajes: List[Dict[str, str]], max_chars: int = 100) -> str:
    parts = []
    for m in mensajes[-40:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        content = re.sub(r"\+?\d[\d\s\-]{6,}", "[número]", content)
        tag = "C" if role == "user" else "B" if role == "assistant" else role
        parts.append(f"{tag}: {content}")
    transcript = "\n".join(parts) if parts else "(sin mensajes)"

    sys = (
        "Eres un asistente que resume conversaciones cívicas.\n"
        "Devuelve SOLO un resumen en español de MÁXIMO 100 caracteres (contando espacios).\n"
        "No incluyas nombres, teléfonos ni datos personales.\n"
        "Enfócate en la petición/tema principal y el lugar si existe.\n"
        "No uses comillas."
    )
    usr = f"Transcripción:\n{transcript}\n\nEscribe el resumen breve (≤100):"

    try:
        out = client.chat.completions.create(
            model=OPENAI_MODEL_SUMMARY,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            temperature=0.2,
            max_tokens=60,
        ).choices[0].message.content
    except Exception:
        out = (mensajes[0].get("content") if mensajes else "")[:max_chars]

    return re.sub(r"\s+", " ", (out or "").strip())[:max_chars]

# ═══════════════════════════════════════════════════════════════════════
# CAPA 0: SISTEMA DE CONTEXTO INTELIGENTE
# ═══════════════════════════════════════════════════════════════════════

class ConversationContext:
    """Contexto completo de la conversación para decisiones inteligentes."""
    
    def __init__(self, conv_data: dict, mensaje: str):
        self.mensaje = mensaje
        self.mensaje_norm = _normalize_text(mensaje)
        
        # Estado del flujo
        self.in_proposal_flow = bool(
            conv_data.get("proposal_collected") or
            conv_data.get("proposal_requested") or
            conv_data.get("argument_requested") or
            conv_data.get("contact_intent") == "propuesta"
        )
        
        self.contact_requested = bool(conv_data.get("contact_requested"))
        self.contact_collected = bool(conv_data.get("contact_collected"))
        self.contact_refused = bool(conv_data.get("contact_refused"))
        self.argument_requested = bool(conv_data.get("argument_requested"))
        self.argument_collected = bool(conv_data.get("argument_collected"))
        self.proposal_collected = bool(conv_data.get("proposal_collected"))
        
        # Datos parciales guardados
        self.contact_info = conv_data.get("contact_info") or {}
        self.project_location = conv_data.get("project_location")
        self.current_proposal = conv_data.get("current_proposal")
        
        # Resumen conversacional (100 chars)
        self.resumen = conv_data.get("conversacion_resumida", "")
        
        # Historial reciente (últimos 8 mensajes)
        mensajes = conv_data.get("mensajes", [])
        self.historial = [
            {"role": m["role"], "content": m["content"]}
            for m in mensajes[-8:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        
        # Detección de referencias a mensajes anteriores
        self.has_reference = self._detect_reference()
    
    def _detect_reference(self) -> bool:
        referencias = [
            "eso", "esa", "ese", "lo que", "la que", "el que",
            "lo anterior", "lo que dijiste", "lo que dije",
            "sobre eso", "de eso", "aquello", "aquel",
            "el primero", "el segundo", "la primera", "esa ley",
            "ese proyecto", "esa propuesta"
        ]
        return any(ref in self.mensaje_norm for ref in referencias)
    
    def get_missing_contact_fields(self) -> List[str]:
        missing = []
        if not self.contact_info.get("nombre"):
            missing.append("nombre")
        if not self.contact_info.get("barrio"):
            missing.append("barrio")
        if not self.contact_info.get("telefono"):
            missing.append("celular")
        return missing

# ═══════════════════════════════════════════════════════════════════════
# DETECCIÓN DE PATRONES
# ═══════════════════════════════════════════════════════════════════════

def is_plain_greeting(text: str) -> bool:
    if not text:
        return False
    t = _normalize_text(text)
    kws = ("hola","holaa","holaaa","buenas","buenos dias","buenas tardes","buenas noches","como estas","que mas","q mas","saludos")
    short = len(t) <= 40
    has_kw = any(k in t for k in kws)
    topicish = any(w in t for w in (
        "arregl","propuesta","proponer","daño","danada","hueco","parque","colegio","via",
        "salud","seguridad","ayuda","necesito","quiero","repar","denuncia","idea",
        "queja","reclamo","peticion","petición","tramite","trámite"
    ))
    return short and has_kw and not topicish

def detect_contact_refusal(text: str) -> bool:
    t = _normalize_text(text)
    patterns = [
        "no me gusta dar mis datos",
        "no quiero compartir mis datos",
        "prefiero no dar mis datos",
        "no doy mi celular",
        "no comparto informacion personal",
        "no comparto datos personales",
        "no quiero dar mi telefono",
        "no quiero dar mi numero",
    ]
    return any(p in t for p in patterns)

def is_proposal_denial(text: str) -> bool:
    t = _normalize_text(text)
    pats = [
        r'\b(aun|aún|todavia|todavía)\s+no\b.*\b(propuest|idea|sugerenc)',
        r'\b(no\s+tengo|no\s+he\s+hecho|no\s+te\s+he\s+hecho)\b.*\b(propuest|idea|sugerenc)',
        r'\b(no\s+es)\s+una\s+(propuesta|idea|sugerencia)\b',
        r'\b(olvidalo|olvídalo|mejor\s+no|ya\s+no|mas\s+tarde|más\s+tarde)\b'
    ]
    return any(re.search(p, t) for p in pats)

def is_proposal_intent_heuristic(text: str) -> bool:
    t = _normalize_text(text)
    if "me gustaria que" in t and re.search(r"\b(me\s+dig[ae]n|me\s+expliquen?|me\s+informen?)\b", t):
        return False
    
    kw = [
        "propongo", "propuesta", "sugerencia", "mi idea",
        "quiero proponer", "me gustaria proponer",
        "me gustaria mejorar", "quiero construir", "quisiera arreglar"
    ]
    return any(k in t for k in kw)

def looks_like_proposal_content(text: str) -> bool:
    if is_proposal_denial(text):
        return False

    raw = _normalize_text(text)
    if re.match(
        r'^(?:hola|holaa|buenas|buenos dias|buenas tardes|buenas noches|como estas|que mas|q mas|saludos)?\s*'
        r'(?:quiero|quisiera|me gustar(?:ia|ía))\s+(?:hacer\s+)?(?:una\s+)?(?:propuesta|idea|sugerencia)\s*[.!]?\s*$',
        raw
    ):
        return False

    t = _normalize_text(extract_proposal_text(text))
    if not t or t in {"algo","una idea","una propuesta","un tema","varias cosas"}:
        return False

    if re.match(r'^(?:quiero|quisiera|me gustar(?:ia|ía))\s+proponer\.?$', t):
        return False

    if re.search(r'\b(arregl|mejor|constru|instal|crear|paviment|ilumin|señaliz|ampli|dotar|regular(?!idad|mente)|prohib|mult|beca|subsid|limpi|recog|camar|pint|adecu)\w*', t):
        return True
    if re.search(r'\b(parque|anden|and[eé]n|semaforo|luminaria|cancha|juegos|polideportivo|colegio|hospital|bus|ruta|acera)\b', t):
        return True

    return len(t) >= 20 and not re.match(r'^(?:como estas\s+)?(?:quiero|quisiera|me gustar(?:ia|ía))\b', t)

def has_argument_text(t: str) -> bool:
    t = _normalize_text(t)
    
    obvious_keys = [
        "porque", "ya que", "debido", "necesitamos", "es importante",
        "para que", "con el fin", "urgente", "peligro"
    ]
    
    if any(k in t for k in obvious_keys):
        return True
    
    if len(t) >= 30:
        return True
    
    return False

def is_question_like(text: str) -> bool:
    t = _normalize_text(text)
    return (("?" in text) or any(k in t for k in INTERROGATIVOS)) and not is_proposal_intent_heuristic(text)

# ═══════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════════

def extract_user_name(text: str) -> Optional[str]:
    m = re.search(r'\b(?:soy|me llamo|mi nombre es)\s+([A-Za-zÁÉÍÓÚÑáéíóúñ ]{2,40})', text, flags=re.IGNORECASE)
    if m:
        nombre = m.group(1).strip(" .,")
        return nombre if _normalize_text(nombre) not in DISCOURSE_START_WORDS else None

    m = re.search(
        r'^\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})\s*(?=,|\s+vivo\b|\s+soy\b|\s+mi\b|\s+desde\b|\s+del\b|\s+de\b)',
        text
    )
    if m:
        posible = m.group(1).strip()
        low = _normalize_text(posible)
        if low in DISCOURSE_START_WORDS:
            return None
        if len(posible.split()) == 1 and len(posible) <= 3:
            return None
        return posible

    m = re.search(
        r'(?:^|[,;]\s*)(?:[A-Za-zÁÉÍÓÚÑáéíóúñ ]{0,20})?(?:claro(?:\s+que\s+s[ií])?|gracias|vale|ok|okay|perfecto|listo|de acuerdo)\s*,?\s*'
        r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})\s*,?\s*'
        r'(?:vivo|resido|mi\s+(?:n[uú]mero|tel[eé]fono|celular)|soy)\b',
        text, flags=re.IGNORECASE
    )
    if m:
        return m.group(1).strip(" .,")

    m = re.search(
        r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})\s*,?\s*(?:vivo|resido|mi\s+(?:n[uú]mero|tel[eé]fono|celular)|soy)\b',
        text, flags=re.IGNORECASE
    )
    if m:
        posible = m.group(1).strip(" .,")
        return posible if _normalize_text(posible) not in DISCOURSE_START_WORDS else None

    return None

def extract_phone(text: str) -> Optional[str]:
    m = re.search(r'(\+?\d[\d\s\-]{7,16}\d)', text)
    if not m:
        return None
    tel = re.sub(r'\D', '', m.group(1))
    tel = re.sub(r'^(?:00)?57', '', tel)
    return tel if 8 <= len(tel) <= 12 else None

def extract_user_barrio(text: str) -> Optional[str]:
    m = re.search(
        r'\b(?:vivo|resido)\s+en\s+(?:el\s+)?(?:barrio\s+)?'
        r'([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{1,49}?)'
        r'(?=(?:\s+(?:y|mi|n[uú]mero|tel[eé]fono|celular|desde|de|del|con|para|por)\b|[,.;]|$))',
        text, flags=re.IGNORECASE
    )
    if m:
        return _clean_barrio_fragment(m.group(1))

    m = re.search(r'\bsoy\s+del\s+barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50})', text, flags=re.IGNORECASE)
    if m:
        return _clean_barrio_fragment(m.group(1))

    m = re.search(r'\bmi\s+barrio\s+(?:es\s+)?([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50})', text, flags=re.IGNORECASE)
    if m:
        return _clean_barrio_fragment(m.group(1))

    return None

def extract_project_location(text: str) -> Optional[str]:
    m = re.search(
        r'\b(?:en\s+el\s+|en\s+)barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50}?)(?=(?:\s+(?:para|por|que|donde|con|cerca|y)\b|[,.;]|$))',
        text, flags=re.IGNORECASE
    )
    if m:
        left = text[:m.start()].lower()
        if re.search(r'(vivo|resido)\s+en\s*$', left[-25:]):
            pass
        else:
            return _clean_barrio_fragment(m.group(1))

    m = re.search(
        r'\b(?:del\s+|de\s+el\s+)barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50}?)(?=(?:\s+(?:cerca|para|por|que|donde|con|y)\b|[,.;]|$))',
        text, flags=re.IGNORECASE
    )
    if m:
        return _clean_barrio_fragment(m.group(1))

    if re.search(r'\b(construir|hacer|instalar|crear|mejorar|arreglar|reparar|pintar|adecuar|señalizar)\b', text, flags=re.IGNORECASE):
        m = re.search(
            r'\bbarrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50}?)(?=(?:\s+(?:cerca|para|por|que|donde|con|y)\b|[,.;]|$))',
            text, flags=re.IGNORECASE
        )
        if m:
            return _clean_barrio_fragment(m.group(1))

    return None

def extract_proposal_text(text: str) -> str:
    t = text.strip()

    t = re.sub(
        r'^\s*(?:hola|holaa+|buenas(?:\s+(?:tardes|noches|dias))?|buen(?:\s*dia)?|hey|que tal|qué tal|saludos)[,!\s\-–—]*',
        '', t, flags=re.IGNORECASE
    )

    m = re.search(
        r'(?:me\s+gustar[íi]a\s+(?:proponer|que|hacer\s+una\s+propuesta)|'
        r'quisiera\s+(?:proponer|que|hacer\s+una\s+propuesta)|'
        r'quiero\s+(?:proponer|hacer\s+una\s+propuesta)|'
        r'tengo\s+una\s+(?:propuesta|idea)|'
        r'propongo\s+que|propongo|'
        r'mi\s+(?:idea|propuesta)\s+(?:es|ser[ií]a))\s*[:\-–—]?\s*(.*)',
        t, flags=re.IGNORECASE
    )
    if m:
        t = m.group(1).strip()
    else:
        m2 = re.search(r'(?:propuesta|idea)\s+(?:es|ser[ií]a)\s*[:\-–—]?\s*(.*)', t, flags=re.IGNORECASE)
        if m2:
            t = m2.group(1).strip()

    t = re.sub(r'\s+', ' ', t).strip()
    return limit_sentences(t, 2)

def llm_extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    sys = (
        "Extrae información de contacto del mensaje.\n"
        "Devuelve SOLO JSON con claves: nombre, barrio, telefono\n"
        "Si algo no está presente, usa null.\n\n"
        "Ejemplos:\n"
        "- 'Juliana Salazar, vivo en Miñitas, 3168207240' → {\"nombre\": \"Juliana Salazar\", \"barrio\": \"Miñitas\", \"telefono\": \"3168207240\"}\n"
        "- 'Juan Pérez del barrio Centro' → {\"nombre\": \"Juan Pérez\", \"barrio\": \"Centro\", \"telefono\": null}\n"
        "- 'Mi número es 3001234567' → {\"nombre\": null, \"barrio\": null, \"telefono\": \"3001234567\"}\n"
    )
    
    usr = f"Mensaje: {text}\n\nExtrae la información y devuelve el JSON."
    
    try:
        out = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            temperature=0.0,
            max_tokens=150
        ).choices[0].message.content.strip()
        
        data = json.loads(out)
        return {
            "nombre": data.get("nombre"),
            "barrio": data.get("barrio"),
            "telefono": data.get("telefono")
        }
    except:
        return {"nombre": None, "barrio": None, "telefono": None}

# ═══════════════════════════════════════════════════════════════════════
# CAPA 1: RESPUESTAS DETERMINISTAS
# ═══════════════════════════════════════════════════════════════════════

def layer1_deterministic_response(ctx: ConversationContext, conv_data: dict) -> Optional[str]:
    """Capa 1: Respuestas instantáneas sin LLM."""
    
    # 1. Saludo inicial sin historial
    if not ctx.historial and is_plain_greeting(ctx.mensaje):
        print("[LAYER1] ✓ Saludo inicial")
        return BOT_INTRO_TEXT
    
    # 2. Rechazo de datos
    if ctx.contact_requested and not ctx.contact_collected:
        if detect_contact_refusal(ctx.mensaje):
            refusal_count = int(conv_data.get("contact_refusal_count", 0))
            print(f"[LAYER1] ✓ Rechazo de datos (intento {refusal_count + 1})")
            
            if refusal_count == 0:
                return PRIVACY_REPLY + " ¿Me compartes tus datos para poder ayudarte?"
            else:
                return (
                    "Entiendo tu decisión y la respeto completamente. "
                    "Si en algún momento cambias de opinión, estaré aquí para ayudarte. "
                    "¡Que tengas un excelente día!"
                )
    
    # 3. Negación de propuesta
    if is_proposal_denial(ctx.mensaje):
        print("[LAYER1] ✓ Negación de propuesta")
        return "Perfecto, sin problema. Cuando la tengas, cuéntamela en 1–2 frases y el barrio del proyecto."
    
    return None

# ═══════════════════════════════════════════════════════════════════════
# CAPA 2: EXTRACCIÓN HÍBRIDA DE DATOS
# ═══════════════════════════════════════════════════════════════════════

def layer2_extract_contact_data(ctx: ConversationContext) -> Dict[str, Optional[str]]:
    """Capa 2: Extracción por capas (Regex → Heurísticas → LLM)."""
    
    # Subcapa 2A: Regex
    name_regex = extract_user_name(ctx.mensaje)
    phone_regex = extract_phone(ctx.mensaje)
    barrio_regex = extract_user_barrio(ctx.mensaje)
    
    if name_regex and phone_regex and barrio_regex:
        print("[LAYER2A] ✓ Regex completo")
        return {"nombre": name_regex, "telefono": phone_regex, "barrio": barrio_regex, "_method": "regex_full"}
    
    # Subcapa 2B: Heurísticas
    if ctx.contact_requested:
        if not name_regex:
            m = re.search(
                r'\b(?:claro|ok|okay|si|sí|perfecto|listo|vale|bueno|de acuerdo)\s*,?\s*'
                r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})',
                ctx.mensaje, flags=re.IGNORECASE
            )
            if m:
                nombre = m.group(1).strip(" .,")
                if _normalize_text(nombre) not in DISCOURSE_START_WORDS:
                    name_regex = nombre
                    print("[LAYER2B] ✓ Nombre por heurística")
        
        if not phone_regex:
            m = re.search(r'\b(\d{10})\b', ctx.mensaje)
            if m:
                phone_regex = m.group(1)
                print("[LAYER2B] ✓ Teléfono por heurística")
        
        if not barrio_regex:
            words = ctx.mensaje.split()
            if len(words) <= 3 and words[0][0].isupper():
                barrio_regex = " ".join(words).strip(" .,")
                print("[LAYER2B] ✓ Barrio por heurística")
    
    if name_regex or phone_regex or barrio_regex:
        print(f"[LAYER2B] ✓ Parcial: name={bool(name_regex)}, phone={bool(phone_regex)}, barrio={bool(barrio_regex)}")
        result = {}
        if name_regex: result["nombre"] = name_regex
        if phone_regex: result["telefono"] = phone_regex
        if barrio_regex: result["barrio"] = barrio_regex
        result["_method"] = "heuristic"
        return result
    
    # Subcapa 2C: LLM
    if ctx.contact_requested and len(ctx.mensaje.split()) >= 8:
        print("[LAYER2C] Llamando LLM para extracción...")
        llm_data = llm_extract_contact_info(ctx.mensaje)
        
        if any(llm_data.values()):
            print(f"[LAYER2C] ✓ LLM: name={bool(llm_data.get('nombre'))}, phone={bool(llm_data.get('telefono'))}, barrio={bool(llm_data.get('barrio'))}")
            llm_data["_method"] = "llm"
            return llm_data
    
    return {}

# ═══════════════════════════════════════════════════════════════════════
# CAPA 3: CLASIFICACIÓN INTELIGENTE
# ═══════════════════════════════════════════════════════════════════════

def layer3_classify_with_context(ctx: ConversationContext) -> Dict[str, Any]:
    """Capa 3: Clasificación híbrida con contexto."""
    
    if ctx.in_proposal_flow:
        return {"tipo": "propuesta", "confianza": "alta", "method": "flow_active"}
    
    # Heurísticas
    if is_proposal_intent_heuristic(ctx.mensaje):
        return {"tipo": "propuesta", "confianza": "alta", "method": "heuristic_proposal"}
    
    if "?" in ctx.mensaje:
        if any(kw in ctx.mensaje_norm for kw in ["wilder", "ley", "proyecto", "que propone"]):
            return {"tipo": "consulta", "confianza": "alta", "method": "heuristic_question"}
    
    # LLM con resumen
    contexto = ctx.resumen if ctx.resumen else "Inicio de conversación"
    
    if ctx.has_reference and ctx.historial:
        ultimo_bot = next((h["content"] for h in reversed(ctx.historial) if h["role"] == "assistant"), "")
        if ultimo_bot:
            contexto += f" | Último: {ultimo_bot[:80]}"
    
    sys = (
        f"Contexto: {contexto}\n"
        f"Mensaje: {ctx.mensaje}\n\n"
        "¿Es 'consulta' (info) o 'propuesta' (acción)?\n"
        "Responde: consulta|propuesta"
    )
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys}],
            temperature=0,
            max_tokens=5,
            timeout=2
        )
        
        answer = response.choices[0].message.content.strip().lower()
        tipo = "propuesta" if "propuesta" in answer else "consulta"
        
        print(f"[LAYER3] ✓ LLM classify: {tipo}")
        return {"tipo": tipo, "confianza": "media", "method": "llm_context"}
        
    except Exception as e:
        print(f"[LAYER3] ✗ LLM failed: {e}")
        return {"tipo": "consulta", "confianza": "baja", "method": "fallback"}

# ═══════════════════════════════════════════════════════════════════════
# RAG
# ═══════════════════════════════════════════════════════════════════════

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

def reformulate_query_with_context(
    current_query: str, 
    conversation_history: List[Dict[str, str]],
    max_history: int = 6
) -> str:
    if len(current_query.split()) > 8 and not any(
        ref in current_query.lower() 
        for ref in ["ella", "él", "eso", "esa", "ese", "lo", "la", "el", "primero", "segundo"]
    ):
        return current_query
    
    recent = conversation_history[-max_history:] if conversation_history else []
    
    if not recent:
        return current_query
    
    transcript = []
    for msg in recent:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        content = msg.get("content", "")[:300]
        transcript.append(f"{role}: {content}")
    
    transcript_text = "\n".join(transcript)
    
    system_prompt = f"""Contexto: {transcript_text}

Query: {current_query}

Reformula para que sea autónoma. Si tiene referencias vagas, reemplázalas.
Máximo 15 palabras. Solo términos clave.
Responde SOLO la query reformulada."""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.1,
            max_tokens=40,
            timeout=2
        )
        
        reformulated = response.choices[0].message.content.strip()
        
        if len(reformulated.split()) < 2:
            reformulated = current_query
        
        print(f"[QUERY] Reformulada: '{reformulated}'")
        return reformulated
        
    except Exception as e:
        print(f"[WARN] Query reformulation failed: {e}")
        return current_query

# ═══════════════════════════════════════════════════════════════════════
# HELPERS DE CONTACTO
# ═══════════════════════════════════════════════════════════════════════

def build_contact_request(missing: List[str]) -> str:
    etiquetas = {
        "nombre": "tu nombre",
        "barrio": "tu barrio",
        "celular": "un número de contacto",
        "project_location": "el barrio del proyecto",
    }
    pedir = [etiquetas[m] for m in missing if m in etiquetas]
    if not pedir:
        return "¿Me confirmas por favor los datos pendientes?"
    frase = pedir[0] if len(pedir) == 1 else (", ".join(pedir[:-1]) + " y " + pedir[-1])
    return f"Para escalar y darle seguimiento, ¿me compartes {frase}? Lo usamos solo para informarte avances."

def build_project_location_request() -> str:
    return (
        "Para ubicar el caso en el mapa: ¿en qué barrio sería exactamente el proyecto? "
        "Si ya lo mencionaste, recuérdamelo por favor."
    )

def craft_argument_question(name: Optional[str], project_location: Optional[str] = None) -> str:
    saludo = f"Hola, {name}. " if name else ""
    lugar = f" en el barrio {project_location}" if project_location else ""
    return (
        f"{saludo}Gracias por compartir tu idea{lugar}. "
        "¿Nos cuentas en pocas palabras por qué es importante, a quién beneficiaría y qué problema ayuda a resolver?"
    )

def positive_ack_and_request_argument(name: Optional[str], project_location: Optional[str] = None) -> str:
    saludo = f"Gracias, {name}. " if name else "¡Gracias por contarme tu idea! "
    lugar = f" para el barrio {project_location}" if project_location else ""
    return (
        f"{saludo}Tu propuesta{lugar} suena muy valiosa para la comunidad. "
        "Para avanzar, ¿podrías contar en pocas palabras por qué sería importante, "
        "a quién beneficiaría y qué problema ayudaría a resolver?"
    )

CONTACT_PATTERNS = re.compile(
    r"(compart\w+|env(í|i)a\w*|dime|indíca\w*|facilita\w*|regálame|regalame|me\s+das|me\s+dejas).{0,60}"
    r"(tu\s+)?(nombre|barrio|celular|tel[eé]fono|n[uú]mero|contacto)",
    re.IGNORECASE
)

def strip_contact_requests(texto: str) -> str:
    sent_split = re.split(r'(?<=[\.\?!])\s+', texto.strip())
    limpio = [s for s in sent_split if not CONTACT_PATTERNS.search(s)]
    if limpio:
        out = " ".join([s for s in limpio if s]).strip()
        return out if out else texto

    cleaned = CONTACT_PATTERNS.sub("", texto).strip()
    return cleaned if len(_normalize_text(cleaned)) >= 5 else \
        "¿Nos cuentas brevemente por qué sería importante y a quién beneficiaría?"

# ═══════════════════════════════════════════════════════════════════════
# CAPA 4: GENERACIÓN CON MEMORIA
# ═══════════════════════════════════════════════════════════════════════

def layer4_generate_response_with_memory(
    ctx: ConversationContext,
    clasificacion: Dict[str, Any],
    rag_hits: List[Dict]
) -> str:
    """Capa 4: Generación completa con RAG y memoria conversacional."""
    
    contexto_rag = "\n".join([f"- {h['texto']}" for h in rag_hits if h.get("texto")])
    
    system_msg = (
        "Actúa como Wilder Escobar, Representante a la Cámara.\n"
        "Tono cercano. Máximo 3-4 frases.\n\n"
    )
    
    if ctx.has_reference and ctx.resumen:
        system_msg += f"CONTEXTO PREVIO: {ctx.resumen}\n"
        system_msg += "El usuario hace referencia a algo anterior. Mantén coherencia.\n\n"
    
    if clasificacion["tipo"] == "consulta":
        system_msg += "Responde la CONSULTA. No pidas datos personales.\n"
    else:
        system_msg += "Es PROPUESTA. Tono positivo.\n"
    
    msgs = [{"role": "system", "content": system_msg}]
    
    if ctx.historial:
        msgs.extend(ctx.historial[-6:])
    
    msgs.append({
        "role": "user",
        "content": f"Contexto:\n{contexto_rag}\n\nMensaje:\n{ctx.mensaje}"
    })
    
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.3,
            max_tokens=280
        )
        
        texto = completion.choices[0].message.content.strip()
        texto = limit_sentences(texto, 3)
        
        print(f"[LAYER4] ✓ Respuesta generada")
        return texto
        
    except Exception as e:
        print(f"[LAYER4] ✗ Error: {e}")
        return "Disculpa, tuve un problema. ¿Podrías reformular?"

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT PRINCIPAL REFACTORIZADO
# ═══════════════════════════════════════════════════════════════════════

@app.post("/responder")
async def responder(data: Entrada):
    """Endpoint principal con sistema de capas completo."""
    try:
        # SETUP
        chat_id = data.chat_id or f"web_{os.urandom(4).hex()}"
        usuario_id = upsert_usuario_o_anon(
            chat_id, data.nombre or data.usuario, data.celular, data.canal
        )
        conv_ref = ensure_conversacion(chat_id, usuario_id, data.faq_origen, data.canal)
        conv_data = conv_ref.get().to_dict() or {}
        
        # CAPA 0: Contexto
        ctx = ConversationContext(conv_data, data.mensaje)
        print(f"\n{'='*60}")
        print(f"[CTX] Mensaje: '{data.mensaje[:50]}...'")
        print(f"[CTX] Flujo: proposal={ctx.in_proposal_flow}, contact={ctx.contact_requested}")
        print(f"[CTX] Referencia: {ctx.has_reference}")
        print(f"{'='*60}\n")
        
        # CAPA 1: Determinista
        layer1_response = layer1_deterministic_response(ctx, conv_data)
        if layer1_response:
            if is_proposal_denial(data.mensaje):
                conv_ref.update({
                    "proposal_requested": False,
                    "proposal_collected": False,
                    "current_proposal": None,
                    "argument_requested": False,
                    "argument_collected": False,
                    "contact_intent": None,
                    "contact_requested": False,
                    "contact_refused": False,
                })
            
            if detect_contact_refusal(data.mensaje):
                refusal_count = int(conv_data.get("contact_refusal_count", 0))
                conv_ref.update({
                    "contact_refused": True,
                    "contact_refusal_count": refusal_count + 1
                })
            
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": layer1_response}
            ])
            return {"respuesta": layer1_response, "fuentes": [], "chat_id": chat_id}
        
        # CAPA 2: Extracción
        extracted_data = layer2_extract_contact_data(ctx)
        
        if extracted_data:
            current_info = ctx.contact_info.copy()
            
            if extracted_data.get("nombre") and not current_info.get("nombre"):
                current_info["nombre"] = extracted_data["nombre"]
            if extracted_data.get("telefono"):
                current_info["telefono"] = extracted_data["telefono"]
            if extracted_data.get("barrio"):
                current_info["barrio"] = extracted_data["barrio"]
            
            conv_ref.update({"contact_info": current_info})
            
            if current_info.get("telefono"):
                conv_ref.update({"contact_collected": True})
                upsert_usuario_o_anon(
                    chat_id,
                    current_info.get("nombre"),
                    current_info.get("telefono"),
                    data.canal,
                    current_info.get("barrio")
                )
            
            ctx.contact_info = current_info
            ctx.contact_collected = bool(current_info.get("telefono"))
        
        proj_loc = extract_project_location(data.mensaje)
        if proj_loc:
            conv_ref.update({"project_location": proj_loc})
            ctx.project_location = proj_loc
        
        # Cierre si datos completos
        if ctx.contact_requested:
            tiene_nombre = bool(ctx.contact_info.get("nombre"))
            tiene_tel = bool(ctx.contact_info.get("telefono"))
            tiene_barrio = bool(ctx.contact_info.get("barrio"))
            tiene_ubicacion = bool(ctx.project_location)
            
            if tiene_nombre and tiene_tel and tiene_barrio and tiene_ubicacion:
                nombre_txt = ctx.contact_info.get("nombre", "")
                texto = (f"Gracias, {nombre_txt}. " if nombre_txt else "Gracias. ")
                texto += "Con estos datos escalamos el caso y te contamos avances."
                
                append_mensajes(conv_ref, [
                    {"role": "user", "content": data.mensaje},
                    {"role": "assistant", "content": texto}
                ])
                return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
            
            missing = ctx.get_missing_contact_fields()
            if not tiene_ubicacion:
                missing.append("project_location")
            
            if missing:
                if missing == ["project_location"]:
                    texto = build_project_location_request()
                else:
                    texto = build_contact_request(missing)
                
                append_mensajes(conv_ref, [
                    {"role": "user", "content": data.mensaje},
                    {"role": "assistant", "content": texto}
                ])
                return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
        
        # CAPA 3: Clasificación
        clasificacion = layer3_classify_with_context(ctx)
        print(f"[CLASSIFY] {clasificacion}")
        
        # FLUJO PROPUESTAS (simplificado para espacio)
        if clasificacion["tipo"] == "propuesta" or ctx.in_proposal_flow:
            # Lógica de propuestas existente
            if not ctx.proposal_collected:
                if looks_like_proposal_content(data.mensaje):
                    conv_ref.update({
                        "current_proposal": extract_proposal_text(data.mensaje),
                        "proposal_collected": True,
                        "argument_requested": True,
                    })
                    texto = positive_ack_and_request_argument(
                        ctx.contact_info.get("nombre"),
                        ctx.project_location
                    )
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                else:
                    conv_ref.update({"proposal_requested": True})
                    texto = "¡Perfecto! ¿Cuál es tu propuesta? Cuéntamela en una o dos frases."
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
            
            if ctx.proposal_collected and not ctx.argument_collected:
                if has_argument_text(data.mensaje) or len(_normalize_text(data.mensaje)) >= 20:
                    conv_ref.update({"argument_collected": True, "contact_requested": True})
                    missing = ctx.get_missing_contact_fields()
                    texto = build_contact_request(missing or ["nombre", "barrio", "celular"])
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                else:
                    texto = craft_argument_question(
                        ctx.contact_info.get("nombre"),
                        ctx.project_location
                    )
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
        
        # CAPA 4: RAG + Generación
        query_for_search = data.mensaje
        if ctx.has_reference:
            query_for_search = reformulate_query_with_context(
                data.mensaje, ctx.historial, max_history=6
            )
        
        hits = rag_search(query_for_search, top_k=5)
        
        texto = layer4_generate_response_with_memory(ctx, clasificacion, hits)
        
        if not ctx.contact_requested:
            texto = strip_contact_requests(texto)
        
        append_mensajes(conv_ref, [
            {"role": "user", "content": data.mensaje},
            {"role": "assistant", "content": texto}
        ])
        
        return {"respuesta": texto, "fuentes": hits, "chat_id": chat_id}
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# ═══════════════════════════════════════════════════════════════════════
# CLASIFICACIÓN (mantenida simple)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/clasificar")
async def clasificar(body: ClasificarIn):
    """Endpoint de clasificación simplificado."""
    try:
        chat_id = body.chat_id
        conv_ref = db.collection("conversaciones").document(chat_id)
        snap = conv_ref.get()
        
        if not snap.exists:
            return {"ok": False, "mensaje": "Conversación no encontrada"}
        
        conv_data = snap.to_dict() or {}
        
        # Obtener propuesta si existe
        propuesta = conv_data.get("current_proposal") or ""
        
        if not propuesta:
            return {"ok": True, "skipped": True, "reason": "sin_propuesta"}
        
        # Clasificar con LLM básico
        sys = "Clasifica esta propuesta ciudadana. Responde JSON: {\"categoria_general\": \"...\", \"titulo_propuesta\": \"...\", \"tono_detectado\": \"positivo|crítico|propositivo\"}"
        usr = f"Propuesta: {propuesta}"
        
        try:
            out = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                temperature=0.2,
                max_tokens=150
            ).choices[0].message.content.strip()
            
            data = json.loads(out)
            categoria = data.get("categoria_general", "General")
            titulo = data.get("titulo_propuesta", "Propuesta ciudadana")
            tono = data.get("tono_detectado", "propositivo")
            
            conv_ref.update({
                "categoria_general": [categoria],
                "titulo_propuesta": [titulo],
                "tono_detectado": tono,
            })
            
            return {"ok": True, "clasificacion": {
                "categoria_general": categoria,
                "titulo_propuesta": titulo,
                "tono_detectado": tono,
            }}
            
        except Exception as e:
            return {"ok": False, "error": str(e)}
        
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ═══════════════════════════════════════════════════════════════════════
# ARRANQUE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)