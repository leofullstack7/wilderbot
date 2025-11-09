# ============================
#  WilderBot API - VERSIÓN MEJORADA
#  Bot más inteligente y personalizado
#  Sistema híbrido optimizado (3 capas)
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "wilder-frases")
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "/etc/secrets/firebase.json"
OPENAI_MODEL_SUMMARY = os.getenv("OPENAI_MODEL_SUMMARY", "gpt-4o-mini")

BOT_INTRO_TEXT = os.getenv(
    "BOT_INTRO_TEXT",
    "¡Hola! Soy el asistente de Wilder Escobar. Estoy aquí para escuchar y canalizar tus "
    "problemas, propuestas o reconocimientos. ¿Qué te gustaría contarme hoy?"
)

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

# =========================================================
#  Utils
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
        "hola", "holaa", "holaaa", "buenas", "buenos días", "saludos",
        "gracias", "ok", "okay", "vale", "listo", "perfecto", "claro", "sí", "si"
    ]
}

def limit_sentences(text: str, max_sentences: int = 3) -> str:
    parts = re.split(r'(?<=[\.\?!])\s+', text.strip())
    out = " ".join([p for p in parts if p][:max_sentences]).strip()
    return out or text

def _titlecase(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).title()

def _clean_barrio_fragment(s: str) -> str:
    s = re.split(r"\s+(?:para|por|que|donde|con)\b|[,.;]|$", s, maxsplit=1, flags=re.IGNORECASE)[0]
    return _titlecase(s)

def remove_redundant_greetings(texto: str, historial: List[Dict[str, str]]) -> str:
    """Elimina saludos redundantes manteniendo naturalidad."""
    if historial and len(historial) > 0:
        greeting_patterns = [
            r'^¡?Hola!?\s*[,.]?\s*',
            r'^¡?Holaa!?\s*[,.]?\s*',
            r'^Buenas\s+(tardes|noches|días)[,.]?\s*',
            r'^¡?Qué\s+tal!?\s*[,.]?\s*',
            r'^¡?Saludos!?\s*[,.]?\s*',
        ]
        
        for pattern in greeting_patterns:
            texto = re.sub(pattern, '', texto, flags=re.IGNORECASE)
        
        texto = re.sub(r'^¡?Hola,?\s+[A-Za-záéíóúñ]+[,.]?\s*', '', texto, flags=re.IGNORECASE)
    
    texto = texto.strip()
    
    if not texto or len(texto) < 10:
        return "Claro, déjame ayudarte con eso."
    
    return texto

# =========================================================
#  BD Helpers
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

    # Resumen preliminar
    try:
        resumen = summarize_conversation_brief(arr, max_chars=100)
        conv_ref.update({"resumen": resumen})
    except:
        pass
    
    # Resumen completo
    try:
        resumen_completo = build_detailed_summary(data, arr)
        conv_ref.update({"resumen_completo": resumen_completo})
    except:
        pass
    
    # Resumen conversacional
    try:
        update_conversation_summary(conv_ref)
    except:
        pass

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

def build_detailed_summary(conv_data: dict, mensajes: List[Dict[str, str]]) -> dict:
    """Genera resumen estructurado completo para el botón "Ver más"."""
    
    summary = {
        "tema_principal": "",
        "consultas": [],
        "propuesta": None,
        "argumento": None,
        "ubicacion": None,
        "contacto": {
            "nombre": None,
            "barrio": None,
            "telefono": None
        },
        "estado": "iniciado",
        "historial_resumido": []
    }
    
    # Tema principal
    categorias = conv_data.get("categoria_general", [])
    titulos = conv_data.get("titulo_propuesta", [])
    
    if titulos:
        summary["tema_principal"] = titulos[-1]
    elif categorias:
        summary["tema_principal"] = categorias[-1]
    else:
        summary["tema_principal"] = "Conversación sin clasificar"
    
    # Consultas
    if "Consulta" in categorias:
        for titulo in titulos:
            if any(kw in _normalize_text(titulo) for kw in ["ley", "proyecto", "apoyo", "posicion", "posición"]):
                summary["consultas"].append(titulo)
    
    # Propuesta
    propuesta = conv_data.get("current_proposal")
    if propuesta:
        summary["propuesta"] = propuesta[:120] + ("..." if len(propuesta) > 120 else "")
    
    # Argumento
    if conv_data.get("argument_collected"):
        for m in mensajes:
            if m.get("role") == "user":
                content = m.get("content", "")
                if has_argument_text(content) and len(content) > 30:
                    summary["argumento"] = content[:120] + ("..." if len(content) > 120 else "")
                    break
    
    # Ubicación
    project_location = conv_data.get("project_location")
    if project_location:
        summary["ubicacion"] = project_location
    
    # Contacto
    contact_info = conv_data.get("contact_info", {})
    if contact_info:
        summary["contacto"] = {
            "nombre": contact_info.get("nombre"),
            "barrio": contact_info.get("barrio"),
            "telefono": contact_info.get("telefono")
        }
    
    # Estado
    if conv_data.get("contact_collected"):
        summary["estado"] = "completado"
    elif conv_data.get("contact_requested"):
        summary["estado"] = "esperando_contacto"
    elif conv_data.get("argument_collected"):
        summary["estado"] = "argumento_recibido"
    elif conv_data.get("proposal_collected"):
        summary["estado"] = "propuesta_recibida"
    elif "Consulta" in categorias:
        summary["estado"] = "consulta_respondida"
    
    # Historial
    recent = mensajes[-10:] if len(mensajes) > 10 else mensajes
    
    for m in recent:
        role = m.get("role")
        content = m.get("content", "")
        
        if role == "user":
            role_display = "Usuario"
        elif role == "assistant":
            role_display = "Asistente"
        else:
            continue
        
        content_short = content[:100] + ("..." if len(content) > 100 else "")
        
        summary["historial_resumido"].append({
            "rol": role_display,
            "mensaje": content_short
        })
    
    return summary

# =========================================================
#  CONTEXTO
# =========================================================

class ConversationContext:
    def __init__(self, conv_data: dict, mensaje: str):
        self.mensaje = mensaje
        self.mensaje_norm = _normalize_text(mensaje)
        
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
        
        self.contact_info = conv_data.get("contact_info") or {}
        self.project_location = conv_data.get("project_location")
        self.current_proposal = conv_data.get("current_proposal")
        self.resumen = conv_data.get("conversacion_resumida", "")
        
        mensajes = conv_data.get("mensajes", [])
        self.historial = [
            {"role": m["role"], "content": m["content"]}
            for m in mensajes[-8:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        
        self.has_reference = self._detect_reference()
    
    def _detect_reference(self) -> bool:
        referencias = [
            "eso", "esa", "ese", "lo que", "la que", "el que",
            "sobre eso", "de eso", "aquello", "aquel",
            "el primero", "el segundo", "esa ley", "ese proyecto"
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

# =========================================================
#  DETECCIÓN
# =========================================================

def is_plain_greeting(text: str) -> bool:
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

def detect_contact_refusal(text: str) -> bool:
    t = _normalize_text(text)
    return any(p in t for p in [
        "no me gusta dar mis datos",
        "no quiero compartir mis datos",
        "no doy mi celular"
    ])

def is_proposal_denial(text: str) -> bool:
    t = _normalize_text(text)
    pats = [
        r'\b(aun|aún|todavia)\s+no\b.*\b(propuest|idea)',
        r'\b(no\s+tengo|no\s+he\s+hecho)\b.*\b(propuest|idea)',
        r'\b(olvidalo|mejor\s+no|mas\s+tarde)\b'
    ]
    return any(re.search(p, t) for p in pats)

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

def is_proposal_intent_heuristic(text: str) -> bool:
    t = _normalize_text(text)
    kw = ["propongo", "propuesta", "sugerencia", "mi idea", "quiero proponer"]
    return any(k in t for k in kw)

def looks_like_proposal_content(text: str) -> bool:
    """Detecta si el texto contiene una propuesta CON CONTENIDO REAL."""
    
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
        text
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

def has_argument_text(text: str) -> bool:
    """Detecta si el texto parece ser un argumento."""
    t = _normalize_text(text)
    
    # Palabras clave de argumentación
    arg_keywords = [
        "porque", "por que", "importante", "necesario", "urgente",
        "ayudaria", "beneficiaria", "mejoraria", "solucionaria",
        "afecta", "problema", "situacion", "comunidad"
    ]
    
    tiene_keyword = any(kw in t for kw in arg_keywords)
    es_largo = len(t) >= 20
    
    return tiene_keyword or es_largo

# =========================================================
#  EXTRACCIÓN
# =========================================================

def extract_user_name(text: str) -> Optional[str]:
    m = re.search(r'\b(?:soy|me llamo|mi nombre es)\s+([A-Za-záéíóúñ ]{2,40})', text, flags=re.IGNORECASE)
    if m:
        nombre = m.group(1).strip(" .,")
        return nombre if _normalize_text(nombre) not in DISCOURSE_START_WORDS else None
    return None

def extract_phone(text: str) -> Optional[str]:
    m = re.search(r'(\+?\d[\d\s\-]{7,16}\d)', text)
    if not m:
        return None
    tel = re.sub(r'\D', '', m.group(1))
    tel = re.sub(r'^(?:00)?57', '', tel)
    return tel if 8 <= len(tel) <= 12 else None

def extract_user_barrio(text: str) -> Optional[str]:
    m = re.search(r'\b(?:vivo|resido)\s+en\s+(?:el\s+)?(?:barrio\s+)?([A-Za-záéíóúñ0-9 \-]{2,50})', text, flags=re.IGNORECASE)
    if m:
        return _clean_barrio_fragment(m.group(1))
    return None

def extract_project_location(text: str) -> Optional[str]:
    m = re.search(r'\ben\s+el\s+barrio\s+([A-Za-záéíóúñ0-9 \-]{2,50})', text, flags=re.IGNORECASE)
    if m:
        return _clean_barrio_fragment(m.group(1))
    return None

def extract_proposal_text(text: str) -> str:
    t = text.strip()
    t = re.sub(r'^\s*(?:hola|buenas)[,!\s\-]*', '', t, flags=re.IGNORECASE)
    return limit_sentences(t, 2)

def llm_extract_contact_info(text: str) -> Dict[str, Optional[str]]:
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

# =========================================================
#  CAPAS
# =========================================================

def layer1_deterministic_response(ctx: ConversationContext, conv_data: dict) -> Optional[str]:
    if not ctx.historial and is_plain_greeting(ctx.mensaje):
        print("[LAYER1] ✓ Saludo inicial")
        return BOT_INTRO_TEXT
    
    if ctx.contact_requested and not ctx.contact_collected:
        if detect_contact_refusal(ctx.mensaje):
            refusal_count = int(conv_data.get("contact_refusal_count", 0))
            if refusal_count == 0:
                return PRIVACY_REPLY + " ¿Me compartes tus datos?"
            else:
                return "Entiendo tu decisión. ¡Que tengas buen día!"
    
    if is_proposal_denial(ctx.mensaje):
        return "Perfecto. Cuando la tengas, cuéntamela."
    
    return None

def layer2_extract_contact_data(ctx: ConversationContext) -> Dict[str, Optional[str]]:
    name_regex = extract_user_name(ctx.mensaje)
    phone_regex = extract_phone(ctx.mensaje)
    barrio_regex = extract_user_barrio(ctx.mensaje)
    
    if name_regex and phone_regex and barrio_regex:
        return {"nombre": name_regex, "telefono": phone_regex, "barrio": barrio_regex}
    
    if ctx.contact_requested:
        if not phone_regex:
            m = re.search(r'\b(\d{10})\b', ctx.mensaje)
            if m:
                phone_regex = m.group(1)
    
    result = {}
    if name_regex: result["nombre"] = name_regex
    if phone_regex: result["telefono"] = phone_regex
    if barrio_regex: result["barrio"] = barrio_regex
    return result

def layer3_classify_with_context(ctx: ConversationContext) -> Dict[str, Any]:
    if ctx.in_proposal_flow:
        return {"tipo": "propuesta", "confianza": "alta"}
    
    if is_proposal_intent_heuristic(ctx.mensaje):
        return {"tipo": "propuesta", "confianza": "alta"}
    
    if "?" in ctx.mensaje:
        return {"tipo": "consulta", "confianza": "alta"}
    
    return {"tipo": "consulta", "confianza": "media"}

# =========================================================
#  RAG
# =========================================================

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
    """Reformula manteniendo tema específico."""
    
    if len(current_query.split()) > 8:
        return current_query
    
    recent = conversation_history[-max_history:] if conversation_history else []
    
    if not recent:
        return current_query
    
    # Extraer tema del último bot
    ultimo_bot = ""
    tema_especifico = ""
    
    for msg in reversed(recent):
        if msg["role"] == "assistant":
            ultimo_bot = msg.get("content", "")
            
            ley_match = re.search(r'Ley\s+(\d+)', ultimo_bot, re.IGNORECASE)
            if ley_match:
                tema_especifico = f"Ley {ley_match.group(1)}"
                break
    
    referencias_vagas = ["ella", "eso", "esa", "sobre eso"]
    if tema_especifico and any(ref in current_query.lower() for ref in referencias_vagas):
        reformulated = f"{tema_especifico} detalles"
        print(f"[QUERY] Reformulada: '{reformulated}'")
        return reformulated
    
    return current_query

def validate_rag_relevance(rag_hits: List[Dict]) -> bool:
    """Valida si hay al menos UN hit con score razonable."""
    if not rag_hits:
        print("[RAG] ⚠️  No hay hits")
        return False
    
    best_score = max((hit.get("score", 0) for hit in rag_hits), default=0)
    
    if best_score >= 0.5:
        print(f"[RAG] ✅ Mejor score: {best_score:.3f}")
        return True
    else:
        print(f"[RAG] ⚠️ Score muy bajo: {best_score:.3f}")
        if len(rag_hits) >= 3:
            print("[RAG] ⚠️ Aceptando por cantidad de hits")
            return True
        return False

# =========================================================
#  GENERACIÓN INTELIGENTE Y PERSONALIZADA
# =========================================================

def layer4_generate_response_with_memory(
    ctx: ConversationContext,
    clasificacion: Dict[str, Any],
    rag_hits: List[Dict]
) -> str:
    """Generación con respeto estricto a RAG pero MÁS NATURAL Y PERSONALIZADA."""
    
    contexto_rag = "\n".join([f"- {h['texto']}" for h in rag_hits if h.get("texto")])
    
    # ✅ System prompt MEJORADO para mayor personalización
    system_msg = (
        "Eres el asistente personal de Wilder Escobar, Representante a la Cámara.\n\n"
        "PERSONALIDAD Y TONO:\n"
        "- Habla como una persona real y cercana, no como un robot\n"
        "- Usa lenguaje natural y conversacional\n"
        "- Muestra empatía y entusiasmo genuino\n"
        "- Adapta tu tono según el contexto de la conversación\n"
        "- Si el usuario parece frustrado, sé más comprensivo\n"
        "- Si está entusiasmado, comparte su energía\n\n"
        "REGLAS CRÍTICAS:\n"
        "1. NO te presentes como Wilder, eres su ASISTENTE\n"
        "2. NO saludes si ya hay historial de conversación\n"
        "3. Usa EXCLUSIVAMENTE información del contexto proporcionado\n"
        "4. Si la información no está en el contexto, di claramente: 'No tengo esa información específica en este momento'\n"
        "5. Máximo 3 frases por respuesta\n"
        "6. Personaliza usando el nombre del usuario cuando lo conozcas\n\n"
    )
    
    # Agregar contexto previo si hay referencia
    if ctx.has_reference and ctx.resumen:
        system_msg += f"CONTEXTO PREVIO: {ctx.resumen}\n"
        system_msg += "Mantén coherencia con ese tema.\n\n"
    
    # Agregar nombre del usuario si lo tenemos
    nombre_usuario = ctx.contact_info.get("nombre")
    if nombre_usuario:
        system_msg += f"NOTA: El usuario se llama {nombre_usuario}. Usa su nombre naturalmente cuando sea apropiado.\n\n"
    
    if clasificacion["tipo"] == "consulta":
        system_msg += (
            "TIPO DE MENSAJE: CONSULTA\n"
            "- Responde de forma clara y directa\n"
            "- Usa SOLO el contexto proporcionado\n"
            "- NO inventes información\n"
            "- Si no tienes la respuesta, sé honesto y sugiere alternativas\n"
        )
    
    msgs = [{"role": "system", "content": system_msg}]
    
    # Incluir historial para continuidad
    if ctx.historial:
        msgs.extend(ctx.historial[-6:])
    
    # Mensaje del usuario con contexto RAG
    msgs.append({
        "role": "user",
        "content": (
            f"CONTEXTO VERIFICADO:\n{contexto_rag}\n\n"
            f"PREGUNTA DEL CIUDADANO:\n{ctx.mensaje}\n\n"
            f"Responde usando SOLO el contexto. Sé natural, cercano y útil."
        )
    })
    
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.4,  # ✅ Mayor creatividad para naturalidad (antes 0.1)
            max_tokens=280,
            presence_penalty=0.3,  # ✅ Evita repeticiones
            frequency_penalty=0.3  # ✅ Fomenta variedad
        )
        
        texto = completion.choices[0].message.content.strip()
        texto = limit_sentences(texto, 3)
        
        # ✅ Eliminar saludos redundantes
        texto = remove_redundant_greetings(texto, ctx.historial)
        
        return texto
        
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "Disculpa, tuve un problema técnico. ¿Puedes reformular tu pregunta?"

# =========================================================
#  HELPERS
# =========================================================

def build_contact_request(missing: List[str]) -> str:
    etiquetas = {"nombre": "tu nombre", "barrio": "tu barrio", "celular": "celular"}
    pedir = [etiquetas[m] for m in missing if m in etiquetas]
    frase = pedir[0] if len(pedir) == 1 else (", ".join(pedir[:-1]) + " y " + pedir[-1])
    return f"¿Me compartes {frase}?"

def build_project_location_request() -> str:
    return "Para ubicar el caso en el mapa: ¿en qué barrio sería exactamente el proyecto?"

def craft_argument_question(name: Optional[str], project_location: Optional[str] = None) -> str:
    saludo = f"{name}, " if name else ""
    return f"{saludo}¿Por qué es importante?"

def positive_ack_and_request_argument(name: Optional[str], project_location: Optional[str] = None) -> str:
    if name:
        return f"Excelente idea, {name}. ¿Por qué sería importante para la comunidad?"
    return "Me parece muy buena idea. ¿Por qué crees que es importante?"

# =========================================================
#  ENDPOINT PRINCIPAL
# =========================================================

@app.post("/responder")
async def responder(data: Entrada):
    try:
        chat_id = data.chat_id or f"web_{os.urandom(4).hex()}"
        usuario_id = upsert_usuario_o_anon(chat_id, data.nombre or data.usuario, data.celular, data.canal)
        conv_ref = ensure_conversacion(chat_id, usuario_id, data.faq_origen, data.canal)
        conv_data = conv_ref.get().to_dict() or {}
        
        ctx = ConversationContext(conv_data, data.mensaje)
        
        # CAPA 1
        layer1_response = layer1_deterministic_response(ctx, conv_data)
        if layer1_response:
            if is_proposal_denial(data.mensaje):
                conv_ref.update({
                    "proposal_requested": False,
                    "proposal_collected": False,
                    "argument_requested": False,
                    "argument_collected": False,
                })
            
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": layer1_response}
            ])
            return {"respuesta": layer1_response, "fuentes": [], "chat_id": chat_id}
        
        # CAPA 2
        extracted_data = layer2_extract_contact_data(ctx)
        
        if extracted_data:
            current_info = ctx.contact_info.copy()
            if extracted_data.get("nombre"):
                current_info["nombre"] = extracted_data["nombre"]
            if extracted_data.get("telefono"):
                current_info["telefono"] = extracted_data["telefono"]
            if extracted_data.get("barrio"):
                current_info["barrio"] = extracted_data["barrio"]
            
            conv_ref.update({"contact_info": current_info})
            
            if current_info.get("telefono"):
                conv_ref.update({"contact_collected": True})
            
            ctx.contact_info = current_info
        
        proj_loc = extract_project_location(data.mensaje)
        if proj_loc:
            conv_ref.update({"project_location": proj_loc})
        
        # CAPA 3
        clasificacion = layer3_classify_with_context(ctx)
        
        # PROPUESTAS
        if clasificacion["tipo"] == "propuesta" or ctx.in_proposal_flow:
            
            # FASE 1: Recopilar propuesta
            if not ctx.proposal_collected:
                
                # Caso A: Intención sin contenido
                if is_proposal_intent_heuristic(data.mensaje) and not looks_like_proposal_content(data.mensaje):
                    conv_ref.update({
                        "proposal_requested": True,
                        "proposal_nudge_count": 0
                    })
                    texto = "¡Perfecto! Cuéntame tu idea. ¿Qué te gustaría que se mejorara y en qué barrio?"
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                # Caso B: Tiene contenido de propuesta
                if looks_like_proposal_content(data.mensaje):
                    propuesta_extraida = extract_proposal_text(data.mensaje)
                    
                    if len(_normalize_text(propuesta_extraida)) < 10:
                        conv_ref.update({
                            "proposal_requested": True,
                            "proposal_nudge_count": 1
                        })
                        texto = "Cuéntame un poco más: ¿qué te gustaría que se hiciera y en qué barrio?"
                        append_mensajes(conv_ref, [
                            {"role": "user", "content": data.mensaje},
                            {"role": "assistant", "content": texto}
                        ])
                        return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                    
                    # Propuesta válida
                    conv_ref.update({
                        "current_proposal": propuesta_extraida,
                        "proposal_collected": True,
                        "proposal_requested": True,
                        "argument_requested": True,
                        "argument_collected": False,
                    })
                    texto = positive_ack_and_request_argument(
                        ctx.contact_info.get("nombre"),
                        ctx.project_location
                    )
                    
                    print(f"[PROPUESTA] ✅ Guardada: '{propuesta_extraida[:50]}...'")
                    
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                # Caso C: Ya pedimos propuesta pero no llegó (NUDGES)
                if conv_data.get("proposal_requested"):
                    nudges = int(conv_data.get("proposal_nudge_count", 0))
                    
                    if is_proposal_denial(data.mensaje):
                        conv_ref.update({
                            "proposal_requested": False,
                            "proposal_collected": False,
                            "proposal_nudge_count": 0,
                            "argument_requested": False,
                            "contact_intent": None
                        })
                        texto = "Perfecto, sin problema. Cuando la tengas, cuéntamela en 1-2 frases."
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
                        texto = "¿Prefieres que responda tu pregunta ahora o seguimos con tu propuesta? Si es propuesta, cuéntamela en 1-2 frases."
                        append_mensajes(conv_ref, [
                            {"role": "user", "content": data.mensaje},
                            {"role": "assistant", "content": texto}
                        ])
                        return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                    
                    # Nudge escalonado
                    nudges += 1
                    conv_ref.update({"proposal_nudge_count": nudges})
                    
                    if nudges == 1:
                        texto = "Claro. ¿Cuál es tu propuesta? Dímela en 1-2 frases y el barrio del proyecto."
                    elif nudges == 2:
                        texto = "Para ayudarte mejor, escribe la propuesta en 1-2 frases (ej: 'Arreglar luminarias del parque de San José')."
                    else:
                        conv_ref.update({
                            "proposal_requested": False,
                            "proposal_nudge_count": 0,
                            "contact_intent": None
                        })
                        texto = "Todo bien. Si prefieres, dime tu pregunta o tema y te ayudo de una."
                    
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
            
            # FASE 2: Recopilar argumento
            if ctx.proposal_collected and not ctx.argument_collected:
                es_argumento = has_argument_text(data.mensaje)
                
                if es_argumento:
                    conv_ref.update({
                        "argument_collected": True,
                        "contact_requested": True,
                    })
                    
                    missing = ctx.get_missing_contact_fields()
                    if not ctx.project_location:
                        missing.append("project_location")
                    
                    if missing:
                        if missing == ["project_location"]:
                            texto = build_project_location_request()
                        else:
                            texto = build_contact_request(missing)
                    else:
                        texto = "Perfecto, con estos datos escalamos tu propuesta."
                    
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
            
            # FASE 3: Recopilar contacto
            if ctx.argument_collected and ctx.contact_requested:
                missing = ctx.get_missing_contact_fields()
                if not ctx.project_location:
                    missing.append("project_location")
                
                if missing:
                    if missing == ["project_location"]:
                        texto = build_project_location_request()
                    else:
                        texto = build_contact_request(missing)
                else:
                    nombre_txt = ctx.contact_info.get("nombre", "")
                    texto = (f"Gracias, {nombre_txt}. " if nombre_txt else "Gracias. ")
                    texto += "Con estos datos escalamos el caso y te contamos avances."
                
                append_mensajes(conv_ref, [
                    {"role": "user", "content": data.mensaje},
                    {"role": "assistant", "content": texto}
                ])
                return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
        
        # CAPA 4: RAG
        query_for_search = data.mensaje
        if ctx.has_reference:
            query_for_search = reformulate_query_with_context(data.mensaje, ctx.historial)
        
        hits = rag_search(query_for_search, top_k=5)

        print(f"\n{'='*60}")
        print(f"[RAG DEBUG] Query: '{query_for_search}'")
        print(f"[RAG DEBUG] Hits encontrados: {len(hits)}")
        for i, hit in enumerate(hits[:3], 1):
            print(f"  Hit {i}: score={hit.get('score', 0):.3f} | texto='{hit.get('texto', '')[:80]}...'")
        print(f"{'='*60}\n")
        
        if not validate_rag_relevance(hits):
            if any(kw in _normalize_text(data.mensaje) for kw in ["ley", "proyecto", "educacion", "educación"]):
                texto = (
                    "No encontré información específica sobre eso en mi base de datos actual. "
                    "Te recomiendo contactar directamente a la oficina de Wilder para información más detallada."
                )
            else:
                texto = "No tengo información específica sobre eso en este momento. ¿Hay algo más en lo que pueda ayudarte?"
        else:
            texto = layer4_generate_response_with_memory(ctx, clasificacion, hits)
        
        append_mensajes(conv_ref, [
            {"role": "user", "content": data.mensaje},
            {"role": "assistant", "content": texto}
        ])
        
        return {"respuesta": texto, "fuentes": hits, "chat_id": chat_id}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# =========================================================
#  CLASIFICACIÓN
# =========================================================

@app.post("/clasificar")
async def clasificar(body: ClasificarIn):
    """Clasifica conversación y guarda categorías."""
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
            return {"ok": False, "mensaje": "No hay mensajes para clasificar"}
        
        propuesta = conv_data.get("current_proposal") or ""
        
        if propuesta:
            tipo = "propuesta"
            texto_clasificar = propuesta
        else:
            es_consulta = (
                ("?" in ultimo_usuario) or
                any(kw in _normalize_text(ultimo_usuario) 
                    for kw in ["que", "como", "cuando", "donde", "wilder", "ley", "proyecto"])
            )
            
            if es_consulta:
                tipo = "consulta"
                texto_clasificar = ultimo_usuario
            else:
                return {"ok": True, "skipped": True, "reason": "ni_consulta_ni_propuesta"}
        
        print(f"[CLASIFICAR] Tipo: {tipo}")
        
        if tipo == "consulta":
            sys = (
                "Clasifica esta CONSULTA ciudadana.\n"
                "Devuelve SOLO JSON:\n"
                '{"categoria_general": "Consulta", "titulo_propuesta": "[tema en 5-8 palabras]", "tono_detectado": "neutral"}\n'
                "Ejemplo: Ley 2420 educación pospandemia"
            )
            usr = f"Consulta: {texto_clasificar}"
        else:
            ubicacion = conv_data.get("project_location") or ""
            sys = (
                "Clasifica esta PROPUESTA ciudadana.\n"
                "Devuelve SOLO JSON:\n"
                '{"categoria_general": "[Infraestructura|Seguridad|Movilidad|Educación|Salud]", "titulo_propuesta": "[Verbo + Qué + Dónde]", "tono_detectado": "propositivo"}\n'
                "Ejemplo: Mejorar alumbrado en Laureles"
            )
            usr = f"Propuesta: {texto_clasificar}\nUbicación: {ubicacion}" if ubicacion else f"Propuesta: {texto_clasificar}"
        
        try:
            out = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                temperature=0.2,
                max_tokens=100
            ).choices[0].message.content.strip()
            
            out = out.replace("```json", "").replace("```", "").strip()
            data = json.loads(out)
        except:
            if tipo == "consulta":
                data = {"categoria_general": "Consulta", "titulo_propuesta": "Consulta ciudadana", "tono_detectado": "neutral"}
            else:
                data = {"categoria_general": "General", "titulo_propuesta": "Propuesta ciudadana", "tono_detectado": "propositivo"}
        
        categoria = data.get("categoria_general", "General")
        titulo = data.get("titulo_propuesta", "Sin título")
        tono = data.get("tono_detectado", "neutral")
        
        categorias_existentes = conv_data.get("categoria_general") or []
        titulos_existentes = conv_data.get("titulo_propuesta") or []
        
        if isinstance(categorias_existentes, str):
            categorias_existentes = [categorias_existentes]
        if isinstance(titulos_existentes, str):
            titulos_existentes = [titulos_existentes]
        
        if categoria not in categorias_existentes:
            categorias_existentes.append(categoria)
        
        titulo_norm = _normalize_text(titulo)
        titulos_norm_exist = [_normalize_text(t) for t in titulos_existentes]
        if titulo_norm not in titulos_norm_exist:
            titulos_existentes.append(titulo)
        
        conv_ref.update({
            "categoria_general": categorias_existentes,
            "titulo_propuesta": titulos_existentes,
            "tono_detectado": tono,
            "ultima_fecha": firestore.SERVER_TIMESTAMP
        })
        
        print(f"[CLASIFICAR] ✅ {categorias_existentes} / {titulos_existentes}")
        
        return {
            "ok": True,
            "clasificacion": {
                "tipo": tipo,
                "categoria_general": categoria,
                "titulo_propuesta": titulo,
                "tono_detectado": tono,
                "todas_categorias": categorias_existentes,
                "todos_titulos": titulos_existentes
            }
        }
        
    except Exception as e:
        print(f"[CLASIFICAR] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)