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

@app.get("/")
async def root():
    """Ruta raíz - información básica de la API"""
    return {
        "service": "WilderBot API",
        "version": "2.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "responder": "POST /responder",
            "clasificar": "POST /clasificar",
            "docs": "/docs"
        }
    }

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
            "project_city": None,
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
    
    # Resumen completo (solo si hay suficientes mensajes)
    if len(arr) >= 4:
        try:
            resumen_completo = build_detailed_summary(data, arr)
            conv_ref.update({"resumen_completo": resumen_completo})
        except:
            pass
    
    # Resumen en caliente
    try:
        update_conversation_summary(conv_ref)
    except:
        pass

# ╔═══════════════════════════════════════════════════════════════════════
# ║ RESUMEN COMPLETO ESTRUCTURADO
# ╚═══════════════════════════════════════════════════════════════════════

def build_detailed_summary(conv_data: dict, mensajes: List[Dict[str, str]]) -> dict:
    """
    Genera un resumen estructurado completo para el botón "Ver más".
    
    Devuelve un objeto con:
    - tema_principal: str
    - consultas: List[str]  (si hubo consultas)
    - propuesta: str (si hay)
    - argumento: str (si hay)
    - ubicacion: str (si hay)
    - contacto: dict (datos recopilados)
    - estado: str (fase actual)
    - historial_resumido: List[dict] (Últimos 5 intercambios)
    """
    
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
    
    # ────────────────────────────────────────────────────────────────
    # 1. TEMA PRINCIPAL
    # ────────────────────────────────────────────────────────────────
    categorias = conv_data.get("categoria_general", [])
    titulos = conv_data.get("titulo_propuesta", [])
    
    if titulos:
        summary["tema_principal"] = titulos[-1] if isinstance(titulos, list) else titulos
    elif categorias:
        summary["tema_principal"] = categorias[-1] if isinstance(categorias, list) else categorias
    else:
        summary["tema_principal"] = "Conversación sin clasificar"
    
    # ────────────────────────────────────────────────────────────────
    # 2. CONSULTAS (si la categoría incluye "Consulta")
    # ────────────────────────────────────────────────────────────────
    if isinstance(categorias, list) and "Consulta" in categorias:
        for titulo in (titulos if isinstance(titulos, list) else [titulos]):
            if titulo and any(kw in _normalize_text(titulo) for kw in ["ley", "proyecto", "apoyo", "posicion", "posición"]):
                summary["consultas"].append(titulo)
    
    # ────────────────────────────────────────────────────────────────
    # 3. PROPUESTA
    # ────────────────────────────────────────────────────────────────
    propuesta = conv_data.get("current_proposal")
    if propuesta:
        summary["propuesta"] = propuesta[:120] + ("..." if len(propuesta) > 120 else "")
    
    # ────────────────────────────────────────────────────────────────
    # 4. ARGUMENTO (buscar en los mensajes del usuario)
    # ────────────────────────────────────────────────────────────────
    if conv_data.get("argument_collected"):
        for m in mensajes:
            if m.get("role") == "user":
                content = m.get("content", "")
                if has_argument_text(content) and len(content) > 30:
                    summary["argumento"] = content[:120] + ("..." if len(content) > 120 else "")
                    break
    
    # ────────────────────────────────────────────────────────────────
    # 5. UBICACIÓN
    # ────────────────────────────────────────────────────────────────
    project_location = conv_data.get("project_location")
    if project_location:
        if project_location == "Ciudad" and conv_data.get("project_city"):
            summary["ubicacion"] = f"{project_location}: {conv_data['project_city']}"
        else:
            summary["ubicacion"] = project_location
    
    # ────────────────────────────────────────────────────────────────
    # 6. CONTACTO
    # ────────────────────────────────────────────────────────────────
    contact_info = conv_data.get("contact_info", {})
    if contact_info:
        summary["contacto"] = {
            "nombre": contact_info.get("nombre"),
            "barrio": contact_info.get("barrio"),
            "telefono": contact_info.get("telefono")
        }
    
    # ────────────────────────────────────────────────────────────────
    # 7. ESTADO (fase actual del flujo)
    # ────────────────────────────────────────────────────────────────
    if conv_data.get("contact_collected"):
        summary["estado"] = "completado"
    elif conv_data.get("contact_requested"):
        summary["estado"] = "esperando_contacto"
    elif conv_data.get("argument_collected"):
        summary["estado"] = "argumento_recibido"
    elif conv_data.get("proposal_collected"):
        summary["estado"] = "propuesta_recibida"
    elif isinstance(categorias, list) and "Consulta" in categorias:
        summary["estado"] = "consulta_respondida"
    else:
        summary["estado"] = "iniciado"
    
    # ────────────────────────────────────────────────────────────────
    # 8. HISTORIAL RESUMIDO (Últimos 5 intercambios = 10 mensajes)
    # ────────────────────────────────────────────────────────────────
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
    
    # Rechazos explícitos
    rechazos_explicitos = [
        "no me gusta dar mis datos",
        "no quiero compartir mis datos",
        "no doy mi celular",
        "no quiero dar",
        "no me gusta dar",
        "no dare",
        "no daré",
        "prefiero no dar",
        "no me gusta dar esa informacion",
        "no me gusta dar esa información"
    ]
    
    # Rechazos simples (solo si el mensaje es corto)
    if len(t) <= 15:
        rechazos_simples = ["no quiero", "no puedo", "no", "prefiero no"]
        if any(r == t for r in rechazos_simples):
            return True
    
    return any(p in t for p in rechazos_explicitos)

def is_asking_why(text: str) -> bool:
    """
    Detecta si el usuario pregunta "¿para qué?" o similares cuando se le piden datos.
    """
    t = _normalize_text(text)
    
    # Preguntas directas sobre por qué se piden datos
    preguntas_why = [
        "para que",
        "para qué",
        "por que",
        "por qué",
        "por que razon",
        "por qué razón",
        "para que es",
        "para qué es",
        "por que lo necesitas",
        "por qué lo necesitas",
        "para que lo necesitas",
        "para qué lo necesitas",
        "por que me lo pides",
        "para que sirve"
    ]
    
    # Si el mensaje es corto (menos de 20 chars) y tiene pregunta "para qué"
    if len(t) <= 20 and any(p in t for p in preguntas_why):
        return True
    
    return False

def detect_tone_from_argument(text: str) -> str:
    """
    Detecta el tono del argumento del usuario.
    Opciones: propositivo, urgente, preocupado, neutral
    """
    t = _normalize_text(text)
    
    # Tono URGENTE
    if any(kw in t for kw in ["urgente", "rapido", "ya", "inmediato", "critico", "grave"]):
        return "urgente"
    
    # Tono PREOCUPADO
    if any(kw in t for kw in ["miedo", "inseguridad", "peligro", "accidente", "riesgo", "preocupa"]):
        return "preocupado"
    
    # Tono PROPOSITIVO (default para propuestas)
    if any(kw in t for kw in ["mejor", "beneficio", "oportun", "positiv", "important"]):
        return "propositivo"
    
    # Neutral (fallback)
    return "neutral"

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
    """
    Extrae barrio del proyecto con detección FLEXIBLE.
    Acepta múltiples formatos:
      - "en el barrio Aranjuez"
      - "proyecto es en Aranjuez"
      - "Aranjuez" (cuando ya se pidió el dato)
    """
    
    # Patrón 1: "en el barrio X"
    m = re.search(
        r'\ben\s+el\s+barrio\s+([A-Za-záéíóúñ0-9 \-]{2,50})',
        text, 
        flags=re.IGNORECASE
    )
    if m:
        return _clean_barrio_fragment(m.group(1))
    
    # Patrón 2: "proyecto (es/sería) en X"
    m = re.search(
        r'\bproyecto\s+(?:es|seria|sería)\s+en\s+([A-Za-záéíóúñ0-9 \-]{2,50})',
        text,
        flags=re.IGNORECASE
    )
    if m:
        return _clean_barrio_fragment(m.group(1))
    
    # Patrón 3: "el proyecto en X"
    m = re.search(
        r'\bel\s+proyecto\s+en\s+([A-Za-záéíóúñ0-9 \-]{2,50})',
        text,
        flags=re.IGNORECASE
    )
    if m:
        return _clean_barrio_fragment(m.group(1))
    
    # Patrón 4: "se haría en X" o "se hará en X"
    m = re.search(
        r'\bse\s+(?:haria|haría|hara|hará)\s+en\s+([A-Za-záéíóúñ0-9 \-]{2,50})',
        text,
        flags=re.IGNORECASE
    )
    if m:
        return _clean_barrio_fragment(m.group(1))
    
    # Patrón 5: Buscar cualquier nombre con mayúscula que NO sea nombre de persona
    # Solo si el mensaje es corto (< 50 chars) y tiene pocas palabras
    if len(text) < 50 and len(text.split()) <= 5:
        palabras_mayuscula = re.findall(
            r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,2})\b',
            text
        )
        
        for posible_barrio in palabras_mayuscula:
            normalized = _normalize_text(posible_barrio)
            
            # Validar que NO sea palabra de discurso ni nombre común
            if normalized in DISCOURSE_START_WORDS:
                continue
            
            # Validar que NO sea un nombre de persona (sin apellido)
            # Si tiene 1 sola palabra y está sola, probablemente es barrio
            palabras = posible_barrio.split()
            if len(palabras) <= 3:
                return _titlecase(posible_barrio)
    
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
    

def llm_extract_personal_contact(text: str) -> Dict[str, Optional[str]]:
    """
    Usa LLM para extraer DATOS PERSONALES: nombre, barrio de residencia, teléfono.
    Versión mejorada con prompt específico.
    """
    sys = (
        "Extrae los DATOS PERSONALES del usuario del siguiente mensaje.\n"
        "El bot le pidió: 'tu nombre, el barrio donde vives y tu número de celular'\n\n"
        "Devuelve un JSON con:\n"
        "{\n"
        '  "nombre": "nombre completo del usuario (ej: Juan Pérez)",\n'
        '  "barrio": "barrio donde VIVE/RESIDE el usuario (ej: Laureles)",\n'
        '  "telefono": "número de celular sin espacios (ej: 3001234567)"\n'
        "}\n\n"
        "REGLAS CRÍTICAS:\n"
        "- El barrio es donde VIVE el usuario (su RESIDENCIA)\n"
        "- Si dice 'Cesar, 312658945, Centro' → Centro es su barrio de RESIDENCIA\n"
        "- Si dice 'Juan, Laureles, 3001234567' → Laureles es su barrio de RESIDENCIA\n"
        "- Si no encuentras algún dato, pon null\n"
        "- El teléfono debe ser solo números (quitar espacios y guiones)\n"
        "- El nombre debe estar en formato Title Case\n"
        "\n"
        "EJEMPLOS:\n"
        '- "Cesar, 312658945, Centro" → {"nombre": "Cesar", "barrio": "Centro", "telefono": "312658945"}\n'
        '- "María López, vivo en Laureles, 3001234567" → {"nombre": "María López", "barrio": "Laureles", "telefono": "3001234567"}\n'
        '- "Pedro, 3156789012, mi barrio es Aranjuez" → {"nombre": "Pedro", "barrio": "Aranjuez", "telefono": "3156789012"}\n'
    )
    
    usr = f"Mensaje del usuario:\n{text}"
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr}
            ],
            temperature=0.0,
            max_tokens=150,
            timeout=5
        )
        
        result = response.choices[0].message.content.strip()
        
        # Limpiar posibles backticks
        result = result.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(result)
        
        # Validar y limpiar teléfono
        if data.get("telefono"):
            tel = re.sub(r'\D', '', data["telefono"])
            tel = re.sub(r'^(?:00)?57', '', tel)
            data["telefono"] = tel if 8 <= len(tel) <= 12 else None
        

        return data
        
    except Exception as e:

        return {"nombre": None, "barrio": None, "telefono": None}


def llm_extract_project_location(text: str) -> Dict[str, Optional[str]]:
    """
    Usa LLM para extraer la UBICACIÓN del proyecto.
    Puede ser un barrio específico o "Ciudad" si es para toda la ciudad.
    """
    sys = (
        "Extrae la UBICACIÓN del proyecto del siguiente mensaje.\n"
        "El usuario ya dio sus datos personales. Ahora está diciendo DÓNDE sería el proyecto.\n"
        "Devuelve un JSON con:\n"
        "{\n"
        '  "barrio_proyecto": "ubicación del proyecto"\n'
        "}\n\n"
        "REGLAS CRÍTICAS:\n"
        "1. Si dice que es para TODA LA CIUDAD, la ciudad completa, toda Medellín, toda Manizales, etc:\n"
        '   → Devuelve: {"barrio_proyecto": "Ciudad"}\n'
        "\n"
        "2. Si menciona un BARRIO ESPECÍFICO:\n"
        '   → Devuelve el nombre del barrio en Title Case (ej: {"barrio_proyecto": "Aranjuez"})\n'
        "\n"
        "3. Si menciona una COMUNA:\n"
        '   → Devuelve "Comuna X" (ej: {"barrio_proyecto": "Comuna 1"})\n'
        "\n"
        "4. Si NO menciona ubicación o es ambiguo:\n"
        '   → Devuelve: {"barrio_proyecto": null}\n'
        "\n"
        "EJEMPLOS:\n"
        '- "En toda la ciudad" → {"barrio_proyecto": "Ciudad"}\n'
        '- "Para toda Medellín" → {"barrio_proyecto": "Ciudad"}\n'
        '- "A nivel ciudad" → {"barrio_proyecto": "Ciudad"}\n'
        '- "En el barrio Laureles" → {"barrio_proyecto": "Laureles"}\n'
        '- "Aranjuez" → {"barrio_proyecto": "Aranjuez"}\n'
        '- "En la comuna 1" → {"barrio_proyecto": "Comuna 1"}\n'
    )
    
    usr = f"Mensaje del usuario:\n{text}"
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr}
            ],
            temperature=0.0,
            max_tokens=100,
            timeout=5
        )
        
        result = response.choices[0].message.content.strip()
        result = result.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(result)
        
        # Validar y normalizar
        ubicacion = data.get("barrio_proyecto")
        
        if ubicacion:
            # Si dice "ciudad", normalizarlo a "Ciudad"
            if ubicacion.lower() in ["ciudad", "toda la ciudad", "la ciudad"]:
                data["barrio_proyecto"] = "Ciudad"
            else:
                # Capitalizar correctamente
                data["barrio_proyecto"] = _titlecase(ubicacion)
        
        print(f"[LLM EXTRACT] Ubicación proyecto: {data}")
        return data
        
    except Exception as e:
        print(f"[LLM EXTRACT] Error: {e}")
        return {"barrio_proyecto": None}


def llm_extract_city_name(text: str) -> Dict[str, Optional[str]]:
    """
    Usa LLM para extraer el NOMBRE DE LA CIUDAD.
    Se usa cuando el usuario ya dijo "toda la ciudad" y ahora especifica cuál.
    """
    sys = (
        "Extrae el NOMBRE DE LA CIUDAD del siguiente mensaje.\n"
        "El usuario ya dijo que el proyecto es para 'toda la ciudad'. "
        "Ahora está especificando cuál ciudad.\n"
        "Devuelve un JSON con:\n"
        "{\n"
        '  "ciudad": "nombre de la ciudad en Title Case (ej: Medellín, Manizales, Bogotá)"\n'
        "}\n\n"
        "REGLAS:\n"
        "- Solo extrae el nombre de la ciudad\n"
        "- Debe estar en formato Title Case\n"
        "- Si no encuentras el nombre, pon null\n"
        "\n"
        "EJEMPLOS:\n"
        '- "Medellín" → {"ciudad": "Medellín"}\n'
        '- "En Manizales" → {"ciudad": "Manizales"}\n'
        '- "Es para Bogotá" → {"ciudad": "Bogotá"}\n'
        '- "Cali" → {"ciudad": "Cali"}\n'
    )
    
    usr = f"Mensaje del usuario:\n{text}"
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr}
            ],
            temperature=0.0,
            max_tokens=50,
            timeout=5
        )
        
        result = response.choices[0].message.content.strip()
        result = result.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(result)
        
        # Validar y capitalizar
        if data.get("ciudad"):
            data["ciudad"] = _titlecase(data["ciudad"])
        
        print(f"[LLM EXTRACT] Ciudad: {data}")
        return data
        
    except Exception as e:
        print(f"[LLM EXTRACT] Error: {e}")
        return {"ciudad": None}
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
    
    # ══════════════════════════════════════════════════════════════
    # CONSULTAS OBVIAS (prioridad alta)
    # ══════════════════════════════════════════════════════════════

    # Tiene signo de interrogación?
    if "?" in mensaje:
        # Es consulta si NO tiene palabras de propuesta
        palabras_propuesta = ["propongo", "sugiero", "me gustaria construir", "quisiera que", "deberian"]
        es_propuesta = any(kw in t for kw in palabras_propuesta)
        
        if not es_propuesta:
            print("[CLASSIFY] ✅ Es consulta (tiene '?' y no palabras de propuesta)")
            return {"tipo": "consulta", "confianza": "alta", "metodo": "heuristica_consulta"}

    # Pregunta sobre Wilder o leyes (sin signo de interrogación)
    palabras_consulta = [
        "wilder", "ley", "leyes", "proyecto de ley", "proyectos de ley",
        "que propone", "que apoya", "cual es su posicion", "apoya alguna",
        "ha presentado", "que piensa", "que opina"
    ]

    if any(kw in t for kw in palabras_consulta):
        # Solo si NO tiene verbos de acción física
        verbos_accion = ["arregl", "constru", "instal", "paviment", "ilumin", "repar", "pint"]
        tiene_verbo_accion = any(v in t for v in verbos_accion)
        
        if not tiene_verbo_accion:
            print("[CLASSIFY] ✅ Es consulta (pregunta sobre Wilder/leyes sin verbos de acción)")
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
    Versión mejorada con mejor detección de leyes y referencias plurales.
    """
    
    # Si la query ya es específica y larga, no reformular
    if len(current_query.split()) > 8:
        return current_query
    
    recent = conversation_history[-max_history:] if conversation_history else []
    
    if not recent:
        return current_query
    
    # ──────────────────────────────────────────────────────────
    # PASO 1: Extraer TODAS las leyes mencionadas
    # ──────────────────────────────────────────────────────────
    leyes_mencionadas = []
    temas_mencionados = []
    
    for msg in recent:
        if msg["role"] == "assistant":
            content = msg.get("content", "")
            
            # Patrón 1: "Ley XXXX de YYYY" (ej: Ley 2402 de 2024)
            leyes_patron1 = re.findall(r'Ley\s+(\d+)\s+de\s+(\d{4})', content, re.IGNORECASE)
            for num_ley, año in leyes_patron1:
                ley_completa = f"Ley {num_ley} de {año}"
                if ley_completa not in leyes_mencionadas:
                    leyes_mencionadas.append(ley_completa)
            
            # Patrón 2: "la Ley XXXX" (sin año)
            leyes_patron2 = re.findall(r'la\s+Ley\s+(\d+)', content, re.IGNORECASE)
            for num_ley in leyes_patron2:
                ley_simple = f"Ley {num_ley}"
                # Solo agregar si no hay una versión completa ya
                if not any(f"Ley {num_ley}" in l for l in leyes_mencionadas):
                    leyes_mencionadas.append(ley_simple)
            
            # Patrón 3: Extraer temas después de "sobre", "relacionadas con", "de"
            temas_patron = re.findall(
                r'(?:sobre|relacionadas?\s+con|de|para)\s+(?:el\s+|la\s+)?([a-záéíóúñ\s]{8,60}?)(?:\.|,|y\s+la|$)',
                content.lower()
            )
            
            for tema in temas_patron:
                tema_limpio = tema.strip()
                if len(tema_limpio) >= 8 and tema_limpio not in temas_mencionados:
                    temas_mencionados.append(tema_limpio)
    
    
    # ──────────────────────────────────────────────────────────
    # PASO 2: Detectar tipo de referencia en la query actual
    # ──────────────────────────────────────────────────────────
    query_lower = current_query.lower()
    
    # Referencias PLURALES en español
    referencias_plurales = [
        "las dos", "ambas", "esas", "esos", "las tres", "todos", 
        "ellas", "sobre ellas", "de ellas", "las leyes",
        "esas leyes", "ambas leyes", "las mencionadas"
    ]
    
    # Referencias SINGULARES
    referencias_singulares = [
        "eso", "esa", "ese", "sobre eso", "de eso", "aquello",
        "la primera", "el primero", "esa ley", "ese proyecto"
    ]
    
    # ──────────────────────────────────────────────────────────
    # CASO 1: Referencias PLURALES → Tomar TODAS las leyes/temas
    # ──────────────────────────────────────────────────────────
    es_plural = any(ref in query_lower for ref in referencias_plurales)
    
    if es_plural:
        if len(leyes_mencionadas) >= 2:
            # Tomar TODAS las leyes mencionadas (no solo las últimas)
            todas_leyes = " y ".join(leyes_mencionadas)
            reformulated = f"{todas_leyes} detalles completos"
            print(f"[QUERY] Reformulada (plural - TODAS): '{reformulated}'")
            return reformulated
        
        elif len(leyes_mencionadas) == 1 and len(temas_mencionados) >= 1:
            # Una ley + temas
            reformulated = f"{leyes_mencionadas[0]} {temas_mencionados[0]} detalles"
            print(f"[QUERY] Reformulada (plural - ley+tema): '{reformulated}'")
            return reformulated
        
        elif len(temas_mencionados) >= 2:
            # Solo temas
            todos_temas = " y ".join(temas_mencionados[:2])
            reformulated = f"{todos_temas} información completa"
            print(f"[QUERY] Reformulada (plural - temas): '{reformulated}'")
            return reformulated
    
    # ──────────────────────────────────────────────────────────
    # CASO 2: Referencias SINGULARES → Tomar el ÚLTIMO
    # ──────────────────────────────────────────────────────────
    es_singular = any(ref in query_lower for ref in referencias_singulares)
    
    if es_singular:
        if leyes_mencionadas:
            # Detectar si pide "la primera" o "el primero"
            if "primera" in query_lower or "primero" in query_lower:
                ultimo_tema = leyes_mencionadas[0]  # La primera mencionada
            else:
                ultimo_tema = leyes_mencionadas[-1]  # La última mencionada
            
            reformulated = f"{ultimo_tema} detalles"
            print(f"[QUERY] Reformulada (singular): '{reformulated}'")
            return reformulated
        
        elif temas_mencionados:
            ultimo_tema = temas_mencionados[-1]
            reformulated = f"{ultimo_tema} información"
            print(f"[QUERY] Reformulada (singular - tema): '{reformulated}'")
            return reformulated
    
    # ──────────────────────────────────────────────────────────
    # CASO 3: Palabras como "más", "ampliación", "detalles"
    # ──────────────────────────────────────────────────────────
    palabras_expansion = ["más", "mas", "ampliar", "detalles", "información", "informacion"]
    
    if any(p in query_lower for p in palabras_expansion):
        if leyes_mencionadas:
            if len(leyes_mencionadas) >= 2:
                # Si mencionó varias, asumir que quiere todas
                todas = " y ".join(leyes_mencionadas)
                reformulated = f"{todas} información completa"
                print(f"[QUERY] Reformulada (expansión - todas): '{reformulated}'")
                return reformulated
            else:
                # Solo una ley
                reformulated = f"{leyes_mencionadas[-1]} detalles"
                print(f"[QUERY] Reformulada (expansión - una): '{reformulated}'")
                return reformulated
    
    # ──────────────────────────────────────────────────────────
    # CASO 4: No hay referencias claras → Query original
    # ──────────────────────────────────────────────────────────
    return current_query


# ═══════════════════════════════════════════════════════════════════════
# SECCIÓN 8: HELPERS DE CONTACTO
# ═══════════════════════════════════════════════════════════════════════

def build_contact_request(missing: List[str]) -> str:
    """
    Genera mensaje pidiendo datos (ahora con explicación).
    Mantiene compatibilidad con llamadas antiguas.
    """
    # Delegar a la función mejorada (sin nombre porque es legacy)
    return build_contact_request_with_explanation(None, missing)

def build_project_location_request() -> str:
    """Pide ubicación del proyecto."""
    return (
        "Para ubicar el caso en el mapa: ¿en qué barrio sería el proyecto? "
    )

def explain_why_project_location() -> str:
    """Explica por qué necesitamos la ubicación del proyecto."""
    return (
        "Es para ubicar tu propuesta en el mapa y asignarla al área correspondiente. "
        "Así el equipo de Wilder puede priorizar y gestionar mejor cada caso según la zona. "
        "¿En qué barrio sería el proyecto?"
    )

def craft_argument_question(name: Optional[str], project_location: Optional[str] = None) -> str:
    """Pregunta por argumentos."""
    saludo = f"{name}, " if name else ""
    return f"{saludo}¿Por qué es importante?"

def positive_ack_and_request_argument(name: Optional[str], project_location: Optional[str] = None) -> str:
    """Reconoce propuesta y pide argumento."""
    return "Excelente idea. ¿Por qué sería importante?"

def positive_feedback_after_argument(
    name: Optional[str], 
    missing_fields: List[str],
    propuesta: str,
    argumento: str
) -> str:
    """
    Genera feedback PERSONALIZADO con LLM sobre la propuesta y el argumento.
    Versión robusta con múltiples niveles de fallback.
    """
    
    # ══════════════════════════════════════════════════════════════
    # PASO 1: Validar inputs
    # ══════════════════════════════════════════════════════════════
    if not propuesta or not argumento:
        print("[WARN] Propuesta o argumento vacíos, usando fallback genérico")
        if name:
            feedback = f"Gracias, {name}. Es una propuesta valiosa para la comunidad. "
        else:
            feedback = "Gracias por compartir tu propuesta. Es muy valiosa para la comunidad. "
    else:
        # ══════════════════════════════════════════════════════════════
        # PASO 2: Intentar generar feedback personalizado con LLM
        # ══════════════════════════════════════════════════════════════
        
        sys = (
            "Eres un asistente que valida propuestas ciudadanas.\n"
            "Da una opinión positiva BREVE (máximo 2 frases cortas) sobre la propuesta.\n"
            "Debe sonar natural, empático y motivador.\n"
            "NO uses frases genéricas como 'tiene mucho sentido'.\n"
            "Enfócate en el IMPACTO que menciona el ciudadano.\n\n"
            "Ejemplos:\n"
            "- 'Me parece excelente. Mejorar la seguridad con iluminación es clave para las familias.'\n"
            "- 'Muy buena idea. Un espacio recreativo fortalece la comunidad.'\n"
            "- 'Totalmente de acuerdo. Mejorar las vías evitará accidentes.'\n"
        )
        
        # Limitar tamaño para evitar errores
        propuesta_corta = propuesta[:200]
        argumento_corto = argumento[:200]
        
        usr = (
            f"Propuesta: {propuesta_corta}\n"
            f"Motivo: {argumento_corto}\n\n"
            "Da tu opinión positiva BREVE (máximo 2 frases)."
        )
        
        try:
            print("[LLM] Generando feedback personalizado...")
            
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr}
                ],
                temperature=0.7,
                max_tokens=80,
                timeout=8  # Aumentado a 8 segundos
            )
            
            feedback_llm = response.choices[0].message.content.strip()
            feedback_llm = feedback_llm.strip('"').strip("'").strip()
            
            # Validar que el LLM dio una respuesta útil
            if len(feedback_llm) < 10:
                raise ValueError("Respuesta LLM muy corta")
            
            saludo = f"Gracias, {name}. " if name else "Gracias por compartir eso. "
            feedback = saludo + feedback_llm + " "
            
            print(f"[LLM] ✅ Feedback generado: {feedback[:100]}...")
            
        except Exception as e:
            print(f"[WARN] LLM feedback failed: {e}")
            
            # ══════════════════════════════════════════════════════════════
            # FALLBACK 1: Feedback basado en palabras clave
            # ══════════════════════════════════════════════════════════════
            argumento_lower = argumento.lower()
            propuesta_lower = propuesta.lower()
            
            if "niño" in argumento_lower or "niña" in argumento_lower or "jugar" in argumento_lower:
                feedback_especifico = "Es una excelente idea. Un espacio para los niños fortalece la comunidad y les brinda seguridad."
            elif "seguridad" in argumento_lower or "inseguridad" in argumento_lower or "miedo" in argumento_lower:
                feedback_especifico = "Me parece muy importante. Mejorar la seguridad es fundamental para la tranquilidad de las familias."
            elif "via" in propuesta_lower or "calle" in propuesta_lower or "anden" in propuesta_lower:
                feedback_especifico = "Totalmente de acuerdo. Mejorar la infraestructura vial beneficia a toda la comunidad."
            elif "alumbrado" in propuesta_lower or "luminaria" in propuesta_lower or "luz" in propuesta_lower:
                feedback_especifico = "Excelente propuesta. Un buen alumbrado público mejora la seguridad y calidad de vida."
            elif "parque" in propuesta_lower or "cancha" in propuesta_lower:
                feedback_especifico = "Me parece muy valioso. Los espacios recreativos son esenciales para la comunidad."
            else:
                feedback_especifico = "Es una propuesta muy valiosa que puede mejorar la calidad de vida en el sector."
            
            if name:
                feedback = f"Gracias, {name}. {feedback_especifico} "
            else:
                feedback = f"Gracias por compartir eso. {feedback_especifico} "
            
            print(f"[FALLBACK] ✅ Usando feedback basado en keywords")
    
    # ══════════════════════════════════════════════════════════════
    # PASO 3: Agregar explicación de por qué necesita datos
    # ══════════════════════════════════════════════════════════════
    
    # Si NO hay datos faltantes
    if not missing_fields:
        return feedback + "Ya tengo todos tus datos, así que vamos a escalar esto. ¡Gracias!"
    
    explanation = "Para darle seguimiento y escalar tu propuesta con el equipo de Wilder, necesito "

    # Construir lista de datos faltantes
    etiquetas = {
        "nombre": "tu nombre",
        "barrio": "el barrio donde vives",  # ✅ Este debe estar SIEMPRE
        "celular": "tu número de celular"
    }
    
    pedir = [etiquetas[m] for m in missing_fields if m in etiquetas]
    
    if not pedir:
        return feedback + "Ya tengo todos tus datos, así que vamos a escalar esto. ¡Gracias!"
    
    # Construir frase natural
    if len(pedir) == 1:
        frase = pedir[0]
    elif len(pedir) == 2:
        frase = pedir[0] + " y " + pedir[1]
    else:
        frase = ", ".join(pedir[:-1]) + " y " + pedir[-1]
    
    return feedback + explanation + frase + ". ¿Me los compartes?"


def build_contact_request_with_explanation(name: Optional[str], missing: List[str]) -> str:
    """
    Genera mensaje pidiendo datos CON EXPLICACIÓN de por qué se necesitan.
    """
    # Saludo personalizado
    if name:
        intro = f"{name}, para darle seguimiento y escalar tu propuesta con el equipo de Wilder, "
    else:
        intro = "Para darle seguimiento y escalar tu propuesta con el equipo de Wilder, "
    
    # Si NO hay datos faltantes
    if not missing:
        return "Ya tengo todos los datos necesarios. ¡Gracias!"
    
    # Explicación corta
    explanation = "necesito "
    
    # Construir lista de datos faltantes
    etiquetas = {
        "nombre": "tu nombre",
        "barrio": "el barrio donde vives",
        "celular": "tu número de celular",
        "project_location": "el barrio exacto del proyecto"
    }
    
    pedir = [etiquetas[m] for m in missing if m in etiquetas]
    
    if not pedir:
        return "Ya tengo todos los datos necesarios. ¡Gracias!"
    
    # Construir frase natural
    if len(pedir) == 1:
        frase = pedir[0]
    elif len(pedir) == 2:
        frase = pedir[0] + " y " + pedir[1]
    else:
        frase = ", ".join(pedir[:-1]) + " y " + pedir[-1]
    
    return intro + explanation + frase + ". ¿Me los compartes?"

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
        "Eres el ASISTENTE de Wilder Escobar, Representante a la Cámara en Colombia.\n"
        "IMPORTANTE: NO eres Wilder, eres su asistente que lo representa.\n\n"
        "REGLAS:\n"
        "- Habla en TERCERA PERSONA sobre Wilder: 'Wilder ha apoyado...', 'El representante propone...'\n"
        "- NUNCA digas 'yo' o 'he apoyado' o 'mi trabajo'\n"
        "- Tono cercano, profesional y claro\n"
        "- Máximo 3-4 frases, sin párrafos largos\n"
        "- No saludes de nuevo si ya hay historial\n"
        "- No pidas contacto si no corresponde\n"
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
        # CAPA 1: Extracción de Datos CON LLM
        # ═══════════════════════════════════════════════════════════════

        name = None
        phone = None
        user_barrio = None
        proj_loc = None
        project_city = None

        # ══════════════════════════════════════════════════════════════
        # ESTRATEGIA: Usar LLM SIEMPRE que se estén pidiendo datos
        # ══════════════════════════════════════════════════════════════

        # FASE 3C: ¿Está pidiendo el nombre de la ciudad?
        if (conv_data.get("contact_collected") and 
            conv_data.get("project_location") == "Ciudad" and 
            not conv_data.get("project_city")):
            
            print("[EXTRACT] Fase 3C: Extrayendo nombre de la ciudad con LLM...")
            
            # Extraer nombre de la ciudad
            city_result = llm_extract_city_name(data.mensaje)
            project_city = city_result.get("ciudad")
            
            if project_city:
                print(f"[EXTRACT] ✅ Ciudad: {project_city}")

        # FASE 3B: ¿Está pidiendo ubicación del proyecto?
        elif conv_data.get("contact_collected") and not conv_data.get("project_location"):
            print("[EXTRACT] Fase 3B: Extrayendo ubicación del proyecto con LLM...")
            
            # Usar LLM para extraer ubicación del proyecto
            llm_result = llm_extract_project_location(data.mensaje)
            
            if llm_result.get("barrio_proyecto"):
                proj_loc = llm_result["barrio_proyecto"]
                print(f"[EXTRACT] ✅ Ubicación proyecto: {proj_loc}")

        # FASE 3A: ¿Está pidiendo datos personales?
        elif conv_data.get("contact_requested") and not conv_data.get("contact_collected"):
            
            # Usar LLM para extraer nombre, barrio de residencia y teléfono
            llm_result = llm_extract_personal_contact(data.mensaje)
            
            if llm_result.get("nombre"):
                name = llm_result["nombre"]
            
            if llm_result.get("barrio"):
                user_barrio = llm_result["barrio"]
            
            if llm_result.get("telefono"):
                phone = llm_result["telefono"]


        # ══════════════════════════════════════════════════════════════
        # Actualizar info de contacto en BD
        # ══════════════════════════════════════════════════════════════
        if name or phone or user_barrio:
            current_info = conv_data.get("contact_info") or {}
            new_info = dict(current_info)
            
            # Actualizar solo los campos que se extrajeron
            if name:
                new_info["nombre"] = name
            if user_barrio:
                new_info["barrio"] = user_barrio
            if phone:
                new_info["telefono"] = phone
            
            conv_ref.update({"contact_info": new_info})
            
            # ✅ Marcar como "recolectado" solo si tiene LOS 3 DATOS
            tiene_nombre = bool(new_info.get("nombre"))
            tiene_barrio = bool(new_info.get("barrio"))
            tiene_telefono = bool(new_info.get("telefono"))
            
            print(f"[EXTRACT] Estado: nombre={tiene_nombre}, barrio={tiene_barrio}, tel={tiene_telefono}")
            
            if tiene_nombre and tiene_barrio and tiene_telefono:
                print(f"[EXTRACT] ✅ TODOS LOS DATOS COMPLETOS")
                conv_ref.update({"contact_collected": True})
                
                # Crear usuario en BD
                upsert_usuario_o_anon(
                    chat_id,
                    new_info["nombre"],
                    new_info["telefono"],
                    data.canal,
                    new_info["barrio"]
                )
            else:
                print(f"[EXTRACT] ⚠️ Faltan datos todavía")

        # ══════════════════════════════════════════════════════════════
        # Actualizar ubicación del proyecto (Fase 3B)
        # ══════════════════════════════════════════════════════════════
        if proj_loc:
            conv_ref.update({"project_location": proj_loc})
            print(f"[EXTRACT] ✅ Ubicación del proyecto guardada: {proj_loc}")

        # ══════════════════════════════════════════════════════════════
        # Actualizar nombre de la ciudad (Fase 3C)
        # ══════════════════════════════════════════════════════════════
        if project_city:
            conv_ref.update({"project_city": project_city})
            print(f"[EXTRACT] ✅ Ciudad del proyecto guardada: {project_city}")
        
        # ═══════════════════════════════════════════════════════════════
        # CAPA 2: Clasificación Inteligente
        # ═══════════════════════════════════════════════════════════════
        
        resumen_contexto = conv_data.get("conversacion_resumida", "")
        clasificacion = smart_classify_with_context(data.mensaje, resumen_contexto, conv_data)
        
        # ══════════════════════════════════════════════════════════════
        # VALIDAR: ¿El usuario cambió de propuesta a consulta?
        # ══════════════════════════════════════════════════════════════
        already_in_proposal_flow = bool(
            conv_data.get("proposal_collected") or
            conv_data.get("proposal_requested") or
            conv_data.get("argument_requested") or
            conv_data.get("contact_intent") == "propuesta"
        )

        # Si estaba en flujo de propuesta PERO la nueva clasificación es consulta con alta confianza
        if already_in_proposal_flow and clasificacion["tipo"] == "consulta" and clasificacion["confianza"] == "alta":
            # Usuario cambió de opinión → resetear flujo de propuesta
            conv_ref.update({
                "proposal_requested": False,
                "proposal_collected": False,
                "argument_requested": False,
                "argument_collected": False,
                "contact_requested": False,
                "contact_intent": None
            })
            already_in_proposal_flow = False

        elif already_in_proposal_flow:
            # Mantener flujo de propuesta
            clasificacion["tipo"] = "propuesta"
            clasificacion["confianza"] = "alta"
        
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
        # Si NO está en flujo de propuesta Y el mensaje tiene "?"
        # entonces NO es propuesta aunque clasificacion diga que sí
        if not already_in_proposal_flow and "?" in data.mensaje and clasificacion["tipo"] == "propuesta":
            print("[CLASSIFY] ⚠️ Mensaje tiene '?' pero clasificó como propuesta. Corrigiendo a consulta.")
            is_proposal_flow = False
            clasificacion["tipo"] = "consulta"
        
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
                    tono = detect_tone_from_argument(data.mensaje)
                    conv_ref.update({
                        "argument_collected": True,
                        "contact_requested": True,
                        "contact_intent": "propuesta",
                        "tono_detectado": tono  # ✅ NUEVO
                    })
                    
                    # ══════════════════════════════════════════════════════════════
                    # VALIDACIÓN CRÍTICA: Verificar que existe propuesta
                    # ══════════════════════════════════════════════════════════════
                    propuesta_texto = conv_data.get("current_proposal", "")
                    
                    if not propuesta_texto or len(propuesta_texto.strip()) < 10:
                        # ERROR: No hay propuesta pero llegamos hasta aquí
                        # Resetear el flujo y pedir propuesta de nuevo
                        conv_ref.update({
                            "proposal_collected": False,
                            "argument_collected": False,
                            "argument_requested": False,
                            "contact_requested": False,
                            "proposal_requested": True,
                            "proposal_nudge_count": 0
                        })
                        texto = "Parece que no tengo clara tu propuesta. ¿Podrías contármela en 1-2 frases?"
                        append_mensajes(conv_ref, [
                            {"role": "user", "content": data.mensaje},
                            {"role": "assistant", "content": texto}
                        ])
                        return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                    

                    # ══════════════════════════════════════════════════════════════
                    # Determinar qué datos faltan (SOLO datos personales)
                    # ══════════════════════════════════════════════════════════════
                    current_info = conv_data.get("contact_info") or {}
                    missing = []

                    if not (current_info.get("nombre") or name): 
                        missing.append("nombre")
                    if not (current_info.get("barrio") or user_barrio): 
                        missing.append("barrio")
                    if not (current_info.get("telefono") or phone): 
                        missing.append("celular")
                    
                    # ══════════════════════════════════════════════════════════════
                    # Generar feedback personalizado con LLM
                    # ══════════════════════════════════════════════════════════════
                    nombre_actual = current_info.get("nombre") or name
                    argumento_texto = data.mensaje  # El mensaje actual ES el argumento
                    
                    # Usar LLM para generar feedback personalizado
                    texto = positive_feedback_after_argument(
                        nombre_actual, 
                        missing,
                        propuesta_texto,
                        argumento_texto
                    )
                    
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
            

            # ═══════════════════════════════════════════════════════════════
            # FASE 3A: Recopilar datos personales (nombre + teléfono + barrio residencia)
            # ═══════════════════════════════════════════════════════════════
            if conv_data.get("argument_collected") and conv_data.get("contact_requested") and not conv_data.get("contact_collected"):
                
                # ══════════════════════════════════════════════════════════════
                # Detectar si pregunta "¿para qué?"
                # ══════════════════════════════════════════════════════════════
                if is_asking_why(data.mensaje):
                    texto = (
                        "Es para darle seguimiento a tu propuesta y poder escalarlo con el equipo de Wilder. "
                        "Así podemos informarte sobre el avance y gestionar tu caso de manera personalizada. "
                        "¿Me compartes tu nombre, el barrio donde vives y tu número de celular?"
                    )
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                # ══════════════════════════════════════════════════════════════
                # NUEVO: Detectar si rechaza dar datos
                # ══════════════════════════════════════════════════════════════
                if detect_contact_refusal(data.mensaje):
                    refusal_count = int(conv_data.get("contact_refusal_count", 0))
                    
                    # Primera vez que rechaza → explicar
                    if refusal_count == 0:
                        conv_ref.update({"contact_refusal_count": 1})
                        texto = (
                            "Entiendo tu preocupación. "
                            "Tus datos son importantes para poder darte seguimiento personalizado "
                            "y mantenerte informado sobre el avance de tu propuesta. "
                            "Toda tu información está protegida y solo la usa el equipo de Wilder. "
                            "¿Me los compartes?"
                        )
                        append_mensajes(conv_ref, [
                            {"role": "user", "content": data.mensaje},
                            {"role": "assistant", "content": texto}
                        ])
                        return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                    
                    # Segunda vez → respetar decisión y continuar con ubicación
                    else:
                        conv_ref.update({
                            "contact_refused": True,
                            "contact_collected": False,
                            "contact_requested": False
                        })
                        
                        # Continuar pidiendo ubicación del proyecto
                        texto = (
                            "Entiendo y respeto tu decisión. "
                            "Para continuar con tu propuesta, necesito saber en qué barrio sería el proyecto. "
                            "¿Me compartes esa información?"
                        )
                        
                        append_mensajes(conv_ref, [
                            {"role": "user", "content": data.mensaje},
                            {"role": "assistant", "content": texto}
                        ])
                        return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                # ══════════════════════════════════════════════════════════════
                # Verificar qué datos personales faltan
                # ══════════════════════════════════════════════════════════════
                current_info = conv_data.get("contact_info") or {}
                missing = []
                
                if not (current_info.get("nombre") or name): 
                    missing.append("nombre")
                if not (current_info.get("barrio") or user_barrio): 
                    missing.append("barrio")
                if not (current_info.get("telefono") or phone): 
                    missing.append("celular")
                
                if missing:
                    # Todavía faltan datos personales → pedir de nuevo
                    nombre_actual = current_info.get("nombre") or name
                    texto = build_contact_request_with_explanation(nombre_actual, missing)
                    
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                # ══════════════════════════════════════════════════════════════
                # ✅ YA TIENE TODOS LOS DATOS PERSONALES → PEDIR UBICACIÓN DEL PROYECTO
                # ══════════════════════════════════════════════════════════════
                conv_ref.update({"contact_collected": True})
                texto = build_project_location_request()
                
                append_mensajes(conv_ref, [
                    {"role": "user", "content": data.mensaje},
                    {"role": "assistant", "content": texto}
                ])
                return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                # ══════════════════════════════════════════════════════════════
                # ✅ Ya tiene todos los datos personales → IR A FASE 3B
                # ══════════════════════════════════════════════════════════════
                # Marcar como completo y pedir ubicación del proyecto
                # NO hacer return aquí - continuar al siguiente if (FASE 3B)
                
                # ══════════════════════════════════════════════════════════════
                # Ya tiene todos los datos personales → pasar a FASE 3B
                # ══════════════════════════════════════════════════════════════
                # (el flujo continúa abajo en FASE 3B)

            # ═══════════════════════════════════════════════════════════════
            # FASE 3B: Pedir ubicación del proyecto
            # ═══════════════════════════════════════════════════════════════
            if (conv_data.get("contact_collected") or conv_data.get("contact_refused")) and not conv_data.get("project_location"):
                
                # Detectar si pregunta "¿para qué necesitas el barrio?"
                if is_asking_why(data.mensaje):
                    texto = explain_why_project_location()
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                # ¿Ya extrajo la ubicación del proyecto en este mensaje?
                if proj_loc:
                    # ✅ Usuario acaba de dar la ubicación
                    conv_ref.update({"project_location": proj_loc})
                    
                    # Si dijo "Ciudad" → pedir nombre de la ciudad
                    if proj_loc == "Ciudad":
                        texto = "¿Cuál ciudad específicamente?"
                        
                        append_mensajes(conv_ref, [
                            {"role": "user", "content": data.mensaje},
                            {"role": "assistant", "content": texto}
                        ])
                        return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                    
                    # Si dio barrio específico → terminar
                    else:
                        nombre_txt = conv_data.get("contact_info", {}).get("nombre", "")
                        
                        if conv_data.get("contact_refused"):
                            # Usuario rechazó dar datos → despedida sin promesa de seguimiento
                            texto = (
                                "Perfecto. Hemos registrado tu propuesta para el barrio {0}. "
                                "¡Gracias por tu participación!"
                            ).format(proj_loc)
                        else:
                            # Usuario dio datos → despedida con promesa de seguimiento
                            texto = (f"Perfecto, {nombre_txt}. " if nombre_txt else "Perfecto. ")
                            texto += "Con todos estos datos vamos a escalar tu propuesta. ¡Gracias por tu participación!"
                        
                        append_mensajes(conv_ref, [
                            {"role": "user", "content": data.mensaje},
                            {"role": "assistant", "content": texto}
                        ])
                        return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                else:
                    # Todavía falta la ubicación → pedirla
                    texto = build_project_location_request()
                    
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
            # ═══════════════════════════════════════════════════════════════
            # FASE 3C: Pedir nombre de la ciudad
            # ═══════════════════════════════════════════════════════════════
            if (conv_data.get("contact_collected") and 
                conv_data.get("project_location") == "Ciudad" and 
                not conv_data.get("project_city")):
                
                # ¿Ya extrajo el nombre de la ciudad?
                if project_city:
                    # ✅ Usuario acaba de dar el nombre de la ciudad
                    conv_ref.update({"project_city": project_city})
                    
                    nombre_txt = conv_data.get("contact_info", {}).get("nombre", "")
                    texto = (f"Perfecto, {nombre_txt}. " if nombre_txt else "Perfecto. ")
                    texto += f"Vamos a escalar tu propuesta para {project_city}. ¡Gracias por tu participación!"
                    
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                else:
                    # Todavía no dio el nombre → volver a pedir
                    texto = "¿Cuál ciudad específicamente? (Ej: Medellín, Manizales, Bogotá)"
                    
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

            # ═══════════════════════════════════════════════════════════════
            # FASE 3D: Todo completo
            # ═══════════════════════════════════════════════════════════════
            # Tiene datos personales + ubicación del proyecto + (ciudad si aplica)
            if conv_data.get("contact_collected") and conv_data.get("project_location"):
                
                # ¿Falta la ciudad?
                if conv_data.get("project_location") == "Ciudad" and not conv_data.get("project_city"):
                    # No hacer nada, dejar que FASE 3C maneje esto
                    pass
                else:
                    # Ya tiene TODO → confirmar y terminar
                    nombre_txt = conv_data.get("contact_info", {}).get("nombre", "")
                    texto = (f"Gracias, {nombre_txt}. " if nombre_txt else "Gracias. ")
                    texto += "Ya tenemos toda la información para escalar tu propuesta. Te mantendremos informado del avance."
                    
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
        
        # ══════════════════════════════════════════════════════════════
        # LÓGICA MEJORADA: Detectar cambio de tipo (Consulta → Propuesta)
        # ══════════════════════════════════════════════════════════════
        categorias_existentes = conv_data.get("categoria_general") or []
        titulos_existentes = conv_data.get("titulo_propuesta") or []

        if isinstance(categorias_existentes, str):
            categorias_existentes = [categorias_existentes]
        if isinstance(titulos_existentes, str):
            titulos_existentes = [titulos_existentes]

        # ──────────────────────────────────────────────────────────────
        # CASO 1: Usuario cambió de Consulta → Propuesta
        # ──────────────────────────────────────────────────────────────
        tiene_consulta = "Consulta" in categorias_existentes
        es_propuesta_nueva = (tipo == "propuesta" and categoria != "Consulta")

        if tiene_consulta and es_propuesta_nueva:
            # REEMPLAZAR Consulta por la categoría de propuesta
            categorias_existentes = [c for c in categorias_existentes if c != "Consulta"]
            categorias_existentes.append(categoria)
            
            # REEMPLAZAR título de consulta por título de propuesta
            # Remover títulos que no sean específicos de propuesta
            titulos_existentes = []
            titulos_existentes.append(titulo)
            
            
        # ──────────────────────────────────────────────────────────────
        # CASO 2: Normal (sin cambio de tipo)
        # ──────────────────────────────────────────────────────────────
        else:
            # Agregar categoría si no existe
            if categoria not in categorias_existentes:
                categorias_existentes.append(categoria)
            
            # Agregar título si no existe (normalizado)
            titulo_normalizado = _normalize_text(titulo)
            titulos_normalizados_existentes = [_normalize_text(t) for t in titulos_existentes]
            
            if titulo_normalizado not in titulos_normalizados_existentes:
                titulos_existentes.append(titulo)

        # ──────────────────────────────────────────────────────────────
        # Guardar en BD
        # ──────────────────────────────────────────────────────────────
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