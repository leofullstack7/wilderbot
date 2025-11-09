# ============================
#  WilderBot API - VERSIÓN CORREGIDA
#  Correcciones:
#  1. Bot se identifica como ASISTENTE de Wilder (no como Wilder)
#  2. Solo saluda UNA VEZ al inicio
#  3. Respeta estrictamente información RAG (no inventa)
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

# ✅ CORRECCIÓN: Eliminar saludos redundantes
def remove_redundant_greetings(texto: str, historial: List[Dict[str, str]]) -> str:
    """Elimina saludos si ya hubo interacción previa."""
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

    try:
        resumen = summarize_conversation_brief(arr, max_chars=100)
        conv_ref.update({"resumen": resumen})
    except:
        pass
    
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

# ═══════════════════════════════════════════════════════════════════════
# CONTEXTO
# ═══════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════
# DETECCIÓN
# ═══════════════════════════════════════════════════════════════════════

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

def is_proposal_intent_heuristic(text: str) -> bool:
    t = _normalize_text(text)
    kw = ["propongo", "propuesta", "sugerencia", "mi idea", "quiero proponer"]
    return any(k in t for k in kw)

def looks_like_proposal_content(text: str) -> bool:
    if is_proposal_denial(text):
        return False
    t = _normalize_text(text)
    if re.search(r'\b(arregl|mejor|constru|instal|crear|paviment|ilumin)\w*', t):
        return True
    if re.search(r'\b(parque|semaforo|luminaria|cancha)\b', t):
        return True
    return len(t) >= 20

def has_argument_text(t: str) -> bool:
    t = _normalize_text(t)
    if any(k in t for k in ["porque", "ya que", "debido", "es importante"]):
        return True
    if len(t) >= 30:
        return True
    return False

# ═══════════════════════════════════════════════════════════════════════
# EXTRACCIÓN
# ═══════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════
# CAPAS
# ═══════════════════════════════════════════════════════════════════════

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

# ✅ CORRECCIÓN: Reformulación mejorada
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

# ✅ CORRECCIÓN: Validar relevancia RAG
def validate_rag_relevance(rag_hits: List[Dict]) -> bool:
    if not rag_hits:
        return False
    relevant = any(hit.get("score", 0) > 0.7 for hit in rag_hits)
    if not relevant:
        print("[RAG] ⚠️  Sin hits relevantes")
    return relevant

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def build_contact_request(missing: List[str]) -> str:
    etiquetas = {"nombre": "tu nombre", "barrio": "tu barrio", "celular": "celular"}
    pedir = [etiquetas[m] for m in missing if m in etiquetas]
    frase = pedir[0] if len(pedir) == 1 else (", ".join(pedir[:-1]) + " y " + pedir[-1])
    return f"¿Me compartes {frase}?"

def craft_argument_question(name: Optional[str], project_location: Optional[str] = None) -> str:
    saludo = f"{name}, " if name else ""
    return f"{saludo}¿Por qué es importante?"

def positive_ack_and_request_argument(name: Optional[str], project_location: Optional[str] = None) -> str:
    return "Excelente idea. ¿Por qué sería importante?"

def strip_contact_requests(texto: str) -> str:
    return texto

# ✅ CORRECCIÓN CRÍTICA: Generación con validación RAG
def layer4_generate_response_with_memory(
    ctx: ConversationContext,
    clasificacion: Dict[str, Any],
    rag_hits: List[Dict]
) -> str:
    """Generación con respeto estricto a RAG."""
    
    contexto_rag = "\n".join([f"- {h['texto']}" for h in rag_hits if h.get("texto")])
    
    # ✅ System prompt corregido
    system_msg = (
        "Eres el ASISTENTE de Wilder Escobar, Representante a la Cámara.\n\n"
        "REGLAS ESTRICTAS:\n"
        "1. NO te presentes como Wilder, eres su ASISTENTE\n"
        "2. NO saludes si ya hay historial\n"
        "3. Usa SOLO información del contexto proporcionado\n"
        "4. Si el contexto no tiene la info, di: 'No tengo esa información específica'\n"
        "5. Máximo 3 frases\n\n"
    )
    
    if ctx.has_reference and ctx.resumen:
        system_msg += f"CONTEXTO PREVIO: {ctx.resumen}\n"
        system_msg += "Mantén coherencia con ese tema.\n\n"
    
    if clasificacion["tipo"] == "consulta":
        system_msg += (
            "CONSULTA: Usa ÚNICAMENTE el contexto.\n"
            "NO inventes información.\n"
            "Si no está en el contexto, dilo claramente.\n"
        )
    
    msgs = [{"role": "system", "content": system_msg}]
    
    if ctx.historial:
        msgs.extend(ctx.historial[-6:])
    
    msgs.append({
        "role": "user",
        "content": (
            f"CONTEXTO VERIFICADO:\n{contexto_rag}\n\n"
            f"PREGUNTA:\n{ctx.mensaje}\n\n"
            f"Responde SOLO con info del contexto."
        )
    })
    
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.1,  # ✅ Baja temperatura
            max_tokens=280
        )
        
        texto = completion.choices[0].message.content.strip()
        texto = limit_sentences(texto, 3)
        
        # ✅ Eliminar saludos
        texto = remove_redundant_greetings(texto, ctx.historial)
        
        return texto
        
    except Exception as e:
        return "Disculpa, tuve un problema. ¿Reformulas?"

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

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
        
        # PROPUESTAS (simplificado)
        if clasificacion["tipo"] == "propuesta" or ctx.in_proposal_flow:
            if not ctx.proposal_collected:
                if looks_like_proposal_content(data.mensaje):
                    conv_ref.update({
                        "current_proposal": extract_proposal_text(data.mensaje),
                        "proposal_collected": True,
                        "argument_requested": True,
                    })
                    texto = positive_ack_and_request_argument(None, None)
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
        
        # ✅ Validar RAG
        if not validate_rag_relevance(hits):
            texto = "No tengo información específica sobre eso. ¿Puedo ayudarte con algo más?"
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

# ═══════════════════════════════════════════════════════════════════════
# CLASIFICACIÓN COMPLETA (CONSULTAS Y PROPUESTAS)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/clasificar")
async def clasificar(body: ClasificarIn):
    """
    Clasifica la conversación y guarda categorías.
    Maneja CONSULTAS y PROPUESTAS de forma inteligente.
    """
    try:
        chat_id = body.chat_id
        conv_ref = db.collection("conversaciones").document(chat_id)
        snap = conv_ref.get()
        
        if not snap.exists:
            return {"ok": False, "mensaje": "Conversación no encontrada"}
        
        conv_data = snap.to_dict() or {}
        
        # ═══════════════════════════════════════════════════════════════
        # PASO 1: Obtener el último mensaje del usuario
        # ═══════════════════════════════════════════════════════════════
        mensajes = conv_data.get("mensajes", [])
        ultimo_usuario = ""
        
        # Buscar el último mensaje del usuario (de atrás hacia adelante)
        for m in reversed(mensajes):
            if m.get("role") == "user":
                ultimo_usuario = m.get("content", "")
                break
        
        if not ultimo_usuario:
            return {"ok": False, "mensaje": "No hay mensajes para clasificar"}
        
        # ═══════════════════════════════════════════════════════════════
        # PASO 2: Detectar si es CONSULTA o PROPUESTA
        # ═══════════════════════════════════════════════════════════════
        
        # ¿Tiene propuesta guardada?
        propuesta = conv_data.get("current_proposal") or ""
        
        # Clasificar según el contexto
        if propuesta:
            # HAY PROPUESTA → Clasificar como propuesta
            tipo = "propuesta"
            texto_a_clasificar = propuesta
        else:
            # NO HAY PROPUESTA → Podría ser consulta
            # Verificar si parece consulta
            es_consulta_heuristica = (
                ("?" in ultimo_usuario) or
                any(kw in _normalize_text(ultimo_usuario) 
                    for kw in ["que", "qué", "como", "cómo", "cuando", "cuándo", 
                              "donde", "dónde", "wilder", "ley", "proyecto"])
            )
            
            if es_consulta_heuristica:
                tipo = "consulta"
                texto_a_clasificar = ultimo_usuario
            else:
                # No es consulta ni tiene propuesta → skip
                return {"ok": True, "skipped": True, "reason": "ni_consulta_ni_propuesta"}
        
        print(f"[CLASIFICAR] Tipo detectado: {tipo}")
        print(f"[CLASIFICAR] Texto: {texto_a_clasificar[:100]}...")
        
        # ═══════════════════════════════════════════════════════════════
        # PASO 3: Clasificar con LLM según el tipo
        # ═══════════════════════════════════════════════════════════════
        
        if tipo == "consulta":
            # ─────────────────────────────────────────────────────────
            # CLASIFICAR CONSULTA
            # ─────────────────────────────────────────────────────────
            sys = (
                "Clasifica esta CONSULTA ciudadana.\n"
                "Devuelve SOLO JSON con:\n"
                "{\n"
                "  \"categoria_general\": \"Consulta\",\n"
                "  \"titulo_propuesta\": \"[Tema de la consulta en 5-8 palabras]\",\n"
                "  \"tono_detectado\": \"neutral\"\n"
                "}\n\n"
                "Ejemplos de títulos:\n"
                "- 'Ley 2420 educación pospandemia'\n"
                "- 'Proyectos salud adultos mayores'\n"
                "- 'Posición movilidad sostenible'\n"
            )
            usr = f"Consulta del ciudadano:\n{texto_a_clasificar}"
            
        else:  # tipo == "propuesta"
            # ─────────────────────────────────────────────────────────
            # CLASIFICAR PROPUESTA
            # ─────────────────────────────────────────────────────────
            ubicacion = conv_data.get("project_location") or ""
            
            sys = (
                "Clasifica esta PROPUESTA ciudadana.\n"
                "Devuelve SOLO JSON con:\n"
                "{\n"
                "  \"categoria_general\": \"[Infraestructura Urbana|Seguridad|Movilidad|Educación|Salud|Vivienda|Empleo|Medio Ambiente]\",\n"
                "  \"titulo_propuesta\": \"[Acción + Qué + Dónde en máx 60 chars]\",\n"
                "  \"tono_detectado\": \"propositivo\"\n"
                "}\n\n"
                "Reglas para el título:\n"
                "- Comenzar con verbo (Mejorar, Construir, Arreglar, Instalar)\n"
                "- Incluir el QUÉ (alumbrado, parque, vía)\n"
                "- Incluir el DÓNDE si existe\n"
                "- Máximo 60 caracteres\n\n"
                "Ejemplos:\n"
                "- 'Mejorar alumbrado público en Laureles'\n"
                "- 'Construir parque infantil en Aranjuez'\n"
                "- 'Reparar vías barrio Popular'\n"
            )
            
            if ubicacion:
                usr = f"Propuesta: {texto_a_clasificar}\nUbicación: {ubicacion}"
            else:
                usr = f"Propuesta: {texto_a_clasificar}"
        
        # ═══════════════════════════════════════════════════════════════
        # PASO 4: Llamar al LLM
        # ═══════════════════════════════════════════════════════════════
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
            
            # Limpiar respuesta (a veces el LLM devuelve ```json ... ```)
            out = out.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(out)
            
        except Exception as e:
            print(f"[CLASIFICAR] Error LLM: {e}")
            # Fallback básico
            if tipo == "consulta":
                data = {
                    "categoria_general": "Consulta",
                    "titulo_propuesta": "Consulta ciudadana",
                    "tono_detectado": "neutral"
                }
            else:
                data = {
                    "categoria_general": "General",
                    "titulo_propuesta": "Propuesta ciudadana",
                    "tono_detectado": "propositivo"
                }
        
        # Extraer datos
        categoria = data.get("categoria_general", "General")
        titulo = data.get("titulo_propuesta", "Sin título")
        tono = data.get("tono_detectado", "neutral")
        
        print(f"[CLASIFICAR] Categoría: {categoria}")
        print(f"[CLASIFICAR] Título: {titulo}")
        
        # ═══════════════════════════════════════════════════════════════
        # PASO 5: Guardar en Firestore (ACUMULATIVO)
        # ═══════════════════════════════════════════════════════════════
        
        # Obtener categorías existentes
        categorias_existentes = conv_data.get("categoria_general") or []
        titulos_existentes = conv_data.get("titulo_propuesta") or []
        
        # Convertir a lista si es string (por compatibilidad)
        if isinstance(categorias_existentes, str):
            categorias_existentes = [categorias_existentes]
        if isinstance(titulos_existentes, str):
            titulos_existentes = [titulos_existentes]
        
        # Agregar nueva categoría si no existe
        if categoria not in categorias_existentes:
            categorias_existentes.append(categoria)
        
        # Agregar nuevo título si no existe
        titulo_normalizado = _normalize_text(titulo)
        titulos_normalizados_existentes = [_normalize_text(t) for t in titulos_existentes]
        
        if titulo_normalizado not in titulos_normalizados_existentes:
            titulos_existentes.append(titulo)
        
        # Actualizar en BD
        conv_ref.update({
            "categoria_general": categorias_existentes,
            "titulo_propuesta": titulos_existentes,
            "tono_detectado": tono,
            "ultima_fecha": firestore.SERVER_TIMESTAMP
        })
        
        print(f"[CLASIFICAR] ✅ Guardado: {categorias_existentes} / {titulos_existentes}")
        
        # ═══════════════════════════════════════════════════════════════
        # PASO 6: Retornar resultado
        # ═══════════════════════════════════════════════════════════════
        return {
            "ok": True,
            "clasificacion": {
                "tipo": tipo,  # 'consulta' o 'propuesta'
                "categoria_general": categoria,
                "titulo_propuesta": titulo,
                "tono_detectado": tono,
                "todas_categorias": categorias_existentes,  # Para el front
                "todos_titulos": titulos_existentes  # Para el front
            }
        }
        
    except Exception as e:
        print(f"[CLASIFICAR] ❌ Error fatal: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)