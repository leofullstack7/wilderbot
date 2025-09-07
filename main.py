# ============================
#  WilderBot API (FastAPI)
#  - /responder : RAG + historial + guardado en tu BD
#  - /clasificar: clasifica la conversación y actualiza panel
# ============================

from fastapi import FastAPI
from pydantic import BaseModel

# --- OpenAI (SDK v1.x) para chat y embeddings
from openai import OpenAI

# --- Pinecone (SDK nuevo) para búsqueda vectorial
from pinecone import Pinecone

# --- Tipos útiles
from typing import Optional, List, Dict, Any, Tuple

# --- Firestore helpers (incrementos atómicos)
from google.cloud.firestore_v1 import Increment

import json
import os
from dotenv import load_dotenv
import re
import math

from api.ingest import router as ingest_router

from fastapi.middleware.cors import CORSMiddleware

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

BOT_INTRO_TEXT = os.getenv(
    "BOT_INTRO_TEXT",
    "¡Hola! Soy la mano derecha de Wilder Escobar. Estoy aquí para escuchar y canalizar tus "
    "problemas, propuestas o reconocimientos. ¿Qué te gustaría contarme hoy?"
)

# === Clientes ===
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# === Firestore Admin ===
import firebase_admin
from firebase_admin import credentials, firestore

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
    canal: Optional[str] = None       # telegram | whatsapp | web
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
#  Helpers de BD
# =========================================================

def upsert_usuario_o_anon(chat_id: str, nombre: Optional[str], telefono: Optional[str], canal: Optional[str]) -> str:
    usuario_id = chat_id
    if telefono:
        ref = db.collection("usuarios").document(usuario_id)
        doc = ref.get()
        if not doc.exists:
            ref.set({
                "nombre": nombre or "",
                "telefono": telefono,
                "barrio": None,
                "fecha_registro": firestore.SERVER_TIMESTAMP,
                "chats": [chat_id],
                "canal": canal or "web",
            })
        else:
            ref.update({
                "nombre": nombre or doc.to_dict().get("nombre", ""),
                "telefono": telefono,
                "chats": firestore.ArrayUnion([chat_id]),
                "canal": canal or doc.to_dict().get("canal", "web"),
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
            ref.update({
                "nombre": nombre or doc.to_dict().get("nombre", None),
                "chats": firestore.ArrayUnion([chat_id]),
                "canal": canal or doc.to_dict().get("canal", "web"),
            })
    return usuario_id


def ensure_conversacion(chat_id: str, usuario_id: str, faq_origen: Optional[str], canal: Optional[str]):
    """
    Crea conversaciones/{chat_id} si no existe y añade campos de contacto + flags de argumento.
    """
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

            # Estado de conversación/tema
            "last_topic_vec": None,
            "last_topic_summary": None,
            "awaiting_topic_confirm": False,
            "candidate_new_topic_summary": None,
            "candidate_new_topic_vec": None,
            "topics_history": [],

            # Flujo argumento + contacto
            "argument_requested": False,
            "argument_collected": False,

            # Flujo específico de propuestas/sugerencias
            "proposal_requested": False,
            "proposal_collected": False,
            "current_proposal": None,

            "contact_intent": None,   # 'propuesta' | 'problema' | 'reconocimiento' | 'otro'
            "contact_requested": False,
            "contact_collected": False,
            "contact_refused": False,
            "contact_info": {"nombre": None, "barrio": None, "telefono": None},

            # Lugar de la propuesta (no confundir con barrio de residencia)
            "project_location": None,
        })
    else:
        # Solo tocamos 'canal' si viene; si no, no borres el existente
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

# =========================================================
#  Text utils
# =========================================================

def cosine_sim(a: List[float] | None, b: List[float] | None) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

def _normalize_text(t: str) -> str:
    t = t.lower()
    t = (t.replace("á","a").replace("é","e").replace("í","i")
           .replace("ó","o").replace("ú","u").replace("ü","u"))
    t = re.sub(r"[^a-zñ0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def is_plain_greeting(text: str) -> bool:
    if not text:
        return False
    t = _normalize_text(text)
    kws = ("hola","holaa","holaaa","buenas","buenos dias","buenas tardes","buenas noches","como estas","que mas","q mas","saludos")
    short = len(t) <= 30
    has_kw = any(k in t for k in kws)
    topicish = any(w in t for w in (
        "arregl","propuesta","proponer","daño","danada","hueco",
        "parque","colegio","via","salud","seguridad",
        "ayuda","necesito","quiero","repar","denuncia","idea"
    ))
    return short and has_kw and not topicish

def has_argument_text(t: str) -> bool:
    """
    Heurística para detectar 'argumento/razón'.
    Más estricta: NO dispara por 'para' suelto (evita falsos positivos como 'para los niños').
    """
    t = _normalize_text(t)
    keys = [
        "porque", "ya que", "debido", "por que", "porqué",
        "afecta", "impacta", "riesgo", "peligro",
        "falta", "no hay", "urge", "es necesario", "contamina",
        "seguridad", "salud", "empleo", "movilidad", "ambiental"
    ]
    return any(k in t for k in keys)

def has_argument_text(t: str) -> bool:
    """
    Heurística para detectar 'argumento/razón'.
    Más estricta: NO dispara por 'para' suelto (evita falsos positivos como 'para los niños').
    """
    t = _normalize_text(t)
    keys = [
        "porque", "ya que", "debido", "por que", "porqué",
        "afecta", "impacta", "riesgo", "peligro",
        "falta", "no hay", "urge", "es necesario", "contamina",
        "seguridad", "salud", "empleo", "movilidad", "ambiental"
    ]
    return any(k in t for k in keys)

# === NUEVO: heurística fuerte para detectar intención de propuesta/sugerencia ===
def is_proposal_intent(text: str) -> bool:
    t = _normalize_text(text)
    kw = [
        "propongo", "propuesta", "sugerencia", "sugerir", "sugiero",
        "mi idea", "mi propuesta", "quiero proponer", "quisiera proponer",
        "me gustaria proponer", "me gustaria que", "planteo", "plantear",
        "propongo que", "propuse", "propone"
    ]
    return any(k in t for k in kw)

# === NUEVO: recortar a N oraciones (para contener respuestas del LLM) ===
def limit_sentences(text: str, max_sentences: int = 3) -> str:
    parts = re.split(r'(?<=[\.\?!])\s+', text.strip())
    out = " ".join([p for p in parts if p][:max_sentences]).strip()
    return out or text


# =========================================================
#  Extracciones específicas
# =========================================================

def extract_user_name(text: str) -> Optional[str]:
    m = re.search(r'\b(?:soy|me llamo|mi nombre es)\s+([A-Za-zÁÉÍÓÚÑáéíóúñ ]{2,40})', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Nombre al inicio antes de coma o conectores
    m = re.search(r'^\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})\s*(?=,|\s+vivo\b|\s+soy\b|\s+mi\b|\s+desde\b|\s+del\b|\s+de\b)', text)
    if m:
        posible = m.group(1).strip()
        if posible.lower() not in {"hola","buenas","buenos dias","buenas tardes","buenas noches"}:
            return posible
    return None

def extract_phone(text: str) -> Optional[str]:
    m = re.search(r'(\+?\d[\d\s\-]{7,14}\d)', text)
    if not m:
        return None
    tel = re.sub(r'\D', '', m.group(1))
    return tel if 8 <= len(tel) <= 12 else None

def extract_user_barrio(text: str) -> Optional[str]:
    """
    Solo tomamos barrio de RESIDENCIA con patrones explícitos (para no confundir con el del proyecto).
    """
    patterns = [
        r'\bvivo en el barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50})',
        r'\bresido en el barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50})',
        r'\bsoy del barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50})',
        r'\bmi barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50})',
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" .,")
    return None

def _titlecase(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).title()

def _clean_barrio_fragment(s: str) -> str:
    # corta en conectores típicos: para/por/que/donde/con/coma/punto/fin
    s = re.split(r"\s+(?:para|por|que|donde|con)\b|[,.;]|$", s, maxsplit=1, flags=re.IGNORECASE)[0]
    return _titlecase(s)

def extract_project_location(text: str) -> Optional[str]:
    # patrón con lookahead que se detiene antes de conectores o puntuación
    m = re.search(
        r'\ben el barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50}?)(?=(?:\s+(?:para|por|que|donde|con)\b|[,.;]|$))',
        text, flags=re.IGNORECASE
    )
    if m:
        return _clean_barrio_fragment(m.group(1))

    # casos como "construir/instalar ... en el barrio X ..."
    if re.search(r'\b(construir|hacer|instalar|crear|mejorar)\b', text, flags=re.IGNORECASE):
        m = re.search(
            r'\ben el barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50}?)(?=(?:\s+(?:para|por|que|donde|con)\b|[,.;]|$))',
            text, flags=re.IGNORECASE
        )
        if m:
            return _clean_barrio_fragment(m.group(1))
    return None

def extract_proposal_text(text: str) -> str:
    """Extrae la propuesta desde el mensaje del ciudadano, limpiando frases iniciales típicas."""
    t = text.strip()

    # Quita una posible auto-presentación inicial
    t = re.sub(
        r'^\s*(?:soy|me llamo|mi nombre es)\s+[A-Za-zÁÉÍÓÚÑáéíóúñ ]{2,40}[,:\-–—]?\s*',
        '',
        t, flags=re.IGNORECASE
    )

    # Quita 'me gustaría proponer', 'quiero proponer', 'propongo que', etc.
    t = re.sub(
        r'^\s*(?:me\s+gustar[íi]a\s+(?:proponer|que)\s+|quisiera\s+(?:proponer|que)\s+|'
        r'quiero\s+proponer\s+|propongo\s+que\s+|propongo\s+)',
        '',
        t, flags=re.IGNORECASE
    )

    t = re.sub(r'\s+', ' ', t).strip()
    return limit_sentences(t, 2)


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

# =========================================================
#  Pequeñas “LLM tools” (clasificadores ligeros)
# =========================================================

def llm_decide_turn(
    last_topic_summary: str,
    awaiting_confirm: bool,
    last_two_turns: List[Dict[str, str]],
    current_text: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    model = model or OPENAI_MODEL
    sys = (
        "Eres un asistente que clasifica el rol conversacional de un mensaje dado el tema vigente.\n"
        "Responde SOLO un JSON válido con claves: action, reason, current_summary, topic_label.\n"
        "Reglas:\n"
        "- greeting_smalltalk: saludos/cordialidad.\n"
        "- continue_topic: sigue el mismo tema.\n"
        "- new_topic: tema distinto (pide confirmación).\n"
        "- confirm_new_topic / reject_new_topic.\n"
        "- meta: 'ok', 'gracias', etc."
    )
    last_turns_text = "\n".join([f"{t.get('role')}: {t.get('content','')}" for t in last_two_turns[-2:]])
    usr = (
        f"Tema vigente: {last_topic_summary or '(ninguno)'}\n"
        f"¿Esperando confirmación?: {'Sí' if awaiting_confirm else 'No'}\n"
        f"Últimos turnos:\n{last_turns_text}\n\n"
        f"Mensaje actual:\n{current_text}\n\n"
        "Devuelve el JSON ahora."
    )
    out = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
        temperature=0.1,
        max_tokens=220
    ).choices[0].message.content

    try:
        data = json.loads(out)
    except Exception:
        data = {"action":"continue_topic","reason":"fallback","current_summary":current_text[:120],"topic_label":""}

    data.setdefault("action","continue_topic")
    data.setdefault("current_summary", current_text[:120])
    data.setdefault("topic_label","")
    data.setdefault("reason","")
    return data


def llm_contact_policy(summary_so_far: str, last_user: str) -> Dict[str, Any]:
    sys = (
        "Eres un clasificador de intención y oportunidad de contacto para un bot cívico.\n"
        "Analiza el tema/resumen y el último mensaje del ciudadano.\n"
        "Responde SOLO JSON con: should_request(bool), intent('propuesta'|'problema'|'reconocimiento'|'otro'), reason.\n"
        "Reglas:\n"
        "- PIDE contacto cuando hay propuesta/acción específica o problema concreto con detalles.\n"
        "- NO preguntes si es un reconocimiento/elogio sin pedido de gestión.\n"
        "- Evita desviar con preguntas como '¿ya hablaste con X?'. El bot es el intermediario."
    )
    usr = (
        f"Tema/Resumen vigente: {summary_so_far or '(ninguno)'}\n"
        f"Último mensaje del ciudadano: {last_user}\n"
        "Devuelve el JSON ahora."
    )
    out = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
        temperature=0.0,
        max_tokens=120
    ).choices[0].message.content
    try:
        data = json.loads(out)
    except Exception:
        data = {"should_request": False, "intent": "otro", "reason": "fallback"}
    data.setdefault("should_request", False)
    data.setdefault("intent", "otro")
    data.setdefault("reason", "")
    return data

# =========================================================
#  Contact helpers y control de flujo
# =========================================================

def build_contact_request(missing: List[str]) -> str:
    etiquetas = {"nombre": "tu nombre", "barrio": "tu barrio", "celular": "un número de contacto"}
    pedir = [etiquetas[m] for m in missing]
    frase = pedir[0] if len(pedir) == 1 else (", ".join(pedir[:-1]) + " y " + pedir[-1])
    return f"Para escalar y darle seguimiento, ¿me compartes {frase}? Lo usamos solo para informarte avances."

PRIVACY_REPLY = (
    "Entiendo perfectamente que quieras proteger tu información personal. "
    "Queremos que sepas que tus datos están completamente seguros con nosotros. "
    "Cumplimos con todas las normativas de protección de datos y solo usamos tu información "
    "para escalar tu propuesta y ayudar a que pueda hacerse realidad."
)

# --- NUEVO: helpers para propuesta/argumento/limpieza de pedidos de contacto

# --- NUEVO: helpers para propuesta/argumento/limpieza de pedidos de contacto

def craft_argument_question(name: Optional[str], project_location: Optional[str] = None) -> str:
    """
    Pregunta genérica para argumentar, con un saludo corto si hay nombre.
    """
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
    r"(compart(e|ir)|env(í|i)a|dime|indícame|facilítame).{0,40}"
    r"(tu\s+)?(nombre|barrio|celular|tel[eé]fono|n[uú]mero|contacto)", re.IGNORECASE)

def strip_contact_requests(texto: str) -> str:
    # Elimina frases que pidan datos de contacto si aún no corresponde
    sent_split = re.split(r'(?<=[\.\?!])\s+', texto.strip())
    limpio = [s for s in sent_split if not CONTACT_PATTERNS.search(s)]
    out = " ".join([s for s in limpio if s])
    return out if out else texto


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

# =========================================================
#  Prompt constructor
# =========================================================

def build_messages(
    user_text: str,
    rag_snippets: List[str],
    historial: List[Dict[str, str]],
    topic_change_suspect: bool = False,
    prev_summary: str = "",
    new_summary: str = "",
    intro_hint: bool = False,
    emphasize_contact: bool = False,
    emphasize_argument: bool = False,
    intent: str = "otro",
    contact_already_requested: bool = False,
    contact_already_collected: bool = False,
):
    contexto = "\n".join([f"- {s}" for s in rag_snippets if s.strip()])

    system_msg = (
        "Actúa como Wilder Escobar, Representante a la Cámara en Colombia.\n"
        "Tono cercano y claro. **Máximo 3–4 frases, sin párrafos largos.**\n"
        "Sé el intermediario: no preguntes si ya habló con otras autoridades.\n"
        "No saludes de nuevo. Llama por el nombre solo si ya lo sabes.\n"
        "Si pides datos, haz UNA sola solicitud (nombre, barrio y celular)."
    )

    if intro_hint:
        system_msg += "\nSi es solo un saludo sin tema, invita con UNA pregunta a contar la situación o idea."

    if emphasize_argument:
        system_msg += "\nPrioriza UNA pregunta breve para entender motivo/impacto. No pidas contacto en este turno."

    if emphasize_contact and intent in ("propuesta", "problema") and (not contact_already_collected):
        system_msg += "\nAhora pide contacto con suavidad (nombre, barrio y celular) para escalar y dar seguimiento."
    elif intent == "reconocimiento":
        system_msg += "\nSi es reconocimiento, agradece sin pedir contacto salvo que lo solicite."

    if topic_change_suspect and new_summary:
        human_q = f'¿Seguimos con "{(prev_summary or "el tema anterior")}" o pasamos a "{new_summary}"?'
        system_msg += "\nSi detectas cambio de tema, pregunta primero: " + human_q

    contexto_msg = "Contexto recuperado (frases reales de Wilder):\n" + (contexto if contexto else "(sin coincidencias relevantes)")

    msgs = [{"role": "system", "content": system_msg}]
    if historial:
        msgs.extend(historial[-8:])
    msgs.append({"role": "user", "content": f"{contexto_msg}\n\nMensaje del ciudadano:\n{user_text}"})
    return msgs

# =========================================================
#  Endpoint principal
# =========================================================

@app.post("/responder")
async def responder(data: Entrada):
    try:
        chat_id = data.chat_id or f"web_{os.urandom(4).hex()}"
        usuario_id = upsert_usuario_o_anon(chat_id, data.nombre or data.usuario, data.celular, data.canal)
        conv_ref = ensure_conversacion(chat_id, usuario_id, data.faq_origen, data.canal)

        conv_data = (conv_ref.get().to_dict() or {})
        prev_vec = conv_data.get("last_topic_vec")
        prev_sum = (conv_data.get("last_topic_summary") or "").strip()
        awaiting_confirm = bool(conv_data.get("awaiting_topic_confirm"))

        # Saludo directo
        if not prev_sum and not awaiting_confirm and is_plain_greeting(data.mensaje):
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": BOT_INTRO_TEXT}
            ])
            return {"respuesta": BOT_INTRO_TEXT, "fuentes": [], "chat_id": chat_id}

        historial_for_decider = load_historial_para_prompt(conv_ref)

        decision = llm_decide_turn(
            last_topic_summary=prev_sum,
            awaiting_confirm=awaiting_confirm,
            last_two_turns=historial_for_decider[-2:] if historial_for_decider else [],
            current_text=data.mensaje
        )
        action = decision.get("action")
        curr_sum = (decision.get("current_summary") or data.mensaje[:120]).strip()
        intro_hint = (action in ("greeting_smalltalk", "meta")) and (not prev_sum) and (not awaiting_confirm)

        if intro_hint:
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": BOT_INTRO_TEXT}
            ])
            return {"respuesta": BOT_INTRO_TEXT, "fuentes": [], "chat_id": chat_id}

        topic_change_suspect = False
        curr_vec = None

        if action in ("confirm_new_topic", "new_topic", "continue_topic"):
            curr_vec = client.embeddings.create(model=EMBEDDING_MODEL, input=data.mensaje).data[0].embedding
            if action == "new_topic":
                topic_change_suspect = True
                conv_ref.update({
                    "awaiting_topic_confirm": True,
                    "candidate_new_topic_summary": curr_sum,
                    "candidate_new_topic_vec": curr_vec,
                    "ultima_fecha": firestore.SERVER_TIMESTAMP
                })
            elif action == "confirm_new_topic":
                cand_vec = conv_data.get("candidate_new_topic_vec") or curr_vec
                cand_sum = conv_data.get("candidate_new_topic_summary") or curr_sum
                conv_ref.update({
                    "last_topic_vec": cand_vec,
                    "last_topic_summary": cand_sum,
                    "awaiting_topic_confirm": False,
                    "candidate_new_topic_summary": None,
                    "candidate_new_topic_vec": None,
                    "ultima_fecha": firestore.SERVER_TIMESTAMP
                })
        elif action == "continue_topic":
            update_payload = {
                "last_topic_summary": curr_sum,
                "awaiting_topic_confirm": False,
                "ultima_fecha": firestore.SERVER_TIMESTAMP
            }
            if prev_vec is None and curr_vec is not None:
                update_payload["last_topic_vec"] = curr_vec
            conv_ref.update(update_payload)
        elif action == "reject_new_topic":
            conv_ref.update({
                "awaiting_topic_confirm": False,
                "candidate_new_topic_summary": None,
                "candidate_new_topic_vec": None,
                "ultima_fecha": firestore.SERVER_TIMESTAMP
            })
        else:
            conv_ref.update({"ultima_fecha": firestore.SERVER_TIMESTAMP})

        # === EXTRAER datos sueltos del texto ===
        name = extract_user_name(data.mensaje)
        phone = extract_phone(data.mensaje)
        user_barrio = extract_user_barrio(data.mensaje)  # residencia
        proj_loc = extract_project_location(data.mensaje)
        if proj_loc and (conv_data.get("project_location") or "").strip().lower() != proj_loc.lower():
            conv_ref.update({"project_location": proj_loc})


        partials = {}
        if name:   partials["nombre"] = name
        if phone:  partials["telefono"] = phone
        if user_barrio: partials["barrio"] = user_barrio

        if partials:
            current_info = (conv_data.get("contact_info") or {})
            new_info = {**current_info, **{k: v or current_info.get(k) for k, v in partials.items()}}
            conv_ref.update({"contact_info": new_info})
            if phone:
                conv_ref.update({"contact_collected": True})
                upsert_usuario_o_anon(chat_id, new_info.get("nombre") or data.nombre or data.usuario, phone, data.canal)

        # === Política argumento + contacto ===
        policy = llm_contact_policy(prev_sum or curr_sum, data.mensaje)
        intent = policy.get("intent", "otro")

        # === NUEVO: si el texto del usuario sugiere propuesta, forzamos intent='propuesta'
        if is_proposal_intent(data.mensaje):
            intent = "propuesta"
                # =========================
        # =========================
        # FLUJO DETERMINISTA: PROPUESTAS / SUGERENCIAS
        # =========================
        is_proposal_flow = (
            intent == "propuesta"
            or conv_data.get("contact_intent") == "propuesta"
            or bool(conv_data.get("proposal_requested"))
            or bool(conv_data.get("proposal_collected"))
        )

        if is_proposal_flow:
            # 1) ¿Este turno YA trae la propuesta? -> capturar y pasar a argumento
            if not conv_data.get("proposal_collected"):
                # Si el texto ya contiene una propuesta (o al menos una frase concreta), la guardamos ya
                if is_proposal_intent(data.mensaje) or len(_normalize_text(data.mensaje)) >= 20:
                    proposal_text = extract_proposal_text(data.mensaje)
                    conv_ref.update({
                        "current_proposal": proposal_text,
                        "proposal_requested": True,     # iniciamos el flujo
                        "proposal_collected": True,     # y ya la tenemos
                        "argument_requested": True,     # pasamos a pedir argumento
                        "contact_intent": "propuesta",
                        "ultima_fecha": firestore.SERVER_TIMESTAMP
                    })
                    texto = positive_ack_and_request_argument(
                        name,
                        conv_data.get("project_location") or proj_loc
                    )
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

                # Si llegó con intención pero sin contenido concreto, pedir la propuesta
                conv_ref.update({
                    "proposal_requested": True,
                    "contact_intent": "propuesta",
                    "ultima_fecha": firestore.SERVER_TIMESTAMP
                })
                texto = "¡Perfecto! ¿Cuál es tu propuesta o sugerencia? Cuéntamela en una o dos frases."
                append_mensajes(conv_ref, [
                    {"role": "user", "content": data.mensaje},
                    {"role": "assistant", "content": texto}
                ])
                return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

            # 2) Ya pedimos propuesta y aún no la hemos guardado -> tomar este turno como propuesta
            if conv_data.get("proposal_requested") and not conv_data.get("proposal_collected"):
                conv_ref.update({
                    "current_proposal": extract_proposal_text(data.mensaje),
                    "proposal_collected": True,
                    "argument_requested": True,  # pasamos a etapa de argumento
                    "ultima_fecha": firestore.SERVER_TIMESTAMP
                })
                texto = positive_ack_and_request_argument(
                    name,
                    conv_data.get("project_location") or proj_loc
                )
                append_mensajes(conv_ref, [
                    {"role": "user", "content": data.mensaje},
                    {"role": "assistant", "content": texto}
                ])
                return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}


            # 3) Ya tenemos propuesta y estamos pidiendo argumento
            if conv_data.get("proposal_collected") and not conv_data.get("argument_collected"):
                if has_argument_text(data.mensaje) or len(_normalize_text(data.mensaje)) >= 20:
                    conv_ref.update({
                        "argument_collected": True,
                        "ultima_fecha": firestore.SERVER_TIMESTAMP
                    })
                    # Pide contacto (nombre, barrio, celular)
                    info_actual = (conv_data.get("contact_info") or {})
                    faltan = []
                    if not info_actual.get("nombre"):   faltan.append("nombre")
                    if not info_actual.get("barrio"):   faltan.append("barrio")
                    if not info_actual.get("telefono"): faltan.append("celular")
                    conv_ref.update({"contact_requested": True})
                    texto = build_contact_request(faltan or ["nombre", "barrio", "celular"])
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                else:
                    # Reforzar petición de argumento (corta)
                    texto = craft_argument_question(name, conv_data.get("project_location"))
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

            # 4) Ya tenemos propuesta + argumento y falta contacto
            if conv_data.get("argument_collected") and not (conv_data.get("contact_collected") or phone):
                info_actual = (conv_data.get("contact_info") or {})
                missing = []
                if not (info_actual.get("nombre") or name):        missing.append("nombre")
                if not (info_actual.get("barrio") or user_barrio): missing.append("barrio")
                if not (info_actual.get("telefono") or phone):     missing.append("celular")
                if missing:
                    conv_ref.update({"contact_requested": True})
                    texto = build_contact_request(missing)
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
            # Si ya llegó el celular, el cierre amable lo maneja el POST de abajo.

            # En cualquiera de los casos anteriores retornamos; si ya está todo completo, seguimos flujo normal.

        # ¿El usuario acaba de responder nuestra pregunta de argumento?
        last_role = historial_for_decider[-1]["role"] if historial_for_decider else None
        just_asked_argument = bool(conv_data.get("argument_requested")) and (last_role == "assistant")
        user_replied_after_arg_q = just_asked_argument and (len(_normalize_text(data.mensaje)) >= 5)

        # Consideramos argumento listo SOLO si respondió a nuestra pregunta
        # o si hay señales fuertes (porque/ya que/debido…).
        argument_ready = user_replied_after_arg_q or has_argument_text(data.mensaje)
        if argument_ready and not conv_data.get("argument_collected"):
            conv_ref.update({"argument_collected": True})

        # ¿Debemos pedir argumento (UNA sola vez)?
        need_argument_now = (intent in ("propuesta", "problema")
                             and not conv_data.get("argument_requested")
                             and not conv_data.get("argument_collected"))
        if need_argument_now:
            conv_ref.update({"argument_requested": True})

        # --- NUEVO: si toca pedir argumento, no usamos LLM; respondemos con UNA sola pregunta
        if need_argument_now:
            texto = craft_argument_question(name, proj_loc)
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto}
            ])
            return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

        # ¿El usuario rechazó dar datos?
        if detect_contact_refusal(data.mensaje):
            conv_ref.update({"contact_refused": True, "contact_requested": True})

        # ¿Debemos pedir contacto ahora? -> SOLO después de tener argumento (no en 1er turno)
        already_req = bool(conv_data.get("contact_requested"))
        already_col = bool(conv_data.get("contact_collected")) or bool(phone)
        contact_refused = bool(conv_data.get("contact_refused"))
        should_ask_now = (policy.get("should_request")
                          and intent in ("propuesta", "problema")
                          and bool(conv_data.get("argument_collected"))   # <-- clave: ya tenemos argumento
                          and not already_col
                          and not contact_refused)

        if should_ask_now and not already_req:
            conv_ref.update({"contact_intent": intent, "contact_requested": True})

        # Si toca pedir contacto y el usuario no dio nada aún -> atajo directo
        if should_ask_now and not (name or phone or user_barrio):
            info_actual = (conv_data.get("contact_info") or {})
            faltan = []
            if not info_actual.get("nombre"):   faltan.append("nombre")
            if not info_actual.get("barrio"):   faltan.append("barrio")
            if not info_actual.get("telefono"): faltan.append("celular")
            texto_directo = build_contact_request(faltan or ["celular"])
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto_directo}
            ])
            return {"respuesta": texto_directo, "fuentes": [], "chat_id": chat_id}

        # Si el usuario rechazó dar datos y ya se los habíamos pedido -> envia mensaje de tranquilidad
        if contact_refused and already_req and not already_col:
            texto = PRIVACY_REPLY
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto}
            ])
            return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

        # 4) RAG + prompt final
        hits = rag_search(data.mensaje, top_k=5)
        historial = load_historial_para_prompt(conv_ref)
        messages = build_messages(
            data.mensaje,
            [h["texto"] for h in hits],
            historial,
            topic_change_suspect=topic_change_suspect,
            prev_summary=prev_sum,
            new_summary=curr_sum,
            intro_hint=intro_hint,
            emphasize_contact=should_ask_now,
            emphasize_argument=False,   # ya no toca: si tocaba, devolvimos antes
            intent=intent,
            contact_already_requested=already_req,
            contact_already_collected=already_col,
        )

        # 5) LLM (respuesta final)
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=280
        )
        texto = completion.choices[0].message.content.strip()
        texto = limit_sentences(texto, 3)

        # --- NUEVO: si aún NO debemos pedir contacto, limpiamos cualquier intento del LLM
        if not should_ask_now:
            texto = strip_contact_requests(texto)

        # --- POST: cierre conciso si llegó teléfono en este turno ---
        if phone:
            nombre_txt = (name or (conv_data.get("contact_info") or {}).get("nombre") or "").strip()
            cierre = (f"Gracias, {nombre_txt}. " if nombre_txt else "Gracias. ")
            cierre += "Con estos datos escalamos el caso y te contamos avances."
            if "tel" in texto.lower() or "celu" in texto.lower() or len(texto) < 30:
                texto = cierre
            else:
                texto += "\n\n" + cierre

        # --- POST: si falta algo y ya toca pedir contacto -> pide lo que falte (sin repetir lo ya dado) ---
        elif should_ask_now:
            info_actual = (conv_data.get("contact_info") or {})
            missing = []
            if not (info_actual.get("nombre") or name):        missing.append("nombre")
            if not (info_actual.get("barrio") or user_barrio): missing.append("barrio")
            if not (info_actual.get("telefono") or phone):     missing.append("celular")
            if missing:
                texto = build_contact_request(missing)

        # 6) Guardar turnos
        append_mensajes(conv_ref, [
            {"role": "user", "content": data.mensaje},
            {"role": "assistant", "content": texto}
        ])

        return {"respuesta": texto, "fuentes": hits, "chat_id": chat_id}

    except Exception as e:
        return {"error": str(e)}

# =========================================================
#  Clasificación y Panel
# =========================================================

def get_prompt_base() -> str:
    doc = db.collection("configuracion").document("prompt_inicial").get()
    if doc.exists:
        data = doc.to_dict() or {}
        return data.get("prompt_base", "")
    return ""

def read_last_user_and_bot(chat_id: str) -> Tuple[str, str, dict]:
    conv_ref = db.collection("conversaciones").document(chat_id)
    snap = conv_ref.get()
    if not snap.exists:
        return "", "", {}
    data = snap.to_dict() or {}
    mensajes = data.get("mensajes", [])
    ultimo_usuario = ""
    ultima_respuesta = ""
    for m in reversed(mensajes):
        role = m.get("role")
        if role == "assistant" and not ultima_respuesta:
            ultima_respuesta = m.get("content", "")
        elif role == "user" and not ultimo_usuario:
            ultimo_usuario = m.get("content", "")
        if ultimo_usuario and ultima_respuesta:
            break
    return ultimo_usuario, ultima_respuesta, data

def build_messages_for_classify(prompt_base: str, texto_base: str, ultima_respuesta_bot: str):
    system_msg = (
        f"{prompt_base}\n\n"
        "TAREA: Clasifica la PROPUESTA del ciudadano y devuelve SOLO JSON:\n"
        '{"categoria_general":"...", "titulo_propuesta":"...", "tono_detectado":"positivo|crítico|preocupación|propositivo", "palabras_clave":["..."]}\n'
        "Reglas:\n"
        "- Ignora saludos, preguntas personales o charla que no sea propuesta.\n"
        "- El TÍTULO resume la propuesta principal en ≤ 70 caracteres.\n"
        "- No inventes ni uses frases genéricas como 'Interacción inicial'."
    )
    user_msg = f"Propuesta del ciudadano:\n{texto_base}\n\nÚltima respuesta del bot:\n{ultima_respuesta_bot}\n\nDevuelve el JSON ahora."
    return [{"role":"system","content":system_msg},{"role":"user","content":user_msg}]

def fallback_category_and_title(texto: str, location: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    t = _normalize_text(texto or "")
    loc = _titlecase(location) if location else None

    # --- Infraestructura / espacio público ---
    if any(k in t for k in ["parque", "juegos", "infantil", "polideportivo", "cancha"]):
        cat = "Infraestructura Urbana"
        tit = f"Construcción de parque infantil en {loc}" if loc else "Construcción de parque infantil"
        return cat, tit

    if any(k in t for k in ["alumbrado", "ilumin", "poste", "luminaria"]):
        cat = "Infraestructura Urbana"
        tit = f"Mejora del alumbrado público en {loc}" if loc else "Mejora del alumbrado público"
        return cat, tit

    # Puedes agregar más reglas si lo necesitas (vías, residuos, seguridad, etc.)
    return None, None

def update_panel_resumen(categoria: str, tono: str, titulo: str, usuario_id: str):
    panel_ref = db.collection("panel_resumen").document("global")
    panel_ref.set({
        "total_conversaciones": Increment(1),
        "propuestas_por_categoria": {categoria: Increment(1)},
        "resumen_tono": {tono: Increment(1)}
    }, merge=True)
    panel_ref.set({
        "ultimas_propuestas": firestore.ArrayUnion([{
            "titulo": titulo,
            "usuario_id": usuario_id,
            "categoria": categoria,
            "fecha": firestore.SERVER_TIMESTAMP
        }])
    }, merge=True)

@app.post("/clasificar")
async def clasificar(body: ClasificarIn):
    try:
        chat_id = body.chat_id
        ultimo_u, ultima_a, conv_data = read_last_user_and_bot(chat_id)
        if not ultimo_u:
            return {"ok": False, "mensaje": "No hay mensajes para clasificar."}
        
        # Solo clasificamos propuestas/sugerencias reales
        if not conv_data.get("proposal_collected"):
            return {"ok": True, "skipped": True, "reason": "sin_propuesta"}

        # (opcional, aún más estricto)
        if (conv_data.get("contact_intent") or "") != "propuesta":
            return {"ok": True, "skipped": True, "reason": "intencion_no_es_propuesta"}


        decision_cls = llm_decide_turn(
            last_topic_summary=(conv_data.get("last_topic_summary") or ""),
            awaiting_confirm=bool(conv_data.get("awaiting_topic_confirm")),
            last_two_turns=[{"role":"user","content": ultimo_u},{"role":"assistant","content": ultima_a}],
            current_text=ultimo_u
        )
        if decision_cls.get("action") in ("greeting_smalltalk", "meta"):
            return {"ok": True, "skipped": True, "reason": "saludo_o_meta_detectado_por_llm"}

        prompt_base = get_prompt_base()
        texto_base = (conv_data.get("current_proposal") or ultimo_u or "").strip()
        msgs = build_messages_for_classify(prompt_base, texto_base, ultima_a)

        model_cls = os.getenv("OPENAI_MODEL_CLASSIFY", OPENAI_MODEL)
        out = client.chat.completions.create(
            model=model_cls,
            messages=msgs,
            temperature=0.2,
            max_tokens=300
        ).choices[0].message.content.strip()
        
        data = json.loads(out)
        categoria = data.get("categoria_general") or data.get("categoria") or "General"
        titulo    = data.get("titulo_propuesta") or data.get("titulo") or "Propuesta ciudadana"
        tono      = data.get("tono_detectado") or "neutral"
        palabras  = data.get("palabras_clave", [])

        # Fallback si el LLM devolvió algo genérico/vacío
        loc_proy = (conv_data.get("project_location")
                    or (conv_data.get("contact_info") or {}).get("barrio"))

        titulo_generico = (not titulo or len(titulo.strip()) < 6 or
                        _normalize_text(titulo) in {
                            "propuesta ciudadana",
                            "interaccion inicial",
                            "interacción inicial",
                            "solicitud de propuesta",
                            "sin titulo",
                            "—"
                        })

        categoria_generica = _normalize_text(categoria) in {
            "general", "otras", "sin clasificar", "otro"
        }

        if titulo_generico or categoria_generica:
            fb_cat, fb_tit = fallback_category_and_title(
                (conv_data.get("current_proposal") or ultimo_u or "").strip(),
                loc_proy
            )
            if fb_cat: categoria = fb_cat
            if fb_tit: titulo = fb_tit

        awaiting = bool(conv_data.get("awaiting_topic_confirm"))
        ya_contado = bool(conv_data.get("panel_contabilizado"))

        if body.contabilizar is None:
            debe_contar = (not awaiting) and (not ya_contado)
        else:
            debe_contar = bool(body.contabilizar)

        conv_ref = db.collection("conversaciones").document(chat_id)

        arr_cat = (conv_data.get("categoria_general") or [])
        arr_tit = (conv_data.get("titulo_propuesta") or [])
        tono_bd = conv_data.get("tono_detectado")

        if isinstance(conv_data.get("categoria_general"), str):
            conv_ref.set({"categoria_general": [conv_data["categoria_general"]]}, merge=True)
            arr_cat = [conv_data["categoria_general"]]
        if isinstance(conv_data.get("titulo_propuesta"), str):
            conv_ref.set({"titulo_propuesta": [conv_data["titulo_propuesta"]]}, merge=True)
            arr_tit = [conv_data["titulo_propuesta"]]

        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").strip().lower())

        cat_in_arr = _norm(categoria) in {_norm(c) for c in arr_cat}
        tit_in_arr = _norm(titulo)    in {_norm(t) for t in arr_tit}

        updates = {"ultima_fecha": firestore.SERVER_TIMESTAMP}
        if not tono_bd and tono:
            updates["tono_detectado"] = tono

        permitir_append = (not awaiting) or (body.contabilizar is True)
        updates["categoria_general"] = [categoria] if categoria else []
        updates["titulo_propuesta"] = [titulo] if titulo else []

        conv_ref.set(updates, merge=True)   # <-- ¡Esto escribe categoría y título!


        hist_last = (conv_data.get("topics_history") or [])
        last_item = hist_last[-1] if hist_last else {}
        should_append = (not last_item or last_item.get("categoria") != categoria or
                         last_item.get("titulo") != titulo or last_item.get("tono") != tono)

        if should_append and permitir_append:
            conv_ref.set({
                "topics_history": firestore.ArrayUnion([{
                    "categoria": categoria,
                    "titulo": titulo,
                    "tono": tono,
                    "fecha": firestore.SERVER_TIMESTAMP
                }])
            }, merge=True)

        db.collection("categorias_tematicas").document(categoria).set({"nombre": categoria}, merge=True)

        usuario_id = conv_data.get("usuario_id", chat_id)
        if debe_contar:
            update_panel_resumen(categoria, tono, titulo, usuario_id)
            conv_ref.set({"panel_contabilizado": True}, merge=True)

        faq = conv_data.get("faq_origen")
        if faq:
            db.collection("faq_logs").add({
                "faq_id": faq,
                "usuario_id": usuario_id,
                "chat_id": chat_id,
                "fecha": firestore.SERVER_TIMESTAMP
            })

        return {"ok": True, "clasificacion": {
            "categoria_general": categoria,
            "titulo_propuesta": titulo,
            "tono_detectado": tono,
            "palabras_clave": palabras,
            "contabilizado_en_panel": bool(debe_contar)
        }}

    except Exception as e:
        return {"ok": False, "error": str(e)}

# =========================================================
#  Arranque local
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)