# ============================
#  WilderBot API (FastAPI)
#  - /responder : RAG + historial + guardado en tu BD
#  - /clasificar: clasifica la conversación y actualiza panel
#  Correcciones:
#   * Gating fuerte de clasificación
#   * Continuidad de tema basada en historial (última pregunta abierta del bot)
#   * Reconocimiento de datos sueltos SOLO si no hay verbos de acción ni argumento
#   * Mensajes de “ack + continuidad” contextuales (no plantilla fija)
#   * Normalización de chat_id en ambos endpoints
#   * **Primero se argumenta y luego, recién, se piden datos**
#   * Guardia dura para impedir pedir contacto en el primer turno
#   * Stripper ampliado para borrar frases de “pásame/darme/bríndame” datos
#   * **Clasificación en-línea** tras la argumentación para guardar título/categoría/tono
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
#  Esquemas
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
#  Helpers de canal / IDs
# =========================================================

def normalize_chat_id(canal: Optional[str], chat_id: Optional[str]) -> str:
    """
    Prefijos por canal para evitar colisiones:
    - whatsapp -> wa_<id>
    - telegram -> tg_<id>
    - web/otros -> como venga o web_<hex> si falta
    """
    if chat_id:
        if (canal or "").lower() == "whatsapp":
            return chat_id if chat_id.startswith("wa_") else f"wa_{chat_id}"
        if (canal or "").lower() == "telegram":
            return chat_id if chat_id.startswith("tg_") else f"tg_{chat_id}"
        return chat_id
    return f"web_{os.urandom(4).hex()}"

def resolve_existing_conversation_id(raw_id: str) -> str:
    """
    Dado un id sin prefijo, intenta resolver el doc existente:
    raw, wa_raw, tg_raw (en ese orden).
    """
    candidates = [raw_id, f"wa_{raw_id}", f"tg_{raw_id}"]
    for cid in candidates:
        if db.collection("conversaciones").document(cid).get().exists:
            return cid
    return raw_id

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


def ensure_conversacion(
    chat_id: str,
    usuario_id: str,
    faq_origen: Optional[str],
    canal: Optional[str],
):
    conv_ref = db.collection("conversaciones").document(chat_id)
    snap = conv_ref.get()
    canal_val = (canal or "web")

    if not snap.exists:
        conv_ref.set({
            "usuario_id": usuario_id,
            "faq_origen": faq_origen or None,
            "categoria_general": [],
            "titulo_propuesta": [],
            "mensajes": [],
            "fecha_inicio": firestore.SERVER_TIMESTAMP,
            "ultima_fecha": firestore.SERVER_TIMESTAMP,
            "tono_detectado": None,

            # Estado de tema
            "last_topic_vec": None,
            "last_topic_summary": None,
            "awaiting_topic_confirm": False,
            "candidate_new_topic_summary": None,
            "candidate_new_topic_vec": None,
            "topics_history": [],

            # Flujo argumento + contacto
            "argument_requested": False,
            "argument_collected": False,
            "contact_intent": None,
            "contact_requested": False,
            "contact_collected": False,
            "contact_refused": False,
            "contact_info": {"nombre": None, "barrio": None, "telefono": None},

            # Lugar de la propuesta
            "project_location": None,

            # Canal
            "canal": canal_val,
        })
    else:
        data = snap.to_dict() or {}
        updates = {"ultima_fecha": firestore.SERVER_TIMESTAMP}
        if "canal" not in data and canal_val:
            updates["canal"] = canal_val
        conv_ref.update(updates)
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
    t = _normalize_text(t)
    keys = [
        "porque", "ya que", "debido", "por que", "porqué",
        "afecta", "impacta", "riesgo", "peligro",
        "falta", "no hay", "urge", "es necesario", "contamina",
        "seguridad", "salud", "empleo", "movilidad", "ambiental"
    ]
    return any(k in t for k in keys)

# =========================================================
#  Extracciones específicas (contacto/detalles)
# =========================================================

def extract_user_name(text: str) -> Optional[str]:
    m = re.search(r'\b(?:soy|me llamo|mi nombre es)\s+([A-Za-zÁÉÍÓÚÑáéíóúñ ]{2,40})', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
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

def extract_any_barrio(text: str) -> Optional[str]:
    """
    Detecta 'en el barrio X' o 'barrio X' como dato suelto (para ubicación del tema/proyecto).
    No confunde con 'vivo/resido/soy del'.
    """
    if re.search(r'\b(vivo|resido|soy del|mi barrio)\b', text, flags=re.IGNORECASE):
        return None  # residencia -> lo maneja extract_user_barrio
    m = re.search(r'\b(?:en\s+el\s+)?barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50})', text, flags=re.IGNORECASE)
    return m.group(1).strip(" .,") if m else None

def extract_project_location(text: str) -> Optional[str]:
    m = re.search(r'(?:parque|colegio|cancha|centro|obra|hospital|biblioteca|sendero|paradero|jardin)\s+(?:en|para)\s+el\s+barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50})', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" .,")
    if re.search(r'\b(construir|hacer|instalar|crear|mejorar)\b', text, flags=re.IGNORECASE):
        m = re.search(r'\ben el barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50})', text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" .,")
    return None

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
#  Heurísticas de continuidad/argumento
# =========================================================

SHORT_ACKS = {"si","sí","no","ok","okay","vale","listo","dale","de acuerdo","tal vez","quizas","quizás","puede ser","gracias"}

def is_short_ack(text: str) -> bool:
    t = _normalize_text(text)
    return t in SHORT_ACKS or (len(t) <= 6 and t in {"si","sí","no","ok"})

ACTION_STEMS = ("propon","mejor","arregl","instal","crear","hacer","quiero","necesit","hay ","repar","constru","gestionar","solic","denunc","impuls","apoy","proyecto","problema","idea")

def is_data_only_reply(text: str) -> bool:
    """Dato suelto (nombre/teléfono/barrio) SIN verbos de acción ni señales de argumento."""
    t = _normalize_text(text)
    has_datum = bool(extract_phone(text) or extract_user_barrio(text) or extract_any_barrio(text) or extract_user_name(text))
    if not has_datum:
        return False
    if has_argument_text(text):
        return False
    if any(st in t for st in ACTION_STEMS):
        return False
    return len(t) <= 80

def intent_heuristic(text: str) -> str:
    t = _normalize_text(text)
    # Propuesta (verbos/expresiones de acción)
    if "propon" in t or "propuesta" in t or (
        any(k in t for k in ("me gustaria","quiero","solicito","pido"))
        and any(a in t for a in ("mejor","instal","arregl","constru","crear","repar","alumbr","luz","seguridad","parque","colegio","via"))
    ):
        return "propuesta"
    # Problema
    if any(p in t for p in ("problema","denuncia","queja","no hay","sin luz","apagado","hueco","dañado","inseguridad","basura","fuga")):
        return "problema"
    # Reconocimiento
    if any(r in t for r in ("gracias","felicit","bien hecho","buen trabajo","apoyo")):
        return "reconocimiento"
    return "otro"

def last_assistant_was_open_question(hist: List[Dict[str,str]]) -> Tuple[bool, str]:
    """Mira el último mensaje del asistente y decide si fue pregunta abierta."""
    last_assistant = ""
    for m in reversed(hist or []):
        if m.get("role") == "assistant":
            last_assistant = m.get("content","")
            break
    if not last_assistant:
        return False, ""
    t = _normalize_text(last_assistant)
    has_qmark = "?" in last_assistant
    cues = ("cuentame","contame","podrias","podrías","me cuentas","ampliar","detalles",
            "como puedo ayudarte","¿como puedo ayudarte","que necesitas","que idea","que quisieras",
            "sobre tu idea","sobre el tema","sobre tu propuesta","impacto","por que","porque")
    is_open = has_qmark and any(c in t for c in cues)
    return is_open, last_assistant

def assistant_asked_argument_recently(hist: List[Dict[str,str]]) -> bool:
    """¿Hay alguna pregunta del asistente pidiendo motivo/impacto/detalles en los últimos turnos?"""
    for m in reversed(hist[-6:]):
        if m.get("role") != "assistant":
            continue
        t = _normalize_text(m.get("content",""))
        if "?" in m.get("content","") and any(k in t for k in ("por que","porque","impacto","cuentame","detalles","ampliar","explicame","mas sobre","en que consiste","que sucede","sobre tu idea","sobre tu propuesta")):
            return True
    return False

def craft_ack_continue(topic_hint: str, found: Dict[str,str]) -> str:
    """Mensaje corto y contextual según qué dato recibimos."""
    parts = []
    if found.get("name"):
        parts.append(f"Gracias, {found['name']}.")
    if found.get("barrio_proj"):
        parts.append(f"Anoté el barrio {found['barrio_proj']}.")
    if found.get("barrio_res"):
        parts.append(f"Vives en el barrio {found['barrio_res']}.")
    if found.get("phone"):
        parts.append("Recibí tu número.")
    topic_txt = f" Sigamos con {topic_hint.lower()}:" if topic_hint else ""
    ask = " ¿Podrías contarme un poco más sobre tu idea o necesidad?"
    return (" ".join(parts) + topic_txt + ask).strip()


# --- POV corto (1 frase de apoyo según el tema) ---
def short_supportive_pov(topic_text: str) -> str:
    t = _normalize_text(topic_text or "")
    rules = [
        (("alumbr","luz","ilumin"), "Mejorar el alumbrado público refuerza la seguridad y la vida en comunidad."),
        (("via","vial","hueco","bache","paviment"), "Arreglar las vías mejora la movilidad y reduce accidentes."),
        (("parque","zona verde","jardin"), "Cuidar los parques crea espacios seguros para la convivencia y el deporte."),
        (("colegio","escuela","educacion","educación"), "Fortalecer la educación abre oportunidades para las familias."),
        (("salud","hospital","eps","centro de salud"), "Una mejor atención en salud protege el bienestar de todos."),
        (("seguridad","atraco","hurto"), "Fortalecer la seguridad permite vivir y trabajar tranquilos."),
        (("basura","residuo","aseo","limpieza"), "Una buena gestión de residuos cuida el ambiente y la salud."),
        (("agua","acueducto","alcantarill","fuga"), "Garantizar agua y saneamiento es esencial para la salud."),
        (("transporte","bus","ruta","movilidad"), "Un transporte eficiente conecta oportunidades y ahorra tiempo."),
        (("empleo","trabajo","emprend"), "Impulsar empleo y emprendimiento dinamiza la economía local."),
    ]
    for keys, phrase in rules:
        if any(k in t for k in keys):
            return phrase
    return "Trabajar en este tema puede mejorar la calidad de vida de la comunidad."


def craft_opinion_and_contact(name: Optional[str], project_location: Optional[str], user_text: str, missing_keys: List[str]) -> str:
    """Arma: opinión corta contextual + pedido de contacto con solo lo que falte."""
    saludo = f"Entiendo, {name}." if name else "Entiendo."
    loc = f" En el barrio {project_location}." if project_location else ""
    pov = " " + short_supportive_pov(user_text)
    pedido = " " + build_contact_request(missing_keys)
    return (saludo + loc + pov + pedido).strip()
# =========================================================
#  Pequeñas “LLM tools”
# =========================================================

def llm_decide_turn(last_topic_summary: str, awaiting_confirm: bool, last_two_turns: List[Dict[str, str]], current_text: str, model: Optional[str] = None) -> Dict[str, Any]:
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

def craft_argument_question(name: Optional[str], project_location: Optional[str], topic_hint: Optional[str] = None) -> str:
    # sin re-saludar; agradece y da POV breve
    saludo = f"Gracias, {name}." if name else "Gracias por contarlo."
    lugar  = f" En el barrio {project_location}." if project_location else ""
    pov    = " " + short_supportive_pov(topic_hint or "")
    return f"{saludo}{lugar}{pov} ¿Podrías contarme por qué es importante y qué impacto tendría?".strip()


# Detectar frases de solicitud de contacto (muy amplio)
CONTACT_PATTERNS = re.compile(
    r"(compart(?:e|ir|eme|enos)|env(?:í|i)a(?:me)?|dime|ind(?:í|i)ca(?:me|nos)?|facil(?:í|i)tame|"
    r"dame|darme|me\s+das|nos\s+das|me\s+podr(?:í|i)as\s+dar|puedes\s+darme|podr(?:í|i)as\s+darme|"
    r"br(?:í|i)ndame|br(?:í|i)ndanos|me\s+brindas|p(?:á|a)same|me\s+pasas)"
    r".{0,80}(tu\s+)?(nombre|barrio|celular|tel[eé]fono|n[uú]mero|contacto)", re.IGNORECASE)

def strip_contact_requests(texto: str) -> str:
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
        # Normaliza canal e id
        canal = (data.canal or "web").lower().strip()
        chat_id = normalize_chat_id(canal, data.chat_id)

        # Ignora vacíos
        if not data.mensaje or not data.mensaje.strip():
            return {"respuesta": "", "fuentes": [], "chat_id": chat_id}

        # Usuario + conversación
        usuario_id = upsert_usuario_o_anon(chat_id, data.nombre or data.usuario, data.celular, canal)
        conv_ref = ensure_conversacion(chat_id, usuario_id, data.faq_origen, canal)

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

        # Si el último mensaje del asistente fue pregunta abierta
        was_open_q, _last_a = last_assistant_was_open_question(historial_for_decider)
        asked_argument_before = assistant_asked_argument_recently(historial_for_decider)

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

        # === EXTRAER datos del turno ===
        name = extract_user_name(data.mensaje)
        phone = extract_phone(data.mensaje)
        user_barrio = extract_user_barrio(data.mensaje)            # residencia
        any_barrio = extract_any_barrio(data.mensaje)              # ubicación del tema/proyecto
        proj_loc = extract_project_location(data.mensaje) or any_barrio

        if proj_loc and (conv_data.get("project_location") or "").strip().lower() != (proj_loc or "").lower():
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
                upsert_usuario_o_anon(chat_id, new_info.get("nombre") or data.nombre or data.usuario, phone, canal)

        # === Política argumento + contacto ===
        policy = llm_contact_policy(prev_sum or curr_sum, data.mensaje)
        intent = policy.get("intent", "otro")

        # Refuerzo determinístico para no fallar en el primer turno
        ih = intent_heuristic(data.mensaje)
        if ih in ("propuesta","problema","reconocimiento"):
            intent = ih

        # EXTRA: si el resumen del tema vigente sugiere propuesta/problema, úsalo
        ih_prev = intent_heuristic(prev_sum or curr_sum)
        if ih_prev in ("propuesta", "problema") and intent not in ("propuesta", "problema"):
            intent = ih_prev
        # ¿Acabamos de pedir argumento?
        last_role = historial_for_decider[-1]["role"] if historial_for_decider else None
        just_asked_argument = bool(conv_data.get("argument_requested")) and (last_role == "assistant")
        user_replied_after_arg_q = just_asked_argument and (len(_normalize_text(data.mensaje)) >= 2)

        # Marcar argumento colectado si corresponde (este mismo turno)
        argument_ready = user_replied_after_arg_q or has_argument_text(data.mensaje)
        if argument_ready and not conv_data.get("argument_collected"):
            conv_ref.update({"argument_collected": True})

        # ⚠️ PRIMERA VEZ con propuesta/problema: SIEMPRE preguntar argumento antes de cualquier cosa.
        need_argument_first = (intent in ("propuesta", "problema")
                               and not conv_data.get("argument_collected")
                               and not asked_argument_before)
        if (intent in ("propuesta","problema") and not conv_data.get("argument_requested")) or need_argument_first:
            conv_ref.update({"argument_requested": True})
            texto = craft_argument_question(name, proj_loc, data.mensaje)
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto}
            ])
            return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

        # -------- ACK + continuidad (contextual) --------
        if was_open_q and not conv_data.get("argument_collected") and (is_data_only_reply(data.mensaje) or is_short_ack(data.mensaje)):
            found = {"name": name, "phone": phone, "barrio_res": user_barrio, "barrio_proj": any_barrio}
            texto = craft_ack_continue(prev_sum or curr_sum, {k:v for k,v in found.items() if v})
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto}
            ])
            return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

        # ¿El usuario rechazó datos?
        if detect_contact_refusal(data.mensaje):
            conv_ref.update({"contact_refused": True, "contact_requested": True})

        # Flags de contacto actuales
        already_req = bool(conv_data.get("contact_requested"))
        already_col = bool(conv_data.get("contact_collected")) or bool(phone)
        contact_refused = bool(conv_data.get("contact_refused"))

        # ✅ Solo se permite pedir contacto si:
        #   1) intención propuesta/problema,
        #   2) YA se preguntó argumento EN ESTA CONVERSACIÓN (historial),
        #   3) argumento ya está recogido,
        #   4) no tenemos aún contacto y no fue rechazado.
        argument_collected_now = bool(conv_data.get("argument_collected") or argument_ready)
        allow_contact = (
            # intención a nivel de conversación (no solo el último turno)
            (intent in ("propuesta", "problema") or intent_heuristic(prev_sum or curr_sum) in ("propuesta", "problema"))
            # sabemos que el bot preguntó argumento (por historial) o marcamos que lo preguntó
            and (asked_argument_before or bool(conv_data.get("argument_requested")))
            # el usuario ya respondió algo tras esa pregunta (o se detectó razonamiento)
            and argument_collected_now
            # aún no tenemos teléfono y no lo rechazó
            and not already_col
            and not contact_refused
        )

        if allow_contact and not already_req:
            conv_ref.update({"contact_intent": intent, "contact_requested": True})

        if allow_contact and not (name or phone or user_barrio):
            info_actual = (conv_data.get("contact_info") or {})
            faltan = []
            if not info_actual.get("nombre"):   faltan.append("nombre")
            if not info_actual.get("barrio"):   faltan.append("barrio")
            if not info_actual.get("telefono"): faltan.append("celular")

            # Opinión breve + solicitud de los datos que falten (sin repetir nombre si ya lo tenemos)
            stored_name = (info_actual.get("nombre") or data.nombre or data.usuario or name)
            texto_directo = craft_opinion_and_contact(stored_name, proj_loc, data.mensaje, faltan or ["celular"])

            # --- Clasificación en-línea para guardar título/categoría/tono ---
            inline_classify_and_update(chat_id, data.mensaje, texto_directo)

            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto_directo}
            ])
            return {"respuesta": texto_directo, "fuentes": [], "chat_id": chat_id}

        if contact_refused and already_req and not already_col:
            texto = PRIVACY_REPLY
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto}
            ])
            return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

        # 4) RAG + respuesta
        hits = rag_search(data.mensaje, top_k=5)
        historial = load_historial_para_prompt(conv_ref)
        messages = build_messages(
            data.mensaje,
            [h["texto"] for h in hits],
            historial,
            topic_change_suspect=False,
            prev_summary=prev_sum,
            new_summary=curr_sum,
            intro_hint=intro_hint,
            emphasize_contact=allow_contact,              # solo si está permitido
            emphasize_argument=False,
            intent=intent,
            contact_already_requested=already_req,
            contact_already_collected=already_col,
        )

        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=280
        )
        texto = completion.choices[0].message.content.strip()

        # Si NO está permitido pedir contacto, borro cualquier frase que lo pida
        if not allow_contact:
            texto = strip_contact_requests(texto)

        # Si llegó el teléfono ahora, cierro con confirmación
        if phone:
            nombre_txt = (name or (conv_data.get("contact_info") or {}).get("nombre") or "").strip()
            cierre = (f"Gracias, {nombre_txt}. " if nombre_txt else "Gracias. ")
            cierre += "Con estos datos escalamos el caso y te contamos avances."
            if "tel" in texto.lower() or "celu" in texto.lower() or len(texto) < 30:
                texto = cierre
            else:
                texto += "\n\n" + cierre

            # Guardar clasificación aunque el contacto llegue rápido
            inline_classify_and_update(chat_id, data.mensaje, texto)

        elif allow_contact:
            info_actual = (conv_data.get("contact_info") or {})
            missing = []
            if not (info_actual.get("nombre") or name):        missing.append("nombre")
            if not (info_actual.get("barrio") or user_barrio): missing.append("barrio")
            if not (info_actual.get("telefono") or phone):     missing.append("celular")
            if missing:
                stored_name = (info_actual.get("nombre") or data.nombre or data.usuario or name)
                texto = craft_opinion_and_contact(stored_name, proj_loc, data.mensaje, missing)

                # Guardar clasificación también en esta ruta
                inline_classify_and_update(chat_id, data.mensaje, texto)

        append_mensajes(conv_ref, [
            {"role": "user", "content": data.mensaje},
            {"role": "assistant", "content": texto}
        ])

        return {"respuesta": texto, "fuentes": hits, "chat_id": chat_id}

    except Exception as e:
        return {"error": str(e)}


# =========================================================
#  Clasificación y Panel (con gating fuerte)
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

def build_messages_for_classify(prompt_base: str, u: str, a: str):
    system_msg = (
        f"{prompt_base}\n\n"
        "TAREA: Clasifica la propuesta y devuelve SOLO JSON:\n"
        '{"categoria_general":"...", "titulo_propuesta":"...", "tono_detectado":"positivo|crítico|preocupación|propositivo", "palabras_clave":["..."]}\n'
        "Reglas: título ≤ 70 caracteres."
    )
    user_msg = f"Último mensaje del ciudadano:\n{u}\n\nÚltima respuesta del bot:\n{a}\n\nDevuelve el JSON ahora."
    return [{"role":"system","content":system_msg},{"role":"user","content":user_msg}]

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

# --- Clasificación en-línea (mismo comportamiento que /clasificar) ---
def inline_classify_and_update(chat_id: str, ultimo_u: str, ultima_a: str):
    try:
        conv_ref = db.collection("conversaciones").document(chat_id)
        snap = conv_ref.get()
        if not snap.exists:
            return
        conv_data = snap.to_dict() or {}

        # Solo clasificar propuestas/problemas y cuando ya hay argumento
        sum_vigente = conv_data.get("last_topic_summary") or ultimo_u
        intent_now = intent_heuristic(sum_vigente)
        if intent_now not in ("propuesta", "problema"):
            intent_now = intent_heuristic(ultimo_u)
            if intent_now not in ("propuesta", "problema"):
                return
        if not conv_data.get("argument_collected"):
            if has_argument_text(ultimo_u):
                conv_ref.update({"argument_collected": True})
            else:
                return

        prompt_base = get_prompt_base()
        msgs = build_messages_for_classify(prompt_base, ultimo_u, ultima_a)
        model_cls = os.getenv("OPENAI_MODEL_CLASSIFY", OPENAI_MODEL)
        out = client.chat.completions.create(
            model=model_cls, messages=msgs, temperature=0.2, max_tokens=300
        ).choices[0].message.content.strip()

        data = json.loads(out)
        categoria = data.get("categoria_general") or data.get("categoria") or "General"
        titulo    = data.get("titulo_propuesta") or data.get("titulo") or "Propuesta ciudadana"
        tono      = data.get("tono_detectado") or "neutral"

        awaiting = bool(conv_data.get("awaiting_topic_confirm"))
        ya_contado = bool(conv_data.get("panel_contabilizado"))
        debe_contar = not awaiting and not ya_contado

        # Normalización arrays y de-duplicación
        arr_cat = conv_data.get("categoria_general") or []
        arr_tit = conv_data.get("titulo_propuesta") or []
        if isinstance(arr_cat, str): arr_cat = [arr_cat]
        if isinstance(arr_tit, str): arr_tit = [arr_tit]

        def _norm(s: str) -> str: return re.sub(r"\s+", " ", (s or "").strip().lower())
        cat_in = _norm(categoria) in {_norm(c) for c in arr_cat}
        tit_in = _norm(titulo)    in {_norm(t) for t in arr_tit}

        updates = {"ultima_fecha": firestore.SERVER_TIMESTAMP}
        if not (conv_data.get("tono_detectado")) and tono:
            updates["tono_detectado"] = tono

        permitir_append = (not awaiting)
        if permitir_append:
            if categoria and not cat_in:
                updates["categoria_general"] = firestore.ArrayUnion([categoria])
            if titulo and not tit_in:
                updates["titulo_propuesta"] = firestore.ArrayUnion([titulo])

        conv_ref.set(updates, merge=True)

        # topics_history
        hist_last = (conv_data.get("topics_history") or [])
        last_item = hist_last[-1] if hist_last else {}
        should_append = (not last_item or last_item.get("categoria") != categoria
                         or last_item.get("titulo") != titulo or last_item.get("tono") != tono)
        if should_append and permitir_append:
            conv_ref.set({"topics_history": firestore.ArrayUnion([{
                "categoria": categoria, "titulo": titulo, "tono": tono,
                "fecha": firestore.SERVER_TIMESTAMP
            }])}, merge=True)

        # catálogo de categorías + panel
        db.collection("categorias_tematicas").document(categoria).set({"nombre": categoria}, merge=True)
        if debe_contar:
            update_panel_resumen(categoria, tono, titulo, conv_data.get("usuario_id", chat_id))
            conv_ref.set({"panel_contabilizado": True}, merge=True)

    except Exception:
        # no romper la conversación si algo falla al clasificar
        pass

@app.post("/clasificar")
async def clasificar(body: ClasificarIn):
    try:
        # Resolver id (con o sin prefijo)
        chat_id_raw = body.chat_id
        chat_id = resolve_existing_conversation_id(chat_id_raw)

        ultimo_u, ultima_a, conv_data = read_last_user_and_bot(chat_id)
        if not ultimo_u:
            return {"ok": False, "mensaje": "No hay mensajes para clasificar."}

        # Gating por intención y argumento
        intent_now = llm_contact_policy(conv_data.get("last_topic_summary") or ultimo_u, ultimo_u).get("intent", "otro")
        if intent_now not in ("propuesta", "problema"):
            return {"ok": True, "skipped": True, "reason": "sin_intencion_de_propuesta_o_problema"}

        if not conv_data.get("argument_collected"):
            return {"ok": True, "skipped": True, "reason": "sin_argumento_recogido"}

        decision_cls = llm_decide_turn(
            last_topic_summary=(conv_data.get("last_topic_summary") or ""),
            awaiting_confirm=bool(conv_data.get("awaiting_topic_confirm")),
            last_two_turns=[{"role":"user","content": ultimo_u},{"role":"assistant","content": ultima_a}],
            current_text=ultimo_u
        )
        if decision_cls.get("action") in ("greeting_smalltalk", "meta"):
            return {"ok": True, "skipped": True, "reason": "saludo_o_meta_detectado_por_llm"}

        prompt_base = get_prompt_base()
        msgs = build_messages_for_classify(prompt_base, ultimo_u, ultima_a)

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
        if permitir_append:
            if not cat_in_arr and categoria:
                updates["categoria_general"] = firestore.ArrayUnion([categoria])
            if not tit_in_arr and titulo:
                updates["titulo_propuesta"] = firestore.ArrayUnion([titulo])

        conv_ref.set(updates, merge=True)

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
