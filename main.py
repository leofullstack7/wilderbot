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
OPENAI_MODEL_SUMMARY = os.getenv("OPENAI_MODEL_SUMMARY", "gpt-4o-mini")


BOT_INTRO_TEXT = os.getenv(
    "BOT_INTRO_TEXT",
    "¡Hola! Soy la mano derecha de Wilder Escobar. Estoy aquí para escuchar y canalizar tus "
    "problemas, propuestas o reconocimientos. ¿Qué te gustaría contarme hoy?"
)

# ---- CONSULTAS: títulos permitidos ----
CONSULTA_TITULOS = [
    "Vida Personal Wilder", "General", "Leyes", "Movilidad",
    "Educación", "Salud", "Seguridad", "Vivienda", "Empleo"
]

# --- Heurística mínima para detectar "consulta" ---
def _is_consulta_heuristic(t: str) -> bool:
    tx = _normalize_text(t)
    if is_proposal_intent(tx):
        return False
    if "?" in t or "¿" in t:
        return True
    q_words = (
        "me gustaria saber", "quisiera saber", "hay alguna ley", "que ley", "qué ley",
        "que normas", "cual es la ley", "cuando", "cuándo", "donde", "dónde",
        "por que", "por qué", "wilder apoya"
    )
    return any(k in tx for k in q_words)

def _pick_consulta_title(t: str) -> str:
    tx = _normalize_text(t)
    if any(k in tx for k in ("ley", "norma", "congreso", "proyecto de ley", "codigo penal", "delito", "embriaguez", "homicidio")):
        return "Leyes"
    if any(k in tx for k in ("salud", "eps", "hospital", "puesto de salud", "adulto mayor", "adulta mayor")):
        return "Salud"
    if any(k in tx for k in ("seguridad", "atraco", "robo", "delincuencia")):
        return "Seguridad"
    if any(k in tx for k in ("movilidad", "transito", "tránsito", "trafico", "tráfico", "bus", "ruta", "via", "semaforo", "semáforo")):
        return "Movilidad"
    if any(k in tx for k in ("vivienda", "arriendo", "subsidio de vivienda", "habitat", "hábitat")):
        return "Vivienda"
    if any(k in tx for k in ("empleo", "trabajo", "oferta laboral", "empleabilidad")):
        return "Empleo"
    if any(k in tx for k in ("colegio", "escuela", "educacion", "educación", "universidad", "beca", "icetex")):
        return "Educación"
    if any(k in tx for k in ("wilder", "familia de wilder", "hijo de wilder", "esposa de wilder")):
        return "Vida Personal Wilder"
    return "General"


# ---- CONSULTAS: heurística rápida y título por palabras clave ----
CONSULTA_KWS = {
    "Leyes": [
        "ley", "proyecto de ley", "norma", "codigo penal", "congreso",
        "apoya", "impulsa", "vota", "tramite", "regimen", "sancion", "delito"
    ],
    "Salud": [
        "salud", "hospital", "puesto de salud", "eps", "ambulancia",
        "adulto mayor", "covid", "medicina", "vacuna"
    ],
    "Movilidad": [
        "movilidad", "transporte", "bus", "metro", "via", "vía", "trancón",
        "trafico", "tráfico", "semaforo", "bici", "peatonal"
    ],
    "Seguridad": [
        "seguridad", "policia", "policía", "atraco", "hurto", "homicidio",
        "violencia", "microtrafico", "drogas"
    ],
    "Educación": [
        "educacion", "educación", "colegio", "escuela", "universidad",
        "beca", "icetex", "cupos"
    ],
    "Vivienda": [
        "vivienda", "casa", "mejoramiento", "titular", "titulación",
        "arriendo", "subsidio", "invasion", "invasión"
    ],
    "Empleo": [
        "empleo", "trabajo", "formalizacion", "formalización",
        "emprendimiento", "empresa"
    ],
    "Vida Personal Wilder": [
        "wilder", "escobar", "biografia", "biografía", "vida personal",
        "quien es wilder", "quién es wilder", "familia de wilder"
    ],
}

INTERROGATIVOS = [
    "que", "qué", "como", "cómo", "cuando", "cuándo", "donde", "dónde",
    "por que", "por qué", "cual", "cuál", "quien", "quién",
    "me gustaria saber", "quisiera saber", "podria decirme",
    "puedes explicarme", "informacion", "información"
]

def heuristic_consulta_title(text: str) -> Optional[str]:
    """
    Devuelve un título de CONSULTA (de tu lista) si el texto parece pregunta;
    None si no parece consulta.
    """
    t = _normalize_text(text)

    # Debe parecer pregunta/consulta
    looks_question = (
        ("?" in text) or
        any(p in t for p in INTERROGATIVOS) or
        t.startswith("me gustaria saber") or
        "saber si" in t
    )
    if not looks_question:
        return None

    # No debe ser propuesta
    if is_proposal_intent(text):
        return None

    # Buscar la categoría más evidente por keywords
    for titulo, kws in CONSULTA_KWS.items():
        for kw in kws:
            if _normalize_text(kw) in t:
                return titulo

    # Si es pregunta pero no calza en ninguna lista, va a "General"
    return "General"

# === Clientes ===
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

print(f"[RAG] Pinecone index en uso: {PINECONE_INDEX}")
print(f"[RAG] Modelo de embeddings: {EMBEDDING_MODEL}")


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
                "barrio": barrio or None,                    # <-- guarda barrio
                "fecha_registro": firestore.SERVER_TIMESTAMP,
                "chats": [chat_id],
                "canal": canal or "web",
            })
        else:
            prev = doc.to_dict() or {}
            ref.update({
                "nombre": nombre or prev.get("nombre", ""),
                "telefono": telefono,
                "barrio": barrio or prev.get("barrio"),       # <-- actualiza barrio si viene
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

            # <<< PUNTO 17: flag para no repetir la nota >>>
            "location_note_sent": False,
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

    # --- Recalcular y guardar resumen corto (≤100 chars) ---
    try:
        resumen = summarize_conversation_brief(arr, max_chars=100)
        conv_ref.update({"resumen": resumen, "resumen_updated_at": firestore.SERVER_TIMESTAMP})
    except Exception:
        # no rompas el flujo si el resumen falla
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

# Palabras/frases iniciales que NO son nombre (normalizadas)
DISCOURSE_START_WORDS = {
    _normalize_text(w) for w in [
        "hola", "holaa", "holaaa",
        "buenas", "buenos días", "buen día", "saludos",
        "gracias", "ok", "okay", "oki", "vale", "de acuerdo",
        "listo", "listos", "bueno", "entendido", "hecho",
        "claro", "claro que sí",
        "sí", "si", "perfecto", "hey",
        "dale", "de una", "genial", "súper", "super"
    ]
}


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


def has_argument_text(t: str) -> bool:
    """
    Heurística RÁPIDA para detectar argumentos obvios.
    Si no está seguro, devuelve True para que llm_is_argument valide.
    """
    t = _normalize_text(t)
    
    # Palabras clave muy claras (detección rápida)
    obvious_keys = [
        "porque", "ya que", "debido", "necesitamos", "es importante",
        "para que", "con el fin", "urgente", "peligro"
    ]
    
    if any(k in t for k in obvious_keys):
        return True
    
    # Si es respuesta mediana/larga (probablemente está argumentando)
    if len(t) >= 30:
        return True
    
    return False


def llm_is_argument(text: str, proposal_context: str = "") -> bool:
    """
    Usa LLM para detectar si el texto es un argumento/justificación.
    Más inteligente que keywords.
    """
    sys = (
        "Eres un clasificador que detecta si un mensaje es un ARGUMENTO/JUSTIFICACIÓN.\n"
        "Un argumento explica:\n"
        "- POR QUÉ algo es importante\n"
        "- A QUIÉN beneficia\n"
        "- QUÉ PROBLEMA resuelve\n"
        "- CONSECUENCIAS de no hacerlo\n\n"
        "Responde SOLO 'SI' o 'NO'."
    )
    
    usr = f"Contexto (propuesta): {proposal_context}\n\nMensaje del usuario: {text}\n\n¿Es un argumento/justificación? (SI/NO)"
    
    try:
        out = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            temperature=0.0,
            max_tokens=10
        ).choices[0].message.content.strip().upper()
        
        return "SI" in out or "YES" in out
    except:
        # Si falla el LLM, usa la heurística
        return has_argument_text(text)

    

# === NUEVO: heurística fuerte para detectar intención de propuesta/sugerencia ===
def is_proposal_intent(text: str) -> bool:
    t = _normalize_text(text)

    # Si es "me gustaría que me dijeran/explicaran/informaran", NO es propuesta (suele ser consulta)
    if "me gustaria que" in t and re.search(r"\b(me\s+dig[ae]n|me\s+expliquen?|me\s+informen?)\b", t):
        return False

    kw = [
        "propongo", "propuesta", "sugerencia", "sugerir", "sugiero",
        "mi idea", "mi propuesta", "quiero proponer", "quisiera proponer",
        "me gustaria proponer", "planteo", "plantear",
        "propongo que", "propuse", "propone"
    ]
    return any(k in t for k in kw)

def is_question_like(text: str) -> bool:
    t = _normalize_text(text)
    return (("?" in text) or any(k in t for k in INTERROGATIVOS)) and not is_proposal_intent(text)

def is_proposal_denial(text: str) -> bool:
    t = _normalize_text(text)
    pats = [
        r'\b(aun|aún|todavia|todavía)\s+no\b.*\b(propuest|idea|sugerenc)',
        r'\b(no\s+tengo|no\s+he\s+hecho|no\s+te\s+he\s+hecho)\b.*\b(propuest|idea|sugerenc)',
        r'\b(no\s+es)\s+una\s+(propuesta|idea|sugerencia)\b',
        r'\b(olvidalo|olvídalo|mejor\s+no|ya\s+no|mas\s+tarde|más\s+tarde)\b'
    ]
    if any(re.search(p, t) for p in pats):
        return True
    return False  # barato y suficiente para el parche

def looks_like_proposal_content(text: str) -> bool:
    if is_proposal_denial(text):
        return False

    raw = _normalize_text(text)
    # Intención pura sin contenido (ej.: "quisiera/quiero/me gustaría (hacer) una propuesta/idea")
    if re.match(
        r'^(?:hola|holaa|buenas|buenos dias|buenas tardes|buenas noches|como estas|que mas|q mas|saludos)?\s*'
        r'(?:quiero|quisiera|me gustar(?:ia|ía))\s+(?:hacer\s+)?(?:una\s+)?(?:propuesta|idea|sugerencia)\s*[.!]?\s*$',
        raw
    ):
        return False

    t = _normalize_text(extract_proposal_text(text))
    if not t or t in {"algo","una idea","una propuesta","un tema","varias cosas"}:
        return False

    # "quiero/quisiera proponer" sin complemento
    if re.match(r'^(?:quiero|quisiera|me gustar(?:ia|ía))\s+proponer\.?$', t):
        return False

    # Señales de contenido real (verbos/nombres típicos)
    if re.search(r'\b(arregl|mejor|constru|instal|crear|paviment|ilumin|señaliz|ampli|dotar|regular(?!idad|mente)|prohib|mult|beca|subsid|limpi|recog|camar|pint|adecu)\w*', t):
        return True
    if re.search(r'\b(parque|anden|and[eé]n|semaforo|luminaria|cancha|juegos|polideportivo|colegio|hospital|bus|ruta|acera)\b', t):
        return True

    # Último recurso: acepta longitud solo si NO es una frase de intención
    return len(t) >= 20 and not re.match(r'^(?:como estas\s+)?(?:quiero|quisiera|me gustar(?:ia|ía))\b', t)



# === NUEVO: recortar a N oraciones (para contener respuestas del LLM) ===
def limit_sentences(text: str, max_sentences: int = 3) -> str:
    parts = re.split(r'(?<=[\.\?!])\s+', text.strip())
    out = " ".join([p for p in parts if p][:max_sentences]).strip()
    return out or text


def _clamp_summary(s: str, limit: int = 100) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if len(s) <= limit else s[:limit].rstrip()

def summarize_conversation_brief(mensajes: List[Dict[str, str]], max_chars: int = 100) -> str:
    """
    Resume TODA la conversación (usuario/bot) en ≤100 caracteres (incluyendo espacios).
    Sin datos personales. Enfocado en la petición/tema y, si aplica, el barrio.
    """
    # Construye una transcripción compacta (hasta 40 turnos más recientes para no gastar tokens de más)
    parts = []
    for m in mensajes[-40:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # Oculta números de teléfono u otros números largos
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
        # Fallback ultra simple si hubiera un error con el LLM
        out = (mensajes[0].get("content") if mensajes else "")[:max_chars]

    return _clamp_summary(out, max_chars)


# =========================================================
#  Extracciones específicas
# =========================================================

def extract_user_name(text: str) -> Optional[str]:
    # 1) Formas explícitas
    m = re.search(r'\b(?:soy|me llamo|mi nombre es)\s+([A-Za-zÁÉÍÓÚÑáéíóúñ ]{2,40})', text, flags=re.IGNORECASE)
    if m:
        nombre = m.group(1).strip(" .,")
        return nombre if _normalize_text(nombre) not in DISCOURSE_START_WORDS else None

    # 2) Nombre suelto al inicio
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

    # 3) Conectores tipo “Claro/Gracias/Ok … [Nombre] … vivo/soy/mi número…”
    m = re.search(
        r'(?:^|[,;]\s*)(?:[A-Za-zÁÉÍÓÚÑáéíóúñ ]{0,20})?(?:claro(?:\s+que\s+s[ií])?|gracias|vale|ok|okay|perfecto|listo|de acuerdo)\s*,?\s*'
        r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})\s*,?\s*'
        r'(?:vivo|resido|mi\s+(?:n[uú]mero|tel[eé]fono|celular)|soy)\b',
        text, flags=re.IGNORECASE
    )
    if m:
        return m.group(1).strip(" .,")

    # 4) Fallback: “Nombre Apellido, vivo/resido/mi número/soy…”
    m = re.search(
        r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})\s*,?\s*(?:vivo|resido|mi\s+(?:n[uú]mero|tel[eé]fono|celular)|soy)\b',
        text, flags=re.IGNORECASE
    )
    if m:
        posible = m.group(1).strip(" .,")
        return posible if _normalize_text(posible) not in DISCOURSE_START_WORDS else None

    return None



def extract_phone(text: str) -> Optional[str]:
    # permite capturar con prefijos largos
    m = re.search(r'(\+?\d[\d\s\-]{7,16}\d)', text)
    if not m:
        return None
    tel = re.sub(r'\D', '', m.group(1))

    # Normaliza prefijos de Colombia (57) con o sin 00/+
    # quita 0057 o 57 al inicio si eso deja un número razonable (8–12 dígitos)
    tel = re.sub(r'^(?:00)?57', '', tel)

    return tel if 8 <= len(tel) <= 12 else None

def extract_user_barrio(text: str) -> Optional[str]:
    """
    Extrae barrio de RESIDENCIA. Acepta:
      - "vivo en Aranjuez ..."
      - "vivo en el barrio Aranjuez ..."
      - "resido en Aranjuez ..."
      - "soy del barrio Aranjuez ..."
      - "mi barrio es Aranjuez"
    """
    # 1) "vivo/resido en (el barrio)? X"
    m = re.search(
        r'\b(?:vivo|resido)\s+en\s+(?:el\s+)?(?:barrio\s+)?'
        r'([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{1,49}?)'
        r'(?=(?:\s+(?:y|mi|n[uú]mero|tel[eé]fono|celular|desde|de|del|con|para|por)\b|[,.;]|$))',
        text, flags=re.IGNORECASE
    )
    if m:
        return _clean_barrio_fragment(m.group(1))

    # 2) "soy del barrio X"
    m = re.search(r'\bsoy\s+del\s+barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50})', text, flags=re.IGNORECASE)
    if m:
        return _clean_barrio_fragment(m.group(1))

    # 3) "mi barrio es X"  o "mi barrio X"
    m = re.search(r'\bmi\s+barrio\s+(?:es\s+)?([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50})', text, flags=re.IGNORECASE)
    if m:
        return _clean_barrio_fragment(m.group(1))

    return None


def llm_extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """
    Usa LLM para extraer nombre, barrio y teléfono cuando el usuario
    los da todos juntos sin formato claro.
    """
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

def _titlecase(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).title()

def _clean_barrio_fragment(s: str) -> str:
    # corta en conectores típicos: para/por/que/donde/con/coma/punto/fin
    s = re.split(r"\s+(?:para|por|que|donde|con)\b|[,.;]|$", s, maxsplit=1, flags=re.IGNORECASE)[0]
    return _titlecase(s)

def extract_project_location(text: str) -> Optional[str]:
    # 1) "en el barrio X" (evita residencia)
    m = re.search(
        r'\b(?:en\s+el\s+|en\s+)barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50}?)(?=(?:\s+(?:para|por|que|donde|con|cerca|y)\b|[,.;]|$))',
        text, flags=re.IGNORECASE
    )
    if m:
        left = text[:m.start()].lower()
        # si viene de "vivo/resido en el barrio", NO es ubicación del proyecto
        if re.search(r'(vivo|resido)\s+en\s*$', left[-25:]):
            pass
        else:
            return _clean_barrio_fragment(m.group(1))

    # 2) "del barrio X" / "de el barrio X"
    m = re.search(
        r'\b(?:del\s+|de\s+el\s+)barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50}?)(?=(?:\s+(?:cerca|para|por|que|donde|con|y)\b|[,.;]|$))',
        text, flags=re.IGNORECASE
    )
    if m:
        return _clean_barrio_fragment(m.group(1))

    # 3) si hay verbo de acción, acepta "barrio X" a secas
    if re.search(r'\b(construir|hacer|instalar|crear|mejorar|arreglar|reparar|pintar|adecuar|señalizar)\b', text, flags=re.IGNORECASE):
        m = re.search(
            r'\bbarrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50}?)(?=(?:\s+(?:cerca|para|por|que|donde|con|y)\b|[,.;]|$))',
            text, flags=re.IGNORECASE
        )
        if m:
            return _clean_barrio_fragment(m.group(1))

    return None


def extract_proposal_text(text: str) -> str:
    """
    Extrae la parte 'sustantiva' de la propuesta aunque venga precedida
    por saludos o muletillas, y aunque el trigger no esté al inicio.
    """
    t = text.strip()

    # 1) Limpia saludos/muletillas al frente (opcionales)
    t = re.sub(
        r'^\s*(?:hola|holaa+|buenas(?:\s+(?:tardes|noches|dias))?|buen(?:\s*dia)?|hey|que tal|qué tal|saludos)[,!\s\-–—]*',
        '', t, flags=re.IGNORECASE
    )

    # 2) Si hay un trigger de propuesta en cualquier parte, corta desde ahí
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
        # Fallback: si dice “propuesta/idea” sin verbo, intenta después de “es/sería”
        m2 = re.search(r'(?:propuesta|idea)\s+(?:es|ser[ií]a)\s*[:\-–—]?\s*(.*)', t, flags=re.IGNORECASE)
        if m2:
            t = m2.group(1).strip()

    # 3) Normaliza espacios y limita a 2 oraciones
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


def llm_consulta_classifier(ultimo_usuario: str, historial_breve: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
    """
    Decide si es CONSULTA y asigna un título de CONSULTA_TITULOS.
    Versión mejorada que prioriza la búsqueda en documentos.
    """
    # ====================================================================
    # NUEVO: Detectar keywords que indican búsqueda de información
    # ====================================================================
    KEYWORDS_BUSQUEDA = [
        # Preguntas directas
        "qué", "que", "cómo", "como", "cuál", "cual", "cuándo", "cuando",
        "dónde", "donde", "por qué", "por que", "quién", "quien",
        
        # Solicitudes de información
        "dime sobre", "cuéntame sobre", "háblame de", "hablame de",
        "información sobre", "informacion sobre", "datos sobre",
        "me gustaría saber", "me gustaria saber", "quisiera saber",
        "quiero saber", "necesito saber",
        
        # Referencias a Wilder/leyes/documentos
        "wilder", "ley", "proyecto de ley", "norma", "congreso",
        "propuesta de wilder", "posición de wilder", "posicion de wilder",
        "qué dice wilder", "que dice wilder",
        
        # Preguntas sobre temas específicos
        "hay alguna", "existe alguna", "se puede", "es posible",
        "está permitido", "esta permitido",
    ]
    
    texto_norm = _normalize_text(ultimo_usuario)
    
    # Si tiene keyword de búsqueda Y no es claramente una propuesta, es consulta
    tiene_keyword_busqueda = any(kw in texto_norm for kw in 
                                  [_normalize_text(k) for k in KEYWORDS_BUSQUEDA])
    
    es_claramente_propuesta = (
        is_proposal_intent(ultimo_usuario) or 
        looks_like_proposal_content(ultimo_usuario)
    )
    
    # REGLA 1: Si tiene keyword de búsqueda y NO es propuesta → CONSULTA
    if tiene_keyword_busqueda and not es_claramente_propuesta:
        h_title = heuristic_consulta_title(ultimo_usuario)
        titulo = h_title if h_title else "General"
        return {"is_consulta": True, "titulo": titulo, "reason": "keyword_busqueda"}
    
    # REGLA 2: Heurística determinista primero (barata y robusta)
    h_title = heuristic_consulta_title(ultimo_usuario)
    if h_title:
        return {"is_consulta": True, "titulo": h_title, "reason": "heuristic"}

    # REGLA 3: Si la heurística no decide, pedimos al LLM (respaldo)
    historia = ""
    if historial_breve:
        historia = "\n".join([f"{m['role']}: {m['content']}" for m in historial_breve[-4:]])

    sys = (
        "Eres un clasificador muy estricto para distinguir CONSULTA (pregunta/información) vs no-consulta.\n"
        "Si el texto pide que el bot explique, informe, aclare o responda '¿qué/cómo/cuándo/dónde/por qué?', "
        "y NO sugiere una acción concreta para ejecutar (no dice 'propongo', 'me gustaría proponer', 'deberían hacer', etc.), "
        "entonces es CONSULTA.\n"
        f"El título DEBE ser uno de: {CONSULTA_TITULOS}.\n"
        "Devuelve SOLO JSON con claves: is_consulta(bool), titulo(str o \"\"), reason(str)."
    )
    usr = (
        f"Historial breve (opcional):\n{historia}\n\n"
        f"Mensaje actual del ciudadano:\n{ultimo_usuario}\n\n"
        "JSON ahora."
    )
    out = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        temperature=0.0,
        max_tokens=160
    ).choices[0].message.content

    try:
        data = json.loads(out)
    except Exception:
        data = {"is_consulta": False, "titulo": "", "reason": "fallback"}

    # Sanitiza título
    titulo = (data.get("titulo") or "").strip()
    if data.get("is_consulta"):
        if titulo not in CONSULTA_TITULOS:
            titulo = "General"
        return {"is_consulta": True, "titulo": titulo, "reason": data.get("reason", "llm")}

    # REGLA 4: Último salvavidas: si el LLM dijo que no, reintenta con heurística "suave"
    soft_title = heuristic_consulta_title(ultimo_usuario)
    if soft_title:
        return {"is_consulta": True, "titulo": soft_title, "reason": "heuristic-soft"}
    
    # Fallback si el LLM falla o dice que no es consulta y la heurística lo ve como consulta
    if not bool(data.get("is_consulta")) and _is_consulta_heuristic(ultimo_usuario):
        titulo_fb = _pick_consulta_title(ultimo_usuario)
        return {"is_consulta": True, "titulo": titulo_fb, "reason": "heuristic_fallback"}

    return {"is_consulta": False, "titulo": "", "reason": data.get("reason", "llm_no")}


# =========================================================
#  Contact helpers y control de flujo
# =========================================================

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


# NUEVO: pedir SOLO el barrio del proyecto con un tono distinto al de contacto
def build_project_location_request() -> str:
    return (
        "Para ubicar el caso en el mapa: ¿en qué barrio sería exactamente el proyecto? "
        "Si ya lo mencionaste, recuérdamelo por favor."
    )

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
    r"(compart\w+|env(í|i)a\w*|dime|indíca\w*|facilita\w*|regálame|regalame|me\s+das|me\s+dejas).{0,60}"
    r"(tu\s+)?(nombre|barrio|celular|tel[eé]fono|n[uú]mero|contacto)",
    re.IGNORECASE
)

def strip_contact_requests(texto: str) -> str:
    # Elimina oraciones completas que pidan contacto
    sent_split = re.split(r'(?<=[\.\?!])\s+', texto.strip())
    limpio = [s for s in sent_split if not CONTACT_PATTERNS.search(s)]
    if limpio:
        out = " ".join([s for s in limpio if s]).strip()
        return out if out else texto

    # Fallback: si TODA la respuesta era un pedido de contacto, quita esa parte
    cleaned = CONTACT_PATTERNS.sub("", texto).strip()
    # Si queda muy corto/roto, devolvemos una pregunta de argumento breve
    return cleaned if len(_normalize_text(cleaned)) >= 5 else \
        "¿Nos cuentas brevemente por qué sería importante y a quién beneficiaría?"

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


# --- NUEVO: mensajes para CONSULTAS (responder con RAG, sin pedir contacto) ---
def build_consulta_messages(user_text: str, rag_snippets: List[str], historial: List[Dict[str, str]]):
    contexto = "\n".join([f"- {s}" for s in rag_snippets if s.strip()])
    sys = (
        "Responde como asistente informativo a una CONSULTA.\n"
        "Sé claro y breve (máx. 2–3 frases). No pidas datos de contacto ni pidas propuestas.\n"
        "Si el contexto no alcanza, dilo y sugiere el siguiente paso."
    )
    ctx = "Contexto recuperado:\n" + (contexto if contexto else "(sin coincidencias relevantes)")
    msgs = [{"role": "system", "content": sys}]
    if historial:
        msgs.extend(historial[-6:])
    msgs.append({"role": "user", "content": f"{ctx}\n\nPregunta:\n{user_text}"})
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
        # --- NEGACIÓN explícita de propuesta: resetea subflujo y no pidas nada más ---
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
                "ultima_fecha": firestore.SERVER_TIMESTAMP
            })
            texto = "Perfecto, sin problema. Cuando la tengas, cuéntamela en 1–2 frases y el barrio del proyecto."
            append_mensajes(conv_ref, [
                {"role":"user","content": data.mensaje},
                {"role":"assistant","content": texto}
            ])
            return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
        prev_vec = conv_data.get("last_topic_vec")
        prev_sum = (conv_data.get("last_topic_summary") or "").strip()
        awaiting_confirm = bool(conv_data.get("awaiting_topic_confirm"))

        # Saludo simple: override incondicional (aunque haya estado previo)
        if is_plain_greeting(data.mensaje):
            # (opcional) soft reset de flags que podrían forzar flujo de propuesta/contacto
            conv_ref.update({
                "awaiting_topic_confirm": False,
                "candidate_new_topic_summary": None,
                "candidate_new_topic_vec": None,
                # limpia solo flags de flujo, NO datos duros
                "proposal_requested": False,
                "proposal_collected": False,
                "current_proposal": None,
                "argument_requested": False,
                "argument_collected": False,
                "contact_intent": None,
                "contact_requested": False,
                "contact_refused": False,
                "ultima_fecha": firestore.SERVER_TIMESTAMP
            })

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
        intro_hint = (action in ("greeting_smalltalk","meta")) and (not awaiting_confirm) and (not prev_sum)

        # --- NUEVO: si huele a propuesta, NO uses el saludo introductorio
        if intro_hint and (is_proposal_intent(data.mensaje) or looks_like_proposal_content(data.mensaje)):
            intro_hint = False

        if intro_hint:
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": BOT_INTRO_TEXT}
            ])
            return {"respuesta": BOT_INTRO_TEXT, "fuentes": [], "chat_id": chat_id}

        # --- NUEVO: si es CONSULTA, responder ya con RAG y salir ---
        consulta_try = llm_consulta_classifier(
            data.mensaje,
            historial_for_decider[-4:] if historial_for_decider else None
        )
        if consulta_try.get("is_consulta"):
            # Limpia cualquier estado de propuesta para no arrastrar
            conv_ref.update({
                "proposal_requested": False,
                "proposal_collected": False,
                "argument_requested": False,
                "argument_collected": False,
                "contact_intent": None,
                "awaiting_topic_confirm": False,
                "ultima_fecha": firestore.SERVER_TIMESTAMP
            })

            # (Opcional) marcar tema como Consulta/<título>
            titulo_cons = consulta_try.get("titulo") or "General"
            conv_ref.set({
                "categoria_general": ["Consulta"],
                "titulo_propuesta": [titulo_cons]
            }, merge=True)

            # RAG y respuesta breve, SIN pedir contacto
            hits = rag_search(data.mensaje, top_k=5)
            historial = load_historial_para_prompt(conv_ref)
            msgs = build_consulta_messages(
                data.mensaje,
                [h["texto"] for h in hits],
                historial
            )
            completion = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=msgs,
                temperature=0.2,
                max_tokens=240
            )
            texto = limit_sentences(completion.choices[0].message.content.strip(), 3)
            texto = strip_contact_requests(texto)

            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto}
            ])
            return {"respuesta": texto, "fuentes": hits, "chat_id": chat_id}



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
                # <<< NUEVO: preguntar ya mismo y salir >>>
                q = f'¿Seguimos con "{(prev_sum or "el tema anterior")}" o pasamos a "{curr_sum}"?'
                append_mensajes(conv_ref, [
                    {"role":"user","content": data.mensaje},
                    {"role":"assistant","content": q}
                ])
                return {"respuesta": q, "fuentes": [], "chat_id": chat_id}
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
        user_barrio = extract_user_barrio(data.mensaje)
        proj_loc = extract_project_location(data.mensaje)
        # Si ya pedimos contacto y el usuario respondió sin formato claro, usa LLM
        if conv_data.get("contact_requested") and not (name or phone or user_barrio):
            llm_data = llm_extract_contact_info(data.mensaje)
            if llm_data.get("nombre"):
                name = llm_data["nombre"]
            if llm_data.get("telefono"):
                phone = llm_data["telefono"]
            if llm_data.get("barrio"):
                user_barrio = llm_data["barrio"]
        if proj_loc and (conv_data.get("project_location") or "").strip().lower() != proj_loc.lower():
            conv_ref.update({"project_location": proj_loc})

               # PUNTO 17: preparar nota si residencia ≠ ubicación proyecto
        # ---------------------------------------------------------
        location_note = ""
        # Usa el barrio de residencia que venga en este turno o el guardado previamente
        residence = user_barrio or ((conv_data.get("contact_info") or {}).get("barrio"))
        if residence and proj_loc and residence.strip().lower() != proj_loc.strip().lower():
            if not bool(conv_data.get("location_note_sent")):
                location_note = f"Entendido: vives en {residence} y la propuesta es para {proj_loc}, ¿cierto?"

        # Helper para anteponer la nota (solo una vez)
        def add_location_note_if_needed(texto: str) -> str:
            nonlocal location_note  # solo lectura; si tu editor se queja, elimina esta línea
            if location_note:
                conv_ref.update({"location_note_sent": True})
                return (location_note + " " + texto).strip()
            return texto
        

        partials = {}
        if name:   partials["nombre"] = name
        if phone:  partials["telefono"] = phone
        if user_barrio: partials["barrio"] = user_barrio

        if partials:
            current_info = (conv_data.get("contact_info") or {})
            new_info = dict(current_info)  # copia

            # no sobrescribir un nombre ya existente
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
                    new_info.get("barrio") or user_barrio       # <-- barrio a usuarios
            )


                # --- AUTO-CIERRE si ya teníamos pedido de contacto y ahora se completó tel + ubicación ---
        info_actual_now = (conv_data.get("contact_info") or {})
        # si en este turno actualizamos nombre/barrio/teléfono, refléjalo:
        if partials:
            info_actual_now = {**info_actual_now, **partials}
        tel_ok = bool(info_actual_now.get("telefono") or phone)
        loc_ok = bool((conv_data.get("project_location") or None) or proj_loc)
        if conv_data.get("contact_requested") and tel_ok and loc_ok:
            nombre_txt = (info_actual_now.get("nombre") or "").strip()
            texto = (f"Gracias, {nombre_txt}. " if nombre_txt else "Gracias. ") + \
                    "Con estos datos escalamos el caso y te contamos avances."
            append_mensajes(conv_ref, [
                {"role":"user","content": data.mensaje},
                {"role":"assistant","content": texto}
            ])
            return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

        # === Política argumento + contacto ===
        policy = llm_contact_policy(prev_sum or curr_sum, data.mensaje)
        intent = policy.get("intent", "otro")

        # NUEVO: también considera contenido concreto como propuesta
        if is_proposal_intent(data.mensaje) or looks_like_proposal_content(data.mensaje):
            intent = "propuesta"
        
        # --- EARLY: manejar rechazo de datos ANTES del flujo de propuesta/contacto ---
        refused_now = detect_contact_refusal(data.mensaje)
        if refused_now and not phone:
            conv_ref.update({"contact_refused": True, "contact_requested": True})

        info_actual = (conv_data.get("contact_info") or {})
        already_col = bool(conv_data.get("contact_collected")) or bool(phone) or bool(info_actual.get("telefono"))
        already_req = bool(conv_data.get("contact_requested"))
        contact_refused_any = bool(conv_data.get("contact_refused")) or refused_now

        # Si ya habíamos pedido datos y ahora rechaza, responde política de privacidad y corta
        if contact_refused_any and already_req and not already_col:
            # Primera negativa: explicar por qué necesitamos los datos
            refusal_count = int(conv_data.get("contact_refusal_count") or 0)
            
            if refusal_count == 0:
                # Primera vez: explica
                conv_ref.update({"contact_refusal_count": 1})
                texto = PRIVACY_REPLY + " ¿Me compartes tus datos para poder ayudarte?"
            else:
                # Segunda negativa: acepta y despide
                conv_ref.update({
                    "contact_refused": True,
                    "contact_requested": False,
                    "contact_refusal_count": 2
                })
                texto = (
                    "Entiendo tu decisión y la respeto completamente. "
                    "Si en algún momento cambias de opinión, estaré aquí para ayudarte. "
                    "¡Que tengas un excelente día!"
                )
            
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto}
            ])
            return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

                # =========================
        # =========================
        # FLUJO DETERMINISTA: PROPUESTAS / SUGERENCIAS
        # =========================
        # PRIORIDAD 1: Si ya estamos en flujo de propuestas, NO salir hasta completar
        already_in_proposal_flow = (
            bool(conv_data.get("proposal_requested")) or
            bool(conv_data.get("proposal_collected")) or
            bool(conv_data.get("argument_requested")) or
            conv_data.get("contact_intent") == "propuesta"
        )

        # PRIORIDAD 2: Detectar nueva propuesta
        new_proposal_detected = (
            not is_plain_greeting(data.mensaje) and (
                intent == "propuesta" or
                looks_like_proposal_content(data.mensaje)
            )
        )

        is_proposal_flow = already_in_proposal_flow or new_proposal_detected

        if is_proposal_flow:
            # 1) ¿Este turno YA trae la propuesta? -> capturar y pasar a argumento
            if not conv_data.get("proposal_collected"):
                # --- extracción robusta: si el trigger no está al inicio, corta desde ahí ---
                proposal_text = extract_proposal_text(data.mensaje)
                m_any = re.search(
                    r'(?:me\s+gustar[íi]a\s+(?:proponer|que)|quisiera\s+(?:proponer|que)|'
                    r'quiero\s+proponer|propongo(?:\s+que)?|mi\s+(?:idea|propuesta)\s+(?:es|ser[ií]a))\s*(.*)',
                    data.mensaje,
                    flags=re.IGNORECASE
                )
                if m_any:
                    tail = (m_any.group(1) or "").strip()
                    if tail:
                        proposal_text = limit_sentences(re.sub(r'\s+', ' ', tail), 2)

                proposal_clean = _normalize_text(proposal_text)

                # mensajes genéricos que NO son propuesta todavía
                generic_only = proposal_clean in {"", "algo", "una idea", "una propuesta", "un tema", "varias cosas"}

                # heurística: contenido mínimo o verbo de acción típico
                has_action = bool(re.search(
                    r"\b(arregl|mejor|constru|instal|crear|paviment|ilumin|señaliz|ampli|dotar|regular(?!idad|mente)|prohib|mult|beca|subsid)\w*",
                    proposal_clean
                ))
                # sustantivos típicos de infraestructura + presencia de ubicación (residencia o proyecto)
                has_infra_noun = bool(re.search(
                    r'\b(parque|and[eé]n|andenes|sema[fF]or[oa]s?|luminarias?|alumbrado|cancha|juegos|polideportivo)\b',
                    proposal_clean
                ))
                has_location_ctx = bool(proj_loc or user_barrio or re.search(r'\bbarrio\b', proposal_clean))

                has_concrete = looks_like_proposal_content(data.mensaje)

                if has_concrete:
                    conv_ref.update({
                        "current_proposal": proposal_text,
                        "proposal_requested": True,
                        "proposal_collected": True,
                        "argument_requested": True,
                        "argument_collected": False,      # <--- NUEVO: reinicia argumento
                        "contact_intent": "propuesta",
                        "contact_requested": False,       # <--- NUEVO: evita arrastrar pedido previo
                        "categoria_general": [],
                        "titulo_propuesta": [],
                        "ultima_fecha": firestore.SERVER_TIMESTAMP
                    })
                    texto = positive_ack_and_request_argument(
                        name,
                        conv_data.get("project_location") or proj_loc
                    )
                    texto = add_location_note_if_needed(texto)
                    append_mensajes(conv_ref, [
                        {"role":"user","content": data.mensaje},
                        {"role":"assistant","content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}


                # Solo intención -> primero pide la propuesta
                conv_ref.update({
                    "proposal_requested": True,
                    "proposal_collected": False,
                    "argument_requested": False,
                    "argument_collected": False,
                    "contact_intent": "propuesta",
                    "contact_requested": False,
                    "ultima_fecha": firestore.SERVER_TIMESTAMP
                })
                texto = "¡Perfecto! ¿Cuál es tu propuesta o sugerencia? Cuéntamela en una o dos frases."

                # <<< PUNTO 17 >>>
                texto = add_location_note_if_needed(texto)

                append_mensajes(conv_ref, [
                    {"role": "user", "content": data.mensaje},
                    {"role": "assistant", "content": texto}
                ])
                return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

            # 2) Ya pedimos propuesta y aún no la hemos guardado -> validar este turno
            if conv_data.get("proposal_requested") and not conv_data.get("proposal_collected"):
                # contador suave de intentos (nudge)
                nudges = int(conv_data.get("proposal_nudge_count") or 0)

                # a) NEGACIÓN explícita / “más tarde”
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
                        "proposal_nudge_count": 0,
                        "ultima_fecha": firestore.SERVER_TIMESTAMP
                    })
                    texto = "Perfecto, sin problema. Cuando la tengas, cuéntamela en 1–2 frases y el barrio del proyecto."
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

                # b) ¿Trae CONTENIDO real de propuesta?
                if looks_like_proposal_content(data.mensaje):
                    conv_ref.update({
                        "current_proposal": extract_proposal_text(data.mensaje),
                        "proposal_collected": True,
                        "argument_requested": True,
                        "proposal_nudge_count": 0,
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

                # c) El usuario trajo una PREGUNTA en lugar de la propuesta
                if is_question_like(data.mensaje):
                    # quitamos el “modo propuesta” para no forzar el flujo
                    conv_ref.update({
                        "proposal_requested": False,
                        "contact_intent": None,
                        "proposal_nudge_count": 0,
                        "ultima_fecha": firestore.SERVER_TIMESTAMP
                    })
                    texto = "¿Prefieres que te responda esa pregunta ahora o seguimos con tu propuesta? Si es propuesta, cuéntamela en 1–2 frases."
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

                # d) No hay contenido aún → NUDGE con escalado suave
                nudges += 1
                conv_ref.update({"proposal_nudge_count": nudges, "ultima_fecha": firestore.SERVER_TIMESTAMP})
                if nudges == 1:
                    texto = "Claro. ¿Cuál es tu propuesta? Dímela en 1–2 frases y el barrio del proyecto."
                elif nudges == 2:
                    texto = "Para ayudarte mejor, escribe la propuesta en 1–2 frases (ej.: “Arreglar luminarias del parque de San José”)."
                else:
                    # salir del modo propuesta para no quedar pegados
                    conv_ref.update({
                        "proposal_requested": False,
                        "contact_intent": None,
                        "proposal_nudge_count": 0
                    })
                    texto = "Todo bien, salgo del modo propuesta. Si prefieres, dime tu pregunta o tema y te ayudo de una."
                append_mensajes(conv_ref, [
                    {"role":"user","content": data.mensaje},
                    {"role":"assistant","content": texto}
                ])
                return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

            # 3) Ya tenemos propuesta y estamos pidiendo argumento
            if conv_data.get("proposal_collected") and not conv_data.get("argument_collected"):
                # Solo dar por válido el argumento si:
                #   a) lo pedimos en el turno anterior (argument_requested=True y el último fue assistant) y el usuario respondió algo (≥5 chars), o
                #   b) hay señales causales claras (porque/ya que/debido…)
                last_role = historial_for_decider[-1]["role"] if historial_for_decider else None
                just_asked_arg = bool(conv_data.get("argument_requested")) and (last_role == "assistant")

                    # Detectar argumento de forma inteligente
                proposal_ctx = conv_data.get("current_proposal") or ""
                is_argument = (
                    has_argument_text(data.mensaje) or  # heurística rápida
                    llm_is_argument(data.mensaje, proposal_ctx) or  # LLM inteligente
                    (just_asked_arg and len(_normalize_text(data.mensaje)) >= 5)  # respondió a pregunta
                )
                
                if is_argument:
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
                    # <<< PUNTO 17 >>>
                    texto = add_location_note_if_needed(texto)

                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}


            # 4) Ya tenemos propuesta + argumento y falta contacto
            if conv_data.get("argument_collected") and not (conv_data.get("contact_collected") or phone):
                # No insistir si el ciudadano rechazó dar datos
                if contact_refused_any:
                    texto = PRIVACY_REPLY
                    append_mensajes(conv_ref, [
                        {"role":"user","content": data.mensaje},
                        {"role":"assistant","content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}

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
                                and bool(conv_data.get("proposal_collected"))
                                and not conv_data.get("argument_requested")
                                and not conv_data.get("argument_collected"))
            if need_argument_now:
                conv_ref.update({"argument_requested": True})
                texto = craft_argument_question(name, proj_loc)
                append_mensajes(conv_ref, [
                    {"role": "user", "content": data.mensaje},
                    {"role": "assistant", "content": texto}
                ])
                return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}



        # nunca pedir contacto sin propuesta + argumento
        if not bool(conv_data.get("proposal_collected")) or not bool(conv_data.get("argument_collected")):
            policy = {"should_request": False, "intent": intent, "reason": "gated_by_phase"}

        # Reusar flags ya calculados arriba
        should_ask_now = (policy.get("should_request")
                        and intent in ("propuesta", "problema")
                        and bool(conv_data.get("argument_collected"))
                        and bool(conv_data.get("argument_requested"))   # <-- NUEVO GATE
                        and not already_col
                        and not contact_refused_any
                        and not already_req)

        if should_ask_now and not already_req:
            conv_ref.update({"contact_intent": intent, "contact_requested": True})

        # Si toca pedir contacto y el usuario no dio nada aún -> atajo directo
        if should_ask_now and not already_req and not (name or phone or user_barrio):
            info_actual = (conv_data.get("contact_info") or {})
            faltan = []
            if not info_actual.get("nombre"):   faltan.append("nombre")
            if not info_actual.get("barrio"):   faltan.append("barrio")
            if not info_actual.get("telefono"): faltan.append("celular")
            if not (conv_data.get("project_location") or proj_loc): faltan.append("project_location")

            if faltan == ["project_location"]:
                texto_directo = build_project_location_request()
            else:
                texto_directo = build_contact_request(faltan or ["celular"])
            # <<< PUNTO 17 >>>
            texto_directo = add_location_note_if_needed(texto_directo)

            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto_directo}
            ])
            return {"respuesta": texto_directo, "fuentes": [], "chat_id": chat_id}



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
            if not (conv_data.get("project_location") or proj_loc):
                texto = build_project_location_request()
            else:
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
            if not (conv_data.get("project_location") or proj_loc): missing.append("project_location")
            if missing:
                texto = (build_project_location_request()
                        if missing == ["project_location"]
                        else build_contact_request(missing))
        texto = add_location_note_if_needed(texto)

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

def last_meaningful_user_from_conv(conv_data: dict) -> str:
    """
    Devuelve el último mensaje de rol 'user' que no sea un saludo/ack trivial
    y tenga al menos 5 caracteres normalizados.
    """
    msgs = (conv_data or {}).get("mensajes", [])
    for m in reversed(msgs):
        if (m.get("role") == "user"):
            txt = (m.get("content") or "").strip()
            if txt and not is_plain_greeting(txt) and len(_normalize_text(txt)) >= 5:
                return txt
    return ""

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
        f"propuestas_por_categoria.{categoria}": Increment(1),
        f"resumen_tono.{tono}": Increment(1),
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
        # Fallback: si el "último usuario" es saludo/ trivial o vacío,
        # toma el último mensaje de usuario con contenido real.
        if (not ultimo_u) or is_plain_greeting(ultimo_u) or len(_normalize_text(ultimo_u)) < 5:
            alt_u = last_meaningful_user_from_conv(conv_data)
            if alt_u:
                ultimo_u = alt_u
        if not ultimo_u:
            return {"ok": False, "mensaje": "No hay mensajes para clasificar."}
        
        # 1) ¿Hay propuesta ya? -> Usamos el flujo de propuesta (más abajo).
        hay_propuesta = bool(conv_data.get("proposal_collected"))

        # 2) Si NO hay propuesta, intentamos clasificar como CONSULTA.
        if not hay_propuesta:
            # 2.1 rol conversacional (para evitar clasificar saludos/acks)
            decision_cls = llm_decide_turn(
                last_topic_summary=(conv_data.get("last_topic_summary") or ""),
                awaiting_confirm=bool(conv_data.get("awaiting_topic_confirm")),
                last_two_turns=[{"role":"user","content": ultimo_u},{"role":"assistant","content": ultima_a}],
                current_text=ultimo_u
            )

            if decision_cls.get("action") in ("greeting_smalltalk", "meta"):
                # intenta rescatar el último mensaje significativo; si no hay, sal
                alt_u = last_meaningful_user_from_conv(conv_data)
                if alt_u and alt_u != ultimo_u:
                    ultimo_u = alt_u
                else:
                    return {"ok": True, "skipped": True, "reason": "saludo_o_meta_detectado_por_llm"}

            # 2.2 aquí SIEMPRE corremos el clasificador de CONSULTA
            consulta = llm_consulta_classifier(
                ultimo_u,
                [{"role":"user","content": ultimo_u},{"role":"assistant","content": ultima_a}]
            )

            # DEBUG (útil): guarda qué decidió
            db.collection("conversaciones").document(chat_id).set({
                "debug_last_classify": {
                    "ts": firestore.SERVER_TIMESTAMP,
                    "ultimo_u": ultimo_u,
                    "consulta_raw": consulta,
                }
            }, merge=True)

            if consulta.get("is_consulta"):
                categoria = "Consulta"
                titulo = consulta.get("titulo") or "General"
                tono = "neutral"
                palabras = []

                conv_ref = db.collection("conversaciones").document(chat_id)
                updates = {
                    "ultima_fecha": firestore.SERVER_TIMESTAMP,
                    "categoria_general": [categoria],
                    "titulo_propuesta": [titulo],
                }
                if not conv_data.get("tono_detectado"):
                    updates["tono_detectado"] = tono
                conv_ref.set(updates, merge=True)

                # historial temático
                hist_last = (conv_data.get("topics_history") or [])
                last_item = hist_last[-1] if hist_last else {}
                if (not last_item or last_item.get("categoria") != categoria or last_item.get("titulo") != titulo):
                    conv_ref.set({
                        "topics_history": firestore.ArrayUnion([{
                            "categoria": categoria,
                            "titulo": titulo,
                            "tono": tono,
                            "fecha": firestore.SERVER_TIMESTAMP
                        }])
                    }, merge=True)

                db.collection("categorias_tematicas").document(categoria).set({"nombre": categoria}, merge=True)

                awaiting = bool(conv_data.get("awaiting_topic_confirm"))
                ya_contado = bool(conv_data.get("panel_contabilizado"))
                debe_contar = (not awaiting) and (not ya_contado) if body.contabilizar is None else bool(body.contabilizar)

                usuario_id = conv_data.get("usuario_id", chat_id)
                if debe_contar:
                    update_panel_resumen(categoria, tono, titulo, usuario_id)
                    conv_ref.set({"panel_contabilizado": True}, merge=True)

                return {"ok": True, "clasificacion": {
                    "categoria_general": categoria,
                    "titulo_propuesta": titulo,
                    "tono_detectado": tono,
                    "palabras_clave": palabras,
                    "contabilizado_en_panel": bool(debe_contar)
                }}

            # No pasó como consulta y tampoco hay propuesta -> no clasificar
            return {"ok": True, "skipped": True, "reason": "no_consulta_y_sin_propuesta"}




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
        # No usar defaults genéricos: si vienen vacíos, mantenlos en None
        categoria = data.get("categoria_general") or data.get("categoria") or None
        titulo    = data.get("titulo_propuesta") or data.get("titulo") or None
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
        
        # --- NUEVO: no persistir categorías/títulos genéricos o vacíos ---
        GENERIC_CATS = {"general", "otras", "sin clasificar", "otro"}
        GENERIC_TITLES = {
            "propuesta ciudadana", "propuesta no especificada",
            "interaccion inicial", "interacción inicial",
            "solicitud de propuesta", "sin titulo", "sin título", "—"
        }

        cat_norm = _normalize_text(categoria or "")
        tit_norm = _normalize_text(titulo or "")

        is_generic_cat = (not categoria) or (cat_norm in GENERIC_CATS)
        is_generic_tit = (not titulo) or (len(titulo.strip()) < 6) or (tit_norm in GENERIC_TITLES)

        if is_generic_cat or is_generic_tit:
            # No escribas nada en categoria/titulo, no toques panel ni historial temático
            return {"ok": True, "skipped": True, "reason": "irrelevante_sin_categoria_titulo"}


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