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
    """
    Si entrega teléfono lo guardamos en 'usuarios', si no, en 'anonimos'.
    Usamos chat_id como identificador estable de ese hilo.
    """
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


def ensure_conversacion(chat_id: str, usuario_id: str, faq_origen: Optional[str]):
    """
    Crea conversaciones/{chat_id} si no existe y añade campos de contacto y flags de argumento.
    """
    conv_ref = db.collection("conversaciones").document(chat_id)
    if not conv_ref.get().exists:   # propiedad .exists
        conv_ref.set({
            "usuario_id": usuario_id,
            "faq_origen": faq_origen or None,
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

            # === Flujo argumento + contacto ===
            "argument_requested": False,
            "argument_collected": False,
            "contact_intent": None,        # 'propuesta' | 'problema' | 'reconocimiento' | 'otro'
            "contact_requested": False,
            "contact_collected": False,    # True solo si hay teléfono
            "contact_info": {"nombre": None, "barrio": None, "telefono": None},
        })
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
    if snap.exists:   # propiedad .exists
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
    """Heurística simple para detectar 'argumento/razón' en el turno del ciudadano."""
    t = _normalize_text(t)
    keys = [
        "porque","ya que","debido","para que","para ","peligro","riesgo","falta","no hay",
        "estan oxida","esta roto","es necesario","urge","hace falta","afecta","impacta","contamina",
        "seguridad","salud","empleo","tránsito","movilidad","ambiental"
    ]
    return any(k in t for k in keys)

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
    """
    Decide si corresponde PEDIR contacto ahora y el tipo de intención.
    Devuelve JSON: { "should_request": bool, "intent": "propuesta|problema|reconocimiento|otro", "reason": "..." }
    """
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


def parse_contact_from_text(text: str) -> Dict[str, Optional[str]]:
    """
    Extrae datos básicos si el usuario los comparte libremente.
    - teléfono: dígitos con 8-12 números (Col: 10 usual).
    - nombre: varias heurísticas (al inicio antes de una coma, 'soy/me llamo/mi nombre es').
    - barrio: variantes de 'en el barrio X' / 'del barrio X', corta en coma/punto o conectores.
    """
    # Teléfono
    tel = None
    m = re.search(r'(\+?\d[\d\s\-]{7,14}\d)', text)
    if m:
        tel = re.sub(r'\D', '', m.group(1))
        if len(tel) < 8 or len(tel) > 12:
            tel = None

    # Nombre (patrones explícitos)
    nombre = None
    m = re.search(r'\b(?:soy|me llamo|mi nombre es)\s+([A-Za-zÁÉÍÓÚÑáéíóúñ ]{2,40})', text, flags=re.IGNORECASE)
    if m:
        nombre = m.group(1).strip()

    # Nombre (al inicio antes de coma o conector: "Leandro, vivo en...")
    if not nombre:
        m = re.search(r'^\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})\s*(?=,|\s+vivo\b|\s+soy\b|\s+mi\b|\s+desde\b|\s+del\b|\s+de\b)', text)
        if m:
            posible = m.group(1).strip()
            if posible.lower() not in {"hola","buenas","buenos dias","buenas tardes","buenas noches"}:
                nombre = posible

    # Barrio
    barrio = None
    m = re.search(r'\b(?:en el|en la|del|de la|de el)?\s*barrio\s+([A-Za-zÁÉÍÓÚÑáéíóúñ0-9 \-]{2,50}?)(?=,|\.|\s+es\b|\s+mi\b|\s+y\b|$)', text, flags=re.IGNORECASE)
    if m:
        barrio = m.group(1).strip(" .,")

    return {"nombre": nombre, "barrio": barrio, "telefono": tel}

def build_contact_request(missing: List[str]) -> str:
    etiquetas = {"nombre": "tu nombre", "barrio": "tu barrio", "celular": "un número de contacto"}
    pedir = [etiquetas[m] for m in missing]
    frase = pedir[0] if len(pedir) == 1 else (", ".join(pedir[:-1]) + " y " + pedir[-1])
    return f"Para escalar y darle seguimiento, ¿me compartes {frase}? Lo usamos solo para informarte avances."

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
        conv_ref = ensure_conversacion(chat_id, usuario_id, data.faq_origen)

        conv_data = (conv_ref.get().to_dict() or {})
        prev_vec = conv_data.get("last_topic_vec")
        prev_sum = (conv_data.get("last_topic_summary") or "").strip()
        awaiting_confirm = bool(conv_data.get("awaiting_topic_confirm"))

        # Saludo directo sin gastar modelo
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

        # === PARSING DE CONTACTO (no cerramos sin teléfono) ===
        parsed = parse_contact_from_text(data.mensaje)           # {"nombre":..., "barrio":..., "telefono":...}
        partials = {k: v for k, v in parsed.items() if v}
        tel = parsed.get("telefono")

        if partials:
            current_info = (conv_data.get("contact_info") or {})
            new_info = {**current_info, **{k: v or current_info.get(k) for k, v in parsed.items()}}
            conv_ref.update({"contact_info": new_info})
            if tel:  # contacto "suficiente"
                conv_ref.update({"contact_collected": True})
                upsert_usuario_o_anon(chat_id, new_info.get("nombre") or data.nombre or data.usuario, tel, data.canal)

        # === Política de argumento y contacto ===
        policy = llm_contact_policy(prev_sum or curr_sum, data.mensaje)
        intent = policy.get("intent", "otro")

        # ¿Este turno contiene argumento?
        prev_role = historial_for_decider[-1]["role"] if historial_for_decider else None
        argument_ready = (prev_role == "assistant") and has_argument_text(data.mensaje)
        if argument_ready and not conv_data.get("argument_collected"):
            conv_ref.update({"argument_collected": True})

        # ¿Debemos pedir argumento ahora?
        need_argument_now = (intent in ("propuesta", "problema")
                             and not (conv_data.get("argument_requested") or conv_data.get("argument_collected")))
        if need_argument_now:
            conv_ref.update({"argument_requested": True})

        # ¿Debemos pedir contacto ahora? (solo después de tener argumento)
        already_req = bool(conv_data.get("contact_requested"))
        already_col = bool(conv_data.get("contact_collected")) or bool(tel)
        should_ask_now = (policy.get("should_request")
                          and intent in ("propuesta", "problema")
                          and (conv_data.get("argument_collected") or argument_ready)
                          and not already_col)

        if should_ask_now and not already_req:
            conv_ref.update({"contact_intent": intent, "contact_requested": True})

        # Si debemos pedir contacto y el usuario no dio nada aún -> atajo (sin LLM)
        if should_ask_now and not partials:
            texto_directo = build_contact_request(["nombre", "celular"])  # barrio opcional
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
            emphasize_argument=need_argument_now,
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

        # --- POST: cierre conciso si llegó teléfono en este turno ---
        if tel:
            nombre_txt = (parsed.get("nombre") or "").strip()
            cierre = (f"Gracias, {nombre_txt}. " if nombre_txt else "Gracias. ")
            cierre += "Con estos datos escalamos el caso y te contamos avances."
            if "tel" in texto.lower() or "celu" in texto.lower() or len(texto) < 30:
                texto = cierre
            else:
                texto += "\n\n" + cierre

        # --- POST: si no hay teléfono pero sí dio algo y tocaba pedir -> pide lo que falta ---
        elif partials and should_ask_now:
            info_actual = (conv_data.get("contact_info") or {})
            missing = []
            if not (info_actual.get("nombre") or parsed.get("nombre")):
                missing.append("nombre")
            if not (info_actual.get("telefono") or parsed.get("telefono")):
                missing.append("celular")
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
    if not snap.exists:   # propiedad .exists
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

@app.post("/clasificar")
async def clasificar(body: ClasificarIn):
    try:
        chat_id = body.chat_id
        ultimo_u, ultima_a, conv_data = read_last_user_and_bot(chat_id)
        if not ultimo_u:
            return {"ok": False, "mensaje": "No hay mensajes para clasificar."}

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
