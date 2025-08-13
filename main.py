# ============================
#  WilderBot API (FastAPI)
#  - /responder : RAG + historial + guardado en tu BD
#  - /clasificar: (nuevo) clasifica la conversación y actualiza panel
#  Nota: este archivo está ampliamente comentado para facilitar mantenimiento.
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

# --- Firestore Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore

import math  # <— nuevo

# Umbral de similitud para detectar cambio de tema (0..1)
TOPIC_SIM_THRESHOLD = float(os.getenv("TOPIC_SIM_THRESHOLD", "0.78"))

# =========================================================
#  Config e inicialización básica
# =========================================================

load_dotenv()                 # Permite usar .env en local; en Render usas env vars

app = FastAPI()               # Instancia de FastAPI

# === Variables de entorno (con valores por defecto coherentes) ===
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o")               # modelo para responder
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") # modelo de embeddings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "wilder-frases")
GOOGLE_CREDS     = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "/etc/secrets/firebase.json"

# === Clientes de terceros ===
client = OpenAI(api_key=OPENAI_API_KEY)       # Cliente OpenAI (chat + embeddings)

pc = Pinecone(api_key=PINECONE_API_KEY)       # Cliente Pinecone
index = pc.Index(PINECONE_INDEX)              # Índice donde guardamos/consultamos frases de Wilder

# === Firestore Admin ===
# Intentamos obtener una app ya inicializada; si no existe, la creamos.
try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate(GOOGLE_CREDS)   # Archivo de credenciales (Render: secret file)
    firebase_admin.initialize_app(cred)

db = firestore.client()   # Cliente Firestore listo para leer/escribir

# =========================================================
#  Esquemas de entrada/salida
# =========================================================

class Entrada(BaseModel):
    """
    Payload de entrada en /responder.
    NOTA: mantenemos compatibilidad con n8n. Solo 'mensaje' es requerido.
    """
    mensaje: str
    usuario: Optional[str] = None
    chat_id: Optional[str] = None     # Ej: tg_12345 (identificador de conversación)
    canal: Optional[str] = None       # telegram | whatsapp | web
    faq_origen: Optional[str] = None  # slug/ID si inició desde FAQ
    nombre: Optional[str] = None      # nombre mostrado (si disponible)
    celular: Optional[str] = None     # "telefono" en tu BD (si el ciudadano lo entregó)

class ClasificarIn(BaseModel):
    """
    Payload mínimo para /clasificar.
    Lo llamamos con el chat_id de la conversación que queremos clasificar.
    """
    chat_id: str

    # Si es None, el backend decide: solo contabiliza si aún no se ha contado.
    # Si es True, fuerza un nuevo registro en panel (para casos confirmados de "nueva propuesta").
    # Si es False, nunca contabiliza (solo actualiza los campos de la conversación).
    contabilizar: Optional[bool] = None

# =========================================================
#  Health Check
# =========================================================

@app.get("/health")
async def health():
    """Endpoint de salud para monitoreo (Render/N8N)."""
    return {"status": "ok"}

# =========================================================
#  Helpers de Base de Datos (ajustados a tu esquema de BD)
# =========================================================

def upsert_usuario_o_anon(chat_id: str, nombre: Optional[str], telefono: Optional[str], canal: Optional[str]) -> str:
    """
    Crea/actualiza el registro del ciudadano en 'usuarios' (si entregó teléfono)
    o en 'anonimos' (si no). Devolvemos 'usuario_id'.
    Decisión: usamos 'chat_id' como 'usuario_id' estable para simplificar.
    """
    usuario_id = chat_id

    if telefono:  # Caso: guardar en 'usuarios'
        ref = db.collection("usuarios").document(usuario_id)
        doc = ref.get()
        if not doc.exists:
            # Primera vez que lo vemos
            ref.set({
                "nombre": nombre or "",
                "telefono": telefono,
                "barrio": None,
                "fecha_registro": firestore.SERVER_TIMESTAMP,
                "chats": [chat_id]
            })
        else:
            # Ya existe: actualizamos nombre/teléfono y agregamos chat si no estaba
            ref.update({
                "nombre": nombre or doc.to_dict().get("nombre", ""),
                "telefono": telefono,
                "chats": firestore.ArrayUnion([chat_id])
            })
    else:  # Caso: guardar en 'anonimos'
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
    """
    Asegura que exista el documento conversaciones/{chat_id}.
    Si no existe, lo crea con los campos de tu modelo.
    """
    conv_ref = db.collection("conversaciones").document(chat_id)
    if not conv_ref.get().exists:
        conv_ref.set({
            "usuario_id": usuario_id,
            "faq_origen": faq_origen or None,
            "categoria_general": None,
            "titulo_propuesta": None,
            "mensajes": [],  # En tu modelo, los turnos viven como arreglo en el documento
            "fecha_inicio": firestore.SERVER_TIMESTAMP,
            "ultima_fecha": firestore.SERVER_TIMESTAMP,
            "tono_detectado": None,
            "last_topic_vec": None,               # embedding del último tema consolidado
            "last_topic_summary": None,           # resumen breve del último tema
            "awaiting_topic_confirm": False,      # estamos esperando confirmación de nuevo tema
            "candidate_new_topic_summary": None,  # resumen del tema candidato
            "candidate_new_topic_vec": None,      # embedding del tema candidato
            "topics_history": [],  # <— NUEVO: historial de temas clasificados
        })
    else:
        # Si ya existe, solo refrescamos la última fecha de actividad
        conv_ref.update({"ultima_fecha": firestore.SERVER_TIMESTAMP})
    return conv_ref


def append_mensajes(conv_ref, nuevos: List[Dict[str, Any]]):
    """
    Agrega mensajes al arreglo 'mensajes'.
    Por orden y duplicados, preferimos leer + extender + reescribir
    (en lugar de ArrayUnion que no preserva duplicados).
    """
    snap = conv_ref.get()
    data = snap.to_dict() or {}
    arr = data.get("mensajes", [])
    arr.extend(nuevos)
    conv_ref.update({"mensajes": arr, "ultima_fecha": firestore.SERVER_TIMESTAMP})


def load_historial_para_prompt(conv_ref) -> List[Dict[str, str]]:
    """
    Carga los últimos turnos para dar continuidad de tema.
    Devuelve una lista de dicts con formato OpenAI [{'role': 'user'|'assistant', 'content': '...'}].
    """
    snap = conv_ref.get()
    if snap.exists:
        data = snap.to_dict() or {}
        msgs = data.get("mensajes", [])
        # Sanear y quedarnos con los últimos 8 turnos (ajustable)
        out = []
        for m in msgs[-8:]:
            role = m.get("role")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": content})
        return out
    return []


def cosine_sim(a: List[float] | None, b: List[float] | None) -> float:
    """Coseno entre dos embeddings."""
    if not a or not b:
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0


def llm_decide_turn(
    last_topic_summary: str,
    awaiting_confirm: bool,
    last_two_turns: List[Dict[str, str]],
    current_text: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Usa el LLM para decidir qué tipo de turno es este mensaje.
    Posibles 'action':
      - greeting_smalltalk: saludo/cordialidad sin contenido programático.
      - continue_topic: profundiza el tema vigente (mismo hilo).
      - new_topic: propone tema distinto (solicitar confirmación).
      - confirm_new_topic: usuario confirma que sí es nuevo tema.
      - reject_new_topic: usuario dice que NO, que siga el tema anterior.
      - meta: mensajes de control, 'gracias', 'ok', etc. (no contaminar).

    Return:
      {
        "action": "...",
        "reason": "breve explicación",
        "current_summary": "resumen corto del mensaje actual",
        "topic_label": "rótulo corto del tema si aplica"
      }
    """
    model = model or OPENAI_MODEL

    sys = (
        "Eres un asistente que clasifica el rol conversacional de un mensaje "
        "dado el tema vigente.\n"
        "Responde SOLO un JSON válido con claves: action, reason, current_summary, topic_label.\n"
        "Reglas:\n"
        "- greeting_smalltalk: saludos, cortesías, '¿cómo estás?', 'gracias', etc.\n"
        "- continue_topic: aporta detalles o responde dentro del mismo tema vigente.\n"
        "- new_topic: aparece un asunto distinto al vigente (si no hay vigente, proponlo como posible nuevo).\n"
        "- confirm_new_topic: el usuario explícitamente acepta cambiar de tema o dice 'pasemos a X'.\n"
        "- reject_new_topic: el usuario dice que NO y que sigan con el tema anterior.\n"
        "- meta: mensajes sin contenido programático (p. ej., 'listo', 'ok', 'repite').\n"
        "Incluye 'topic_label' breve si detectas un tema (p.ej. 'parque La Esperanza', 'muro del colegio', 'mejorar vía al centro')."
    )

    last_turns_text = "\n".join([f"{t.get('role')}: {t.get('content','')}" for t in last_two_turns[-2:]])
    usr = (
        f"Tema vigente (si existe): {last_topic_summary or '(ninguno)'}\n"
        f"¿Estamos esperando confirmación de cambio de tema?: {'Sí' if awaiting_confirm else 'No'}\n"
        f"Últimos turnos recientes:\n{last_turns_text}\n\n"
        f"Mensaje actual del ciudadano:\n{current_text}\n\n"
        "Devuelve el JSON ahora."
    )

    out = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": usr}],
        temperature=0.1,
        max_tokens=220
    ).choices[0].message.content

    try:
        data = json.loads(out)
    except Exception:
        # fallback conservador
        data = {"action": "continue_topic", "reason": "fallback", "current_summary": current_text[:120], "topic_label": ""}

    # sanea claves mínimas
    data.setdefault("action", "continue_topic")
    data.setdefault("current_summary", current_text[:120])
    data.setdefault("topic_label", "")
    data.setdefault("reason", "")
    return data

# =========================================================
#  RAG (embeddings + búsqueda vectorial)
# =========================================================

def rag_search(query: str, top_k: int = 5):
    """
    1) Crea el embedding del texto del ciudadano.
    2) Consulta el índice de Pinecone para traer las frases de Wilder más afines.
    3) Devuelve una lista de hits con {id, texto, score}.
    """
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


def build_messages(
    user_text: str,
    rag_snippets: List[str],
    historial: List[Dict[str, str]],
    topic_change_suspect: bool = False,
    prev_summary: str = "",
    new_summary: str = "",
    intro_hint: bool = False,
):
    """
    Construye los mensajes que enviaremos a OpenAI:
    - system: reglas de estilo y límites (máx 4 frases)
    - historial: continuidad de conversación
    - user: mensaje actual + contexto RAG
    - si hay sospecha de cambio de tema, instruye a formular la pregunta humana.
    - si intro_hint=True: presentación breve enfocada a captar problemas/propuestas/elogios.
    """
    contexto = "\n".join([f"- {s}" for s in rag_snippets if s.strip()])

    system_msg = (
        "Actúa como Wilder Escobar, Representante a la Cámara en Colombia.\n"
        "Tono: muy cercano, empático y humano. Responde en **máximo 4 frases** (sin párrafos largos).\n"
        "Si pides datos, haz **1 pregunta puntual**. Si el usuario cambia de tema, respóndelo sin perder cortesía.\n"
        "Si detectas que el nuevo mensaje ES UN TEMA DISTINTO al que venían tratando, hazlo notar con aprecio y plantea la pregunta de forma natural, por ejemplo:\n"
        "  - \"Valoro mucho tus aportes, ¿quieres que terminemos de hablar acerca de la necesidad de {{tema_anterior}}? "
        "o prefieres que pasemos a hablar sobre {{nuevo_tema}}?\"\n"
        "Reemplaza {{tema_anterior}} y {{nuevo_tema}} por resúmenes cortos y claros de cada asunto.\n"
        "Usa el contexto recuperado para mantener el estilo y coherencia, y evita inventar hechos."
    )


    # Presentación proactiva cuando sea saludo/smalltalk sin tema vigente
    if intro_hint:
        system_msg += (
            "\n\nCUANDO SEA UN SALUDO Y NO HAYA TEMA ACTUAL:\n"
            "- Preséntate brevemente como la mano derecha de Wilder Escobar.\n"
            "- Explica en una sola frase que estás aquí para escuchar problemas, propuestas o comentarios positivos y canalizarlos.\n"
            "- Cierra con **una** pregunta concreta para invitar a contar la situación o idea (máx. 1 línea)."
        )

    if topic_change_suspect and new_summary:
        human_q = (
            f'Valoro mucho tus aportes, ¿quieres que terminemos de hablar acerca de '
            f'"{(prev_summary or "el tema anterior")}" o prefieres que pasemos a hablar '
            f'sobre "{new_summary}"?'
        )
        system_msg += (
            "\n\nDetecto que el mensaje podría ser un **tema distinto**. Primero, hazlo notar con aprecio y pregunta de forma natural:\n"
            f'- "{human_q}"\n'
            "Espera confirmación antes de avanzar con acciones o registrar como propuesta aparte."
        )

    contexto_msg = "Contexto recuperado (frases reales de Wilder):\n" + (contexto if contexto else "(sin coincidencias relevantes)")

    msgs = [{"role": "system", "content": system_msg}]
    if historial:
        msgs.extend(historial[-8:])
    msgs.append({"role": "user", "content": f"{contexto_msg}\n\nMensaje del ciudadano:\n{user_text}"})
    return msgs

# =========================================================
#  Endpoint principal: /responder
# =========================================================

@app.post("/responder")
async def responder(data: Entrada):
    """
    Flujo:
    1) Resolver chat_id y registrar usuario vs anónimo.
    2) Asegurar doc en 'conversaciones'.
    3) Detección de cambio de tema (embeddings) y actualización de banderas.
    4) RAG (Pinecone) + historial (Firestore) → construir prompt.
    5) Llamar a OpenAI para la respuesta final (corta y humana).
    6) Guardar ambos turnos en el arreglo 'mensajes'.
    """
    try:
        # 1) chat_id estable: si no viene, generamos uno temporal web_XXXX
        chat_id = data.chat_id or f"web_{os.urandom(4).hex()}"

        # Registrar/actualizar usuario/anonimo en colecciones respectivas
        usuario_id = upsert_usuario_o_anon(chat_id, data.nombre or data.usuario, data.celular, data.canal)

        # 2) Asegurar la conversación
        conv_ref = ensure_conversacion(chat_id, usuario_id, data.faq_origen)

        # 3) --- Decisión de turno con LLM + (opcional) embeddings ---
        conv_data = (conv_ref.get().to_dict() or {})
        prev_vec = conv_data.get("last_topic_vec")
        prev_sum = (conv_data.get("last_topic_summary") or "").strip()
        awaiting_confirm = bool(conv_data.get("awaiting_topic_confirm"))

        # Tomamos hasta 2 turnos previos para contexto
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


        topic_change_suspect = False  # se usa luego para construir el prompt
        curr_vec = None               # solo calculamos si hace falta

        if action in ("confirm_new_topic", "new_topic", "continue_topic"):
            # Calculamos embedding SOLO cuando el mensaje es relevante para tema
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
                # Consolidar el candidato como tema vigente
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
            # No reescribimos el vector en cada turno; sólo el resumen.
            # Si aún no hay vector consolidado (primera vez), lo guardamos.
            update_payload = {
                "last_topic_summary": curr_sum,
                "awaiting_topic_confirm": False,
                "ultima_fecha": firestore.SERVER_TIMESTAMP
            }
            if prev_vec is None and curr_vec is not None:
                update_payload["last_topic_vec"] = curr_vec  # primera consolidación
            conv_ref.update(update_payload)

        elif action == "reject_new_topic":
            # Mantener el tema anterior, limpiar candidato
            conv_ref.update({
                "awaiting_topic_confirm": False,
                "candidate_new_topic_summary": None,
                "candidate_new_topic_vec": None,
                "ultima_fecha": firestore.SERVER_TIMESTAMP
            })

        else:
            # greeting_smalltalk o meta: no tocar vectores ni candidatos
            conv_ref.update({"ultima_fecha": firestore.SERVER_TIMESTAMP})

        # 4) Recuperar contexto estilo Wilder y últimos turnos
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
        )

        # 5) LLM (respuesta final)
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.5,    # control de creatividad (0.2-0.7)
            max_tokens=350      # respuesta concisa por diseño del prompt
        )
        texto = completion.choices[0].message.content.strip()

        # 6) Guardar turnos en la conversación (en el arreglo 'mensajes')
        append_mensajes(conv_ref, [
            {"role": "user", "content": data.mensaje},
            {"role": "assistant", "content": texto}
        ])

        # Devolvemos también 'fuentes' (hits RAG) por si quieres auditar en n8n o panel
        return {"respuesta": texto, "fuentes": hits, "chat_id": chat_id}

    except Exception as e:
        # No exponemos stacktrace, solo el mensaje de error
        return {"error": str(e)}


# =========================================================
#  Clasificación y Panel (/clasificar)  — NUEVO BLOQUE
#  *No altera el comportamiento actual de /responder*
#  Se puede llamar asíncronamente desde n8n tras cada respuesta.
# =========================================================

def get_prompt_base() -> str:
    """
    Lee el prompt base desde /configuracion/prompt_inicial.prompt_base
    para usar el mismo "marco" en la clasificación.
    """
    doc = db.collection("configuracion").document("prompt_inicial").get()
    if doc.exists:
        data = doc.to_dict() or {}
        return data.get("prompt_base", "")
    return ""


def read_last_user_and_bot(chat_id: str) -> Tuple[str, str, dict]:
    """
    Toma del arreglo 'mensajes' el último mensaje del usuario y la última respuesta del bot.
    Devuelve (ultimo_usuario, ultima_respuesta_bot, data_conversacion).
    """
    conv_ref = db.collection("conversaciones").document(chat_id)
    snap = conv_ref.get()
    if not snap.exists:
        return "", "", {}

    data = snap.to_dict() or {}
    mensajes = data.get("mensajes", [])

    ultimo_usuario = ""
    ultima_respuesta = ""

    # Recorremos de atrás hacia adelante para encontrar los últimos de cada tipo
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
    """
    Prompt de clasificación:
    - Pide SOLO un JSON con: categoria_general, titulo_propuesta, tono_detectado, palabras_clave
    - Limita el título a 70 caracteres.
    """
    system_msg = (
        f"{prompt_base}\n\n"
        "TAREA: Clasifica la propuesta del ciudadano y devuelve SOLO un JSON válido:\n"
        '{"categoria_general":"...", "titulo_propuesta":"...", "tono_detectado":"positivo|crítico|preocupación|propositivo", "palabras_clave":["...", "..."]}\n'
        "Reglas: título ≤ 70 caracteres, categoría clara y general, sin texto extra."
    )
    user_msg = (
        "Último mensaje del ciudadano:\n"
        f"{u}\n\n"
        "Última respuesta del bot:\n"
        f"{a}\n\n"
        "Devuelve el JSON ahora."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


def update_panel_resumen(categoria: str, tono: str, titulo: str, usuario_id: str):
    """
    Actualiza el documento 'panel_resumen/global':
    - total_conversaciones (+1)
    - propuestas_por_categoria[categoria] (+1)
    - resumen_tono[tono] (+1)
    - ultimas_propuestas (ArrayUnion con item {titulo, usuario_id, categoria, fecha})
    """
    panel_ref = db.collection("panel_resumen").document("global")

    # Incrementos atómicos (no requieren leer el valor anterior)
    panel_ref.set({
        "total_conversaciones": Increment(1),
        "propuestas_por_categoria": {categoria: Increment(1)},
        "resumen_tono": {tono: Increment(1)}
    }, merge=True)

    # Añadir una entrada a 'ultimas_propuestas'
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
    """
    Clasifica la conversación indicada por chat_id.
    - Actualiza: categoria_general, titulo_propuesta, tono_detectado.
    - 'panel_resumen/global': solo incrementa la 1ª vez (o cuando contabilizar=True).
    - Si faq_origen existe, registra faq_logs.
    """
    try:
        chat_id = body.chat_id

        # 1) Leer últimos aportes para dar contexto a la clasificación
        ultimo_u, ultima_a, conv_data = read_last_user_and_bot(chat_id)
        if not ultimo_u:
            return {"ok": False, "mensaje": "No hay mensajes para clasificar."}
        
        # --- Saltar clasificación si el último turno es saludo/pequeña charla/meta ---
        decision_cls = llm_decide_turn(
            last_topic_summary=(conv_data.get("last_topic_summary") or ""),
            awaiting_confirm=bool(conv_data.get("awaiting_topic_confirm")),
            last_two_turns=[{"role":"user","content": ultimo_u},
                            {"role":"assistant","content": ultima_a}],
            current_text=ultimo_u
        )
        if decision_cls.get("action") in ("greeting_smalltalk", "meta"):
            return {
                "ok": True,
                "skipped": True,
                "reason": "saludo_o_meta_detectado_por_llm"
            }


        # 2) Prompt base desde configuración + mensajes de clasificación
        prompt_base = get_prompt_base()
        msgs = build_messages_for_classify(prompt_base, ultimo_u, ultima_a)

        # 3) Llamada al modelo (esperamos un JSON plano)
        model_cls = os.getenv("OPENAI_MODEL_CLASSIFY", OPENAI_MODEL)
        out = client.chat.completions.create(
            model=model_cls,
            messages=msgs,
            temperature=0.2,
            max_tokens=300
        ).choices[0].message.content.strip()

        # 4) Parseo del JSON devuelto
        data = json.loads(out)
        categoria = data.get("categoria_general") or data.get("categoria") or "General"
        titulo    = data.get("titulo_propuesta") or data.get("titulo") or "Propuesta ciudadana"
        tono      = data.get("tono_detectado") or "neutral"
        palabras  = data.get("palabras_clave", [])

        # 5) Decidir si se debe contabilizar en panel
        awaiting = bool(conv_data.get("awaiting_topic_confirm"))   # <— NUEVO: hay cambio de tema pendiente?
        ya_contado = bool(conv_data.get("panel_contabilizado"))    # ya se contó alguna vez

        if body.contabilizar is None:
            # Por defecto NO contamos si estamos esperando confirmación por cambio de tema;
            # si no estamos esperando, contamos solo la primera vez.
            debe_contar = (not awaiting) and (not ya_contado)
        else:
            # Respeta lo que indique n8n explícitamente
            debe_contar = bool(body.contabilizar)

        # 6) Actualizar la conversación con los campos de clasificación
        conv_ref = db.collection("conversaciones").document(chat_id)
        conv_ref.set({
            "categoria_general": categoria,
            "titulo_propuesta": titulo,
            "tono_detectado": tono,
            "ultima_fecha": firestore.SERVER_TIMESTAMP
        }, merge=True)

        # --- Historial de temas clasificados ---
        hist_last = (conv_data.get("topics_history") or [])
        last_item = hist_last[-1] if hist_last else {}

        should_append = (
            not last_item or
            last_item.get("categoria") != categoria or
            last_item.get("titulo") != titulo or
            last_item.get("tono") != tono
        )

        if should_append:
            conv_ref.set({
                "topics_history": firestore.ArrayUnion([{
                    "categoria": categoria,
                    "titulo": titulo,
                    "tono": tono,
                    "fecha": firestore.SERVER_TIMESTAMP
                }])
            }, merge=True)


        # 7) Registrar/asegurar la categoría
        db.collection("categorias_tematicas").document(categoria).set({
            "nombre": categoria
        }, merge=True)

        # 8) Panel resumen: sólo si corresponde
        usuario_id = conv_data.get("usuario_id", chat_id)
        if debe_contar:
            update_panel_resumen(categoria, tono, titulo, usuario_id)
            # Marcar que esta conversación ya fue contabilizada al menos una vez
            conv_ref.set({"panel_contabilizado": True}, merge=True)

        # 9) Log de FAQ si aplica (no afecta el conteo)
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
#  Arranque local (uvicorn)
# =========================================================

if __name__ == "__main__":
    import uvicorn
    # Puerto 10000 para mantener compatibilidad con Render y tu render.yml
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
