# ============================
#  WilderBot API - VERSIÓN OPTIMIZADA
#  Sistema de capas para minimizar uso de LLM
# ============================

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from typing import Optional, List, Dict, Any
from google.cloud.firestore_v1 import Increment

import json
import os
from dotenv import load_dotenv
import re

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
#  Utils Básicos
# =========================================================

def _normalize_text(t: str) -> str:
    """Normaliza texto para comparaciones."""
    t = t.lower()
    t = (t.replace("á","a").replace("é","e").replace("í","i")
           .replace("ó","o").replace("ú","u").replace("ü","u"))
    t = re.sub(r"[^a-zñ0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def limit_sentences(text: str, max_sentences: int = 3) -> str:
    """Limita respuesta a N frases."""
    parts = re.split(r'(?<=[\.\?!])\s+', text.strip())
    out = " ".join([p for p in parts if p][:max_sentences]).strip()
    return out or text

def _titlecase(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).title()

def _clean_barrio_fragment(s: str) -> str:
    s = re.split(r"\s+(?:para|por|que|donde|con)\b|[,.;]|$", s, maxsplit=1, flags=re.IGNORECASE)[0]
    return _titlecase(s)

def remove_redundant_greetings(texto: str, historial: List[Dict[str, str]]) -> str:
    """Elimina saludos si ya hay conversación previa."""
    if not historial:
        return texto
    
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
    
    return texto if len(texto) >= 10 else "Claro, déjame ayudarte con eso."

# =========================================================
#  CAPA 1: Detección Rápida (Sin LLM)
# =========================================================

class MessageClassifier:
    """Clasifica mensajes usando heurísticas rápidas."""
    
    GREETING_WORDS = {"hola", "holaa", "buenas", "buenos dias", "saludos"}
    QUESTION_WORDS = {"que", "qué", "como", "cómo", "cuando", "cuándo", 
                     "donde", "dónde", "por que", "por qué", "cual", "cuál"}
    PROPOSAL_VERBS = {"arregl", "mejor", "constru", "instal", "crear", 
                      "paviment", "ilumin", "repar", "pint", "adecu", "señaliz"}
    PROPOSAL_OBJECTS = {"alumbrado", "luminaria", "parque", "semaforo", "cancha", 
                       "via", "calle", "anden", "acera", "juegos", "polideportivo"}
    PROPOSAL_KEYWORDS = {"propongo", "propuesta", "sugerencia", "mi idea", "quiero proponer"}
    
    @staticmethod
    def is_plain_greeting(text: str) -> bool:
        """Detecta saludos simples sin contenido."""
        if not text or len(text) > 40:
            return False
        
        t = _normalize_text(text)
        has_greeting = any(kw in t for kw in MessageClassifier.GREETING_WORDS)
        has_topic = any(w in t for w in MessageClassifier.PROPOSAL_OBJECTS)
        
        return has_greeting and not has_topic
    
    @staticmethod
    def is_question(text: str) -> bool:
        """Detecta si es una pregunta."""
        if "?" in text:
            return True
        
        t = _normalize_text(text)
        return any(kw in t for kw in MessageClassifier.QUESTION_WORDS)
    
    @staticmethod
    def is_proposal_intent(text: str) -> bool:
        """Detecta intención de proponer (sin contenido)."""
        t = _normalize_text(text)
        return any(kw in t for kw in MessageClassifier.PROPOSAL_KEYWORDS)
    
    @staticmethod
    def has_proposal_content(text: str) -> bool:
        """Detecta si tiene contenido real de propuesta."""
        t = _normalize_text(text)
        
        # Debe tener VERBO + OBJETO o VERBO + ubicación
        has_verb = any(v in t for v in MessageClassifier.PROPOSAL_VERBS)
        has_object = any(obj in t for obj in MessageClassifier.PROPOSAL_OBJECTS)
        has_location = bool(re.search(r'\b(barrio|sector|comuna|vereda|en\s+[A-Z])\b', text))
        
        # Debe ser suficientemente largo
        if len(t) < 15:
            return False
        
        return (has_verb and has_object) or (has_verb and has_location and len(t) >= 20)
    
    @staticmethod
    def is_refusal(text: str) -> bool:
        """Detecta negación/rechazo."""
        t = _normalize_text(text)
        patterns = [
            r'\b(no\s+quiero|no\s+tengo|todavia\s+no|aun\s+no)\b',
            r'\b(mas\s+tarde|mejor\s+no|olvidalo)\b',
            r'\b(no\s+me\s+gusta\s+dar|no\s+doy\s+mi)\b'
        ]
        return any(re.search(p, t) for p in patterns)

# =========================================================
#  CAPA 2: Extracción de Datos (Regex Primero)
# =========================================================

class DataExtractor:
    """Extrae datos de contacto usando regex cuando es posible."""
    
    @staticmethod
    def extract_name(text: str) -> Optional[str]:
        """Extrae nombre con regex."""
        m = re.search(r'\b(?:soy|me llamo|mi nombre es)\s+([A-Za-záéíóúñ ]{2,40})', text, flags=re.IGNORECASE)
        if m:
            nombre = m.group(1).strip(" .,")
            # Validar que no sea palabra de discurso
            if _normalize_text(nombre) not in MessageClassifier.GREETING_WORDS:
                return nombre
        return None
    
    @staticmethod
    def extract_phone(text: str) -> Optional[str]:
        """Extrae teléfono."""
        m = re.search(r'(\+?\d[\d\s\-]{7,16}\d)', text)
        if not m:
            return None
        tel = re.sub(r'\D', '', m.group(1))
        tel = re.sub(r'^(?:00)?57', '', tel)
        return tel if 8 <= len(tel) <= 12 else None
    
    @staticmethod
    def extract_barrio_residence(text: str) -> Optional[str]:
        """Extrae barrio de residencia."""
        m = re.search(r'\b(?:vivo|resido)\s+en\s+(?:el\s+)?(?:barrio\s+)?([A-Za-záéíóúñ0-9 \-]{2,50})', 
                     text, flags=re.IGNORECASE)
        return _clean_barrio_fragment(m.group(1)) if m else None
    
    @staticmethod
    def extract_barrio_project(text: str) -> Optional[str]:
        """Extrae barrio del proyecto."""
        m = re.search(r'\ben\s+(?:el\s+)?barrio\s+([A-Za-záéíóúñ0-9 \-]{2,50})', 
                     text, flags=re.IGNORECASE)
        return _clean_barrio_fragment(m.group(1)) if m else None
    
    @staticmethod
    def extract_all_contact(text: str) -> Dict[str, Optional[str]]:
        """Extrae todos los datos de contacto de una vez."""
        return {
            "nombre": DataExtractor.extract_name(text),
            "telefono": DataExtractor.extract_phone(text),
            "barrio": DataExtractor.extract_barrio_residence(text)
        }

# =========================================================
#  Contexto de Conversación
# =========================================================

class ConversationContext:
    """Contexto completo de la conversación."""
    
    def __init__(self, conv_data: dict, mensaje: str):
        self.mensaje = mensaje
        self.mensaje_norm = _normalize_text(mensaje)
        
        # Estado del flujo
        self.proposal_collected = bool(conv_data.get("proposal_collected"))
        self.argument_collected = bool(conv_data.get("argument_collected"))
        self.contact_collected = bool(conv_data.get("contact_collected"))
        self.proposal_requested = bool(conv_data.get("proposal_requested"))
        self.argument_requested = bool(conv_data.get("argument_requested"))
        self.contact_requested = bool(conv_data.get("contact_requested"))
        
        # Datos recopilados
        self.contact_info = conv_data.get("contact_info") or {}
        self.project_location = conv_data.get("project_location")
        self.current_proposal = conv_data.get("current_proposal")
        
        # Historial (últimos 8 mensajes)
        mensajes = conv_data.get("mensajes", [])
        self.historial = [
            {"role": m["role"], "content": m["content"]}
            for m in mensajes[-8:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        
        # Detección de referencia
        self.has_reference = self._detect_reference()
    
    def _detect_reference(self) -> bool:
        """Detecta si hace referencia a mensaje previo."""
        referencias = ["eso", "esa", "ese", "lo que", "la que", "el que", "sobre eso"]
        return any(ref in self.mensaje_norm for ref in referencias)
    
    def get_missing_contact_fields(self) -> List[str]:
        """Qué datos de contacto faltan."""
        missing = []
        if not self.contact_info.get("nombre"):
            missing.append("nombre")
        if not self.contact_info.get("barrio"):
            missing.append("barrio")
        if not self.contact_info.get("telefono"):
            missing.append("celular")
        return missing
    
    @property
    def in_proposal_flow(self) -> bool:
        """Está en flujo de propuesta."""
        return (self.proposal_collected or self.proposal_requested or 
                self.argument_requested)

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
    """Crea o actualiza usuario/anónimo."""
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
    """Asegura que existe el documento de conversación."""
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
            "proposal_requested": False,
            "proposal_collected": False,
            "argument_requested": False,
            "argument_collected": False,
            "contact_requested": False,
            "contact_collected": False,
            "current_proposal": None,
            "contact_info": {"nombre": None, "barrio": None, "telefono": None},
            "project_location": None,
            "proposal_nudge_count": 0,
        })
    else:
        conv_ref.update({"ultima_fecha": firestore.SERVER_TIMESTAMP})
    return conv_ref

def append_mensajes(conv_ref, nuevos: List[Dict[str, Any]]):
    """Agrega mensajes y actualiza resúmenes."""
    snap = conv_ref.get()
    data = snap.to_dict() or {}
    arr = data.get("mensajes", [])
    arr.extend(nuevos)
    conv_ref.update({"mensajes": arr, "ultima_fecha": firestore.SERVER_TIMESTAMP})
    
    # Resumen preliminar (async, no crítico)
    try:
        resumen = summarize_brief(arr)
        conv_ref.update({"resumen": resumen})
    except:
        pass
    
    # Resumen completo
    try:
        resumen_completo = build_detailed_summary(data, arr)
        conv_ref.update({"resumen_completo": resumen_completo})
    except:
        pass

def summarize_brief(mensajes: List[Dict], max_chars: int = 100) -> str:
    """Resumen breve para vista rápida."""
    if not mensajes:
        return ""
    
    # Tomar primeros mensajes del usuario
    user_msgs = [m.get("content", "") for m in mensajes[:5] if m.get("role") == "user"]
    if not user_msgs:
        return ""
    
    texto = " ".join(user_msgs)
    texto = re.sub(r"\+?\d[\d\s\-]{6,}", "[número]", texto)
    return texto[:max_chars]

def build_detailed_summary(conv_data: dict, mensajes: List[Dict]) -> dict:
    """Resumen completo estructurado para 'Ver más'."""
    summary = {
        "tema_principal": "",
        "consultas": [],
        "propuesta": None,
        "argumento": None,
        "ubicacion": None,
        "contacto": {"nombre": None, "barrio": None, "telefono": None},
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
        summary["consultas"] = [t for t in titulos if "ley" in t.lower() or "proyecto" in t.lower()]
    
    # Propuesta
    propuesta = conv_data.get("current_proposal")
    if propuesta:
        summary["propuesta"] = propuesta[:120] + ("..." if len(propuesta) > 120 else "")
    
    # Argumento
    if conv_data.get("argument_collected"):
        for m in mensajes:
            if m.get("role") == "user" and len(m.get("content", "")) > 30:
                summary["argumento"] = m["content"][:120]
                break
    
    # Ubicación
    summary["ubicacion"] = conv_data.get("project_location")
    
    # Contacto
    contact_info = conv_data.get("contact_info", {})
    if contact_info:
        summary["contacto"] = contact_info
    
    # Estado
    if conv_data.get("contact_collected"):
        summary["estado"] = "completado"
    elif conv_data.get("argument_collected"):
        summary["estado"] = "argumento_recibido"
    elif conv_data.get("proposal_collected"):
        summary["estado"] = "propuesta_recibida"
    else:
        summary["estado"] = "iniciado"
    
    # Historial resumido (últimos 5 intercambios)
    recent = mensajes[-10:] if len(mensajes) > 10 else mensajes
    for m in recent:
        role_display = "Usuario" if m.get("role") == "user" else "Asistente"
        content = m.get("content", "")[:100]
        summary["historial_resumido"].append({
            "rol": role_display,
            "mensaje": content + ("..." if len(m.get("content", "")) > 100 else "")
        })
    
    return summary

# =========================================================
#  RAG
# =========================================================

def rag_search(query: str, top_k: int = 5) -> List[Dict]:
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

def validate_rag_relevance(hits: List[Dict]) -> bool:
    """Valida si RAG tiene resultados útiles."""
    if not hits:
        return False
    
    best_score = max((h.get("score", 0) for h in hits), default=0)
    return best_score >= 0.5 or len(hits) >= 3

def reformulate_query_with_context(query: str, historial: List[Dict]) -> str:
    """Reformula query si es muy corta y hay contexto."""
    if len(query.split()) > 8 or not historial:
        return query
    
    # Buscar tema específico en historial reciente
    for msg in reversed(historial[-6:]):
        if msg["role"] == "assistant":
            content = msg.get("content", "")
            ley_match = re.search(r'Ley\s+(\d+)', content, re.IGNORECASE)
            if ley_match:
                return f"Ley {ley_match.group(1)} detalles"
    
    return query

# =========================================================
#  CAPA 3: Generación LLM (Solo cuando es necesario)
# =========================================================

def generate_response_llm(ctx: ConversationContext, rag_hits: List[Dict], 
                         tipo: str = "consulta") -> str:
    """Genera respuesta con LLM."""
    
    contexto_rag = "\n".join([f"- {h['texto']}" for h in rag_hits if h.get("texto")])
    
    system_msg = (
        "Eres el ASISTENTE de Wilder Escobar, Representante a la Cámara.\n\n"
        "REGLAS:\n"
        "1. NO te presentes como Wilder\n"
        "2. NO saludes si ya hay historial\n"
        "3. Usa SOLO información del contexto proporcionado\n"
        "4. Si no está en el contexto, di: 'No tengo esa información específica'\n"
        "5. Máximo 3 frases\n\n"
    )
    
    if tipo == "consulta":
        system_msg += (
            "CONSULTA: Responde SOLO con información del contexto.\n"
            "NO inventes datos.\n"
        )
    
    msgs = [{"role": "system", "content": system_msg}]
    
    if ctx.historial:
        msgs.extend(ctx.historial[-6:])
    
    msgs.append({
        "role": "user",
        "content": f"CONTEXTO:\n{contexto_rag}\n\nPREGUNTA:\n{ctx.mensaje}"
    })
    
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.1,
            max_tokens=280
        )
        
        texto = completion.choices[0].message.content.strip()
        texto = limit_sentences(texto, 3)
        texto = remove_redundant_greetings(texto, ctx.historial)
        
        return texto
        
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return "Disculpa, tuve un problema. ¿Puedes reformular?"

# =========================================================
#  HELPERS DE RESPUESTA
# =========================================================

def build_contact_request(missing: List[str]) -> str:
    """Construye solicitud de datos faltantes."""
    etiquetas = {"nombre": "tu nombre", "barrio": "tu barrio", "celular": "celular"}
    pedir = [etiquetas[m] for m in missing if m in etiquetas]
    
    if not pedir:
        return "Perfecto, con estos datos escalamos tu propuesta."
    
    if len(pedir) == 1:
        return f"¿Me compartes {pedir[0]}?"
    
    frase = ", ".join(pedir[:-1]) + " y " + pedir[-1]
    return f"¿Me compartes {frase}?"

def build_project_location_request() -> str:
    """Pide barrio del proyecto."""
    return "Para ubicar el caso: ¿en qué barrio sería exactamente el proyecto?"

# =========================================================
#  ENDPOINT PRINCIPAL
# =========================================================

@app.post("/responder")
async def responder(data: Entrada):
    """Endpoint principal optimizado con sistema de capas."""
    try:
        # Setup básico
        chat_id = data.chat_id or f"web_{os.urandom(4).hex()}"
        usuario_id = upsert_usuario_o_anon(chat_id, data.nombre or data.usuario, 
                                           data.celular, data.canal)
        conv_ref = ensure_conversacion(chat_id, usuario_id, data.faq_origen, data.canal)
        conv_data = conv_ref.get().to_dict() or {}
        
        ctx = ConversationContext(conv_data, data.mensaje)
        
        print(f"\n{'='*60}")
        print(f"[MSG] {data.mensaje[:80]}")
        print(f"[CTX] Flow={ctx.in_proposal_flow} | Hist={len(ctx.historial)}")
        
        # ═════════════════════════════════════════════════════════
        # CAPA 1: RESPUESTAS DETERMINÍSTICAS (SIN LLM)
        # ═════════════════════════════════════════════════════════
        
        # 1A. Saludo inicial
        if not ctx.historial and MessageClassifier.is_plain_greeting(data.mensaje):
            print("[L1] ✓ Saludo inicial")
            texto = BOT_INTRO_TEXT
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto}
            ])
            return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
        
        # 1B. Rechazo de datos personales
        if ctx.contact_requested and MessageClassifier.is_refusal(data.mensaje):
            refusal_count = int(conv_data.get("contact_refusal_count", 0))
            
            if refusal_count == 0:
                texto = PRIVACY_REPLY + " ¿Me compartes tus datos?"
                conv_ref.update({"contact_refusal_count": 1})
            else:
                texto = "Entiendo tu decisión. ¡Que tengas buen día!"
                conv_ref.update({
                    "contact_requested": False,
                    "contact_refusal_count": 0
                })
            
            append_mensajes(conv_ref, [
                {"role": "user", "content": data.mensaje},
                {"role": "assistant", "content": texto}
            ])
            return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
        
        # ═════════════════════════════════════════════════════════
        # CAPA 2: EXTRACCIÓN DE DATOS (REGEX)
        # ═════════════════════════════════════════════════════════
        
        extracted = DataExtractor.extract_all_contact(data.mensaje)
        
        if any(extracted.values()):
            current_info = ctx.contact_info.copy()
            
            if extracted["nombre"]:
                current_info["nombre"] = extracted["nombre"]
            if extracted["telefono"]:
                current_info["telefono"] = extracted["telefono"]
            if extracted["barrio"]:
                current_info["barrio"] = extracted["barrio"]
            
            conv_ref.update({"contact_info": current_info})
            
            if current_info.get("telefono"):
                conv_ref.update({"contact_collected": True})
            
            ctx.contact_info = current_info
            print(f"[L2] ✓ Datos extraídos: {extracted}")
        
        # Extraer ubicación del proyecto
        proj_loc = DataExtractor.extract_barrio_project(data.mensaje)
        if proj_loc:
            conv_ref.update({"project_location": proj_loc})
            ctx.project_location = proj_loc
            print(f"[L2] ✓ Ubicación proyecto: {proj_loc}")
        
        # ═════════════════════════════════════════════════════════
        # FLUJO DE PROPUESTAS
        # ═════════════════════════════════════════════════════════
        
        if MessageClassifier.is_proposal_intent(data.mensaje) or ctx.in_proposal_flow:
            
            # FASE 1: Capturar propuesta
            if not ctx.proposal_collected:
                
                # Caso A: Dice que quiere proponer (intención pura)
                if (MessageClassifier.is_proposal_intent(data.mensaje) and 
                    not MessageClassifier.has_proposal_content(data.mensaje)):
                    
                    conv_ref.update({
                        "proposal_requested": True,
                        "proposal_nudge_count": 0
                    })
                    texto = "¡Perfecto! ¿Cuál es tu propuesta? Cuéntamela en 1-2 frases y el barrio."
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                # Caso B: Da propuesta con contenido
                if MessageClassifier.has_proposal_content(data.mensaje):
                    propuesta = limit_sentences(data.mensaje, 2)
                    
                    conv_ref.update({
                        "current_proposal": propuesta,
                        "proposal_collected": True,
                        "proposal_requested": True,
                        "argument_requested": True,
                        "proposal_nudge_count": 0
                    })
                    
                    texto = "Excelente idea. ¿Por qué sería importante?"
                    print(f"[PROP] ✓ Guardada: {propuesta[:50]}")
                    
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                
                # Caso C: Ya pedimos propuesta pero no llegó (NUDGES)
                if ctx.proposal_requested:
                    
                    # Usuario rechazó
                    if MessageClassifier.is_refusal(data.mensaje):
                        conv_ref.update({
                            "proposal_requested": False,
                            "proposal_nudge_count": 0
                        })
                        texto = "Perfecto. Cuando la tengas, cuéntamela."
                        append_mensajes(conv_ref, [
                            {"role": "user", "content": data.mensaje},
                            {"role": "assistant", "content": texto}
                        ])
                        return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                    
                    # Usuario hizo pregunta
                    if MessageClassifier.is_question(data.mensaje):
                        conv_ref.update({
                            "proposal_requested": False,
                            "proposal_nudge_count": 0
                        })
                        texto = "¿Prefieres que responda tu pregunta o seguimos con la propuesta?"
                        append_mensajes(conv_ref, [
                            {"role": "user", "content": data.mensaje},
                            {"role": "assistant", "content": texto}
                        ])
                        return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                    
                    # Nudge escalonado
                    nudges = int(conv_data.get("proposal_nudge_count", 0)) + 1
                    conv_ref.update({"proposal_nudge_count": nudges})
                    
                    if nudges == 1:
                        texto = "Claro. ¿Cuál es tu propuesta? 1-2 frases y el barrio."
                    elif nudges == 2:
                        texto = "Para ayudarte: escribe la propuesta en 1-2 frases (ej: 'Arreglar luminarias del parque San José')."
                    else:
                        conv_ref.update({
                            "proposal_requested": False,
                            "proposal_nudge_count": 0
                        })
                        texto = "Todo bien. Si prefieres, dime tu pregunta y te ayudo."
                    
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
            
            # FASE 2: Capturar argumento
            if ctx.proposal_collected and not ctx.argument_collected:
                
                # Validar si es argumento
                if len(_normalize_text(data.mensaje)) >= 20:
                    conv_ref.update({
                        "argument_collected": True,
                        "contact_requested": True
                    })
                    
                    missing = ctx.get_missing_contact_fields()
                    if not ctx.project_location:
                        missing.append("project_location")
                    
                    if missing == ["project_location"]:
                        texto = build_project_location_request()
                    elif missing:
                        texto = build_contact_request(missing)
                    else:
                        texto = "Perfecto, con estos datos escalamos tu propuesta."
                    
                    print(f"[ARG] ✓ Argumento recibido")
                    
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
                else:
                    # Pedir argumento de nuevo
                    texto = "¿Por qué sería importante?"
                    append_mensajes(conv_ref, [
                        {"role": "user", "content": data.mensaje},
                        {"role": "assistant", "content": texto}
                    ])
                    return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
            
            # FASE 3: Capturar contacto
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
                    nombre = ctx.contact_info.get("nombre", "")
                    texto = (f"Gracias, {nombre}. " if nombre else "Gracias. ")
                    texto += "Con estos datos escalamos el caso."
                    conv_ref.update({"contact_collected": True})
                
                append_mensajes(conv_ref, [
                    {"role": "user", "content": data.mensaje},
                    {"role": "assistant", "content": texto}
                ])
                return {"respuesta": texto, "fuentes": [], "chat_id": chat_id}
        
        # ═════════════════════════════════════════════════════════
        # CONSULTAS: RAG + LLM
        # ═════════════════════════════════════════════════════════
        
        query = data.mensaje
        if ctx.has_reference:
            query = reformulate_query_with_context(data.mensaje, ctx.historial)
        
        print(f"[RAG] Query: '{query}'")
        hits = rag_search(query, top_k=5)
        print(f"[RAG] Hits: {len(hits)} | Best score: {hits[0].get('score', 0) if hits else 0:.3f}")
        
        if not validate_rag_relevance(hits):
            texto = "No tengo información específica sobre eso. ¿Hay algo más en lo que pueda ayudarte?"
        else:
            texto = generate_response_llm(ctx, hits, tipo="consulta")
        
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
#  ENDPOINT CLASIFICACIÓN
# =========================================================

@app.post("/clasificar")
async def clasificar(body: ClasificarIn):
    """Clasifica conversación."""
    try:
        conv_ref = db.collection("conversaciones").document(body.chat_id)
        snap = conv_ref.get()
        
        if not snap.exists:
            return {"ok": False, "mensaje": "Conversación no encontrada"}
        
        conv_data = snap.to_dict() or {}
        mensajes = conv_data.get("mensajes", [])
        
        # Obtener último mensaje del usuario
        ultimo_usuario = ""
        for m in reversed(mensajes):
            if m.get("role") == "user":
                ultimo_usuario = m.get("content", "")
                break
        
        if not ultimo_usuario:
            return {"ok": False, "mensaje": "No hay mensajes"}
        
        # Detectar tipo
        propuesta = conv_data.get("current_proposal")
        
        if propuesta:
            tipo = "propuesta"
            texto = propuesta
        elif MessageClassifier.is_question(ultimo_usuario):
            tipo = "consulta"
            texto = ultimo_usuario
        else:
            return {"ok": True, "skipped": True}
        
        print(f"[CLASIF] Tipo: {tipo}")
        
        # Clasificar con LLM
        if tipo == "consulta":
            sys = (
                "Clasifica esta CONSULTA.\n"
                "JSON:\n"
                "{\n"
                '  "categoria_general": "Consulta",\n'
                '  "titulo_propuesta": "[Tema en 5-8 palabras]",\n'
                '  "tono_detectado": "neutral"\n'
                "}\n"
            )
        else:
            ubicacion = conv_data.get("project_location") or ""
            sys = (
                "Clasifica esta PROPUESTA.\n"
                "JSON:\n"
                "{\n"
                '  "categoria_general": "[Infraestructura Urbana|Seguridad|Movilidad|Educación|Salud|Vivienda|Empleo|Medio Ambiente]",\n'
                '  "titulo_propuesta": "[Verbo + Qué + Dónde, máx 60 chars]",\n'
                '  "tono_detectado": "propositivo"\n'
                "}\n"
            )
            texto = f"Propuesta: {texto}\nUbicación: {ubicacion}" if ubicacion else f"Propuesta: {texto}"
        
        try:
            out = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": texto}
                ],
                temperature=0.2,
                max_tokens=150
            ).choices[0].message.content.strip()
            
            out = out.replace("```json", "").replace("```", "").strip()
            data = json.loads(out)
        except:
            data = {
                "categoria_general": "Consulta" if tipo == "consulta" else "General",
                "titulo_propuesta": "Sin título",
                "tono_detectado": "neutral"
            }
        
        categoria = data.get("categoria_general", "General")
        titulo = data.get("titulo_propuesta", "Sin título")
        tono = data.get("tono_detectado", "neutral")
        
        # Guardar (acumulativo)
        categorias_existentes = conv_data.get("categoria_general") or []
        titulos_existentes = conv_data.get("titulo_propuesta") or []
        
        if isinstance(categorias_existentes, str):
            categorias_existentes = [categorias_existentes]
        if isinstance(titulos_existentes, str):
            titulos_existentes = [titulos_existentes]
        
        if categoria not in categorias_existentes:
            categorias_existentes.append(categoria)
        
        titulo_norm = _normalize_text(titulo)
        if titulo_norm not in [_normalize_text(t) for t in titulos_existentes]:
            titulos_existentes.append(titulo)
        
        conv_ref.update({
            "categoria_general": categorias_existentes,
            "titulo_propuesta": titulos_existentes,
            "tono_detectado": tono,
        })
        
        print(f"[CLASIF] ✓ {categoria} | {titulo}")
        
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
        print(f"[CLASIF] ❌ {e}")
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)