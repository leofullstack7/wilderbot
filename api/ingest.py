from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
from services.clients import db
from services.pine import upsert_chunks, delete_by_doc_id
from services.chunkers.docx_chunker import parse_docx
from services.chunkers.xlsx_chunker import parse_xlsx
from services.chunkers.pdf_chunker import parse_pdf
from services.chunkers.text_chunker import parse_text_note
from utils.ids import new_doc_id, chunk_id

from services.pine import upsert_chunks  # ya existe
from services.pine import get_index 

router = APIRouter(tags=["ingest"])

@router.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    categoria: Optional[str] = Form(default=None),
    publicar: Optional[bool] = Form(default=True),
):
    try:
        content = await file.read()
        name = file.filename or "documento"
        title = name.rsplit(".", 1)[0]

        ext = (name.split(".")[-1] or "").lower()
        if ext == "docx":
            items = parse_docx(content, title)
        elif ext == "xlsx":
            items = parse_xlsx(content, title)
        elif ext == "pdf":
            items = parse_pdf(content, title)
        else:
            return JSONResponse({"ok": False, "error": f"Formato no soportado: .{ext} (usa .docx, .xlsx o .pdf)"}, status_code=400)

        if not items:
            return {"ok": False, "error": "No se encontró texto utilizable en el archivo."}

        doc_id = new_doc_id("doc")
        # agrega metadatos comunes
        for it in items:
            md = it.get("metadata") or {}
            md.update({"doc_id": doc_id, "categoria": categoria or "General", "publicado": bool(publicar)})
            it["metadata"] = md

        # IDs de chunk
        for i, it in enumerate(items):
            it["id"] = chunk_id(doc_id, i)

        # upsert a pinecone (sin namespace para que /responder actual los encuentre)
        stat = upsert_chunks(items)

        # registra maestro en firestore
        db.collection("knowledge_docs").document(doc_id).set({
            "doc_id": doc_id,
            "titulo": title,
            "filename": name,
            "categoria": categoria or "General",
            "publicado": bool(publicar),
            "chunks": stat.get("upserted", 0),
            "fecha": db.SERVER_TIMESTAMP if hasattr(db, "SERVER_TIMESTAMP") else None
        }, merge=True)

        return {"ok": True, "doc_id": doc_id, "chunks": stat.get("upserted", 0)}

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.post("/ingest/text")
async def ingest_text(
    titulo: str = Form(...),
    contenido: str = Form(...),
    categoria: Optional[str] = Form(default=None),
    publicar: Optional[bool] = Form(default=True),
):
    try:
        items = parse_text_note(titulo, contenido)
        if not items:
            return {"ok": False, "error": "El contenido está vacío."}

        doc_id = new_doc_id("note")
        for i, it in enumerate(items):
            md = it.get("metadata") or {}
            md.update({"doc_id": doc_id, "categoria": categoria or "General", "publicado": bool(publicar)})
            it["metadata"] = md
            it["id"] = chunk_id(doc_id, i)

        stat = upsert_chunks(items)

        db.collection("knowledge_docs").document(doc_id).set({
            "doc_id": doc_id,
            "titulo": titulo,
            "filename": None,
            "categoria": categoria or "General",
            "publicado": bool(publicar),
            "chunks": stat.get("upserted", 0),
            "fecha": db.SERVER_TIMESTAMP if hasattr(db, "SERVER_TIMESTAMP") else None
        }, merge=True)

        return {"ok": True, "doc_id": doc_id, "chunks": stat.get("upserted", 0)}

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.get("/ingest/status")
async def ingest_status(doc_id: str):
    try:
        snap = db.collection("knowledge_docs").document(doc_id).get()
        if not snap.exists:
            return {"ok": False, "error": "doc_id no encontrado"}
        return {"ok": True, "doc": snap.to_dict()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    

@router.delete("/ingest/delete")
async def ingest_delete(doc_id: str, namespace: Optional[str] = None):
    """
    Elimina el documento de Firestore y purga sus chunks en Pinecone por metadata.doc_id
    """
    try:
        # 1) Pinecone: borrar por filtro (por doc_id)
        try:
            pine_stat = delete_by_doc_id(doc_id, namespace=namespace)
            pine_err = None
        except Exception as e:
            pine_stat = None
            pine_err = str(e)

        # 2) Firestore: borra el documento maestro
        try:
            db.collection("knowledge_docs").document(doc_id).delete()
        except Exception as e:
            return JSONResponse(
                {"ok": False, "error": f"Firestore: {e}", "pinecone_error": pine_err},
                status_code=500
            )

        return {"ok": True, "pinecone_error": pine_err, "pinecone": pine_stat}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
