import React, { useCallback, useRef, useState, useEffect } from "react";
// Listado de documentos
import { listenKnowledgeDocs, deleteKnowledgeDoc } from "../lib/firestoreQueries";

const API = import.meta.env.VITE_API_URL || "http://localhost:10000";
const CATS = ["Vida Personal Wilder","General", "Leyes","Movilidad", "Educación", "Salud", "Seguridad", "Vivienda", "Empleo"];

export default function Conocimiento() {
    // UI state
    const [tab, setTab] = useState("file"); // "file" | "text"
    const [busy, setBusy] = useState(false);
    const [msg, setMsg] = useState({ type: "info", text: "Listo para subir o pegar contenido." });
    const [lastResult, setLastResult] = useState(null);
    const [docs, setDocs] = useState([]);
    useEffect(() => {
        const off = listenKnowledgeDocs(setDocs);
        return () => off && off();
    }, []);


    // File form
    const [file, setFile] = useState(null);
    const [categoriaF, setCategoriaF] = useState("General");
    const [publicarF, setPublicarF] = useState(true);
    const dropRef = useRef(null);

    // Text form
    const [titulo, setTitulo] = useState("");
    const [contenido, setContenido] = useState("");
    const [categoriaT, setCategoriaT] = useState("General");
    const [publicarT, setPublicarT] = useState(true);

    // Helpers
    const formatBytes = (bytes = 0) => {
        if (bytes === 0) return "0 B";
        const k = 1024;
        const sizes = ["B", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    };

    const setAlert = (type, text) => setMsg({ type, text });

    const resetFileForm = () => {
        setFile(null);
        setCategoriaF("General");
        setPublicarF(true);
    };
    const resetTextForm = () => {
        setTitulo("");
        setContenido("");
        setCategoriaT("General");
        setPublicarT(true);
    };

    // Drag & drop
    const onDragOver = (e) => {
        e.preventDefault();
        dropRef.current?.classList.add("dz--over");
    };
    const onDragLeave = (e) => {
        e.preventDefault();
        dropRef.current?.classList.remove("dz--over");
    };
    const onDrop = (e) => {
        e.preventDefault();
        dropRef.current?.classList.remove("dz--over");
        if (!e.dataTransfer.files?.length) return;
        const f = e.dataTransfer.files[0];
        if (!/\.(docx|xlsx|pdf)$/i.test(f.name)) {
            setAlert("warn", "Formato no soportado. Aceptamos .docx, .xlsx y .pdf");
            return;
        }
        setFile(f);
    };

    // Upload handlers
    const onUpload = async (e) => {
        e.preventDefault();
        if (!file) return setAlert("warn", "Selecciona un archivo (.docx / .xlsx / .pdf)");
        setBusy(true);
        setAlert("info", "Subiendo y procesando…");

        try {
            const fd = new FormData();
            fd.append("file", file);
            fd.append("categoria", categoriaF);
            fd.append("publicar", publicarF ? "true" : "false");

            const res = await fetch(`${API}/ingest/upload`, { method: "POST", body: fd });
            const data = await res.json();

            if (data.ok) {
                setLastResult({ source: "file", ...data });
                setAlert("success", `Subido ✔  id: ${data.doc_id} · chunks: ${data.chunks}`);
                resetFileForm();
            } else {
                setAlert("error", `Error: ${data.error || "falló el procesamiento"}`);
            }
        } catch {
            setAlert("error", "Error de red/subida. Verifica la URL del backend.");
        } finally {
            setBusy(false);
        }
    };

    const onPasteText = async (e) => {
        e.preventDefault();
        if (!titulo.trim() || !contenido.trim()) {
            return setAlert("warn", "Completa título y contenido.");
        }
        setBusy(true);
        setAlert("info", "Procesando texto…");

        try {
            const fd = new FormData();
            fd.append("titulo", titulo);
            fd.append("contenido", contenido);
            fd.append("categoria", categoriaT);
            fd.append("publicar", publicarT ? "true" : "false");

            const res = await fetch(`${API}/ingest/text`, { method: "POST", body: fd });
            const data = await res.json();

            if (data.ok) {
                setLastResult({ source: "text", ...data });
                setAlert("success", `Guardado ✔  id: ${data.doc_id} · chunks: ${data.chunks}`);
                resetTextForm();
            } else {
                setAlert("error", `Error: ${data.error || "falló el procesamiento"}`);
            }
        } catch {
            setAlert("error", "Error de red/procesamiento. Verifica la URL del backend.");
        } finally {
            setBusy(false);
        }
    };

    const copyLastId = useCallback(async () => {
        if (!lastResult?.doc_id) return;
        try {
            await navigator.clipboard.writeText(lastResult.doc_id);
            setAlert("success", "doc_id copiado al portapapeles.");
        } catch {
            setAlert("warn", "No se pudo copiar. Copia manualmente del mensaje.");
        }
    }, [lastResult]);


    const handleDelete = async (d) => {
        const ok = confirm(`¿Eliminar "${d.titulo || d.filename || d.doc_id}"? Esta acción no se puede deshacer.`);
        if (!ok) return;

        try {
            // 1) Borra en Firestore (desaparece de la UI al instante por onSnapshot)
            await deleteKnowledgeDoc(d.doc_id);

            // 2) (Opcional) Purgar embeddings en Pinecone vía backend si el endpoint existe
            try {
                const res = await fetch(`${API}/ingest/delete?doc_id=${encodeURIComponent(d.doc_id)}`, { method: "DELETE" });
                // no interrumpimos aunque falle
                console.log("purge pinecone:", await res.json());
            } catch (e) {
                console.warn("No se pudo llamar a /ingest/delete, se eliminó solo en Firestore.", e);
            }

            setAlert("success", "Documento eliminado.");
        } catch (e) {
            console.error(e);
            setAlert("error", "No se pudo eliminar. Revisa permisos/reglas.");
        }
    };


    const fileIcon = (name, isNote) => {
        if (isNote) return "📝";
        const ext = (name || "").split(".").pop()?.toLowerCase();
        if (ext === "docx") return "🟦 W";
        if (ext === "xlsx") return "🟩 X";
        if (ext === "pdf") return "🟥 PDF";
        return "📄";
    };
    const fileBadge = (d) => (d.filename ? (d.filename.split(".").pop() || "").toUpperCase() : "NOTA");


    return (
        <div className="wk-wrap">
            {/* Inline CSS: minimal, claro y accesible */}
            <style>{`
        :root{
          --bg:#f6f7fb; --card:#fff; --muted:#6b7280; --text:#111827;
          --brand:#2563eb; --brand-600:#1d4ed8; --ring:#93c5fd;
          --ok:#10b981; --warn:#f59e0b; --err:#ef4444;
          --border:#e5e7eb;
        }
        .wk-wrap{max-width:980px;margin:24px auto;padding:16px;color:var(--text)}
        .wk-title{font-size:1.6rem;font-weight:700;margin:0 0 4px}
        .wk-sub{margin:0 0 16px;color:var(--muted)}
        .tabs{display:flex;gap:8px;margin:8px 0 16px}
        .tab{border:1px solid var(--border);background:var(--card);padding:8px 12px;border-radius:999px;cursor:pointer;font-weight:600}
        .tab[aria-selected="true"]{background:var(--brand);color:white;border-color:var(--brand)}
        .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
        @media (max-width: 900px){ .grid{grid-template-columns:1fr} }
        .card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px}
        .label{display:block;font-weight:600;margin:10px 0 6px}
        .input, .select, .textarea{
          width:100%;border:1px solid var(--border);border-radius:10px;padding:10px 12px;
          font-size:0.95rem;outline:none;background:white
        }
        .input:focus, .select:focus, .textarea:focus{box-shadow:0 0 0 3px var(--ring);border-color:var(--brand)}
        .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
        .btn{display:inline-flex;gap:8px;align-items:center;border-radius:10px;padding:10px 14px;border:1px solid var(--border);background:#fff;cursor:pointer;font-weight:700}
        .btn:disabled{opacity:.6;cursor:not-allowed}
        .btn.primary{background:var(--brand);border-color:var(--brand);color:white}
        .btn.primary:hover{background:var(--brand-600)}
        .btn.ghost{background:transparent}
        .switch{display:inline-flex;align-items:center;gap:8px;cursor:pointer}
        .switch input{width:1.1rem;height:1.1rem}
        .muted{color:var(--muted)}
        .dz{border:2px dashed var(--border);border-radius:14px;padding:22px;text-align:center;background:#fafafa}
        .dz strong{display:block;margin-bottom:6px}
        .dz small{color:var(--muted)}
        .dz--over{background:#eef4ff;border-color:var(--brand)}
        .file-meta{margin-top:10px;padding:8px;border:1px dashed var(--border);border-radius:10px;font-size:.9rem}
        .alert{border-radius:12px;padding:10px 12px;margin-top:16px}
        .alert.info{background:#eef2ff;border:1px solid #e0e7ff}
        .alert.success{background:#ecfdf5;border:1px solid #d1fae5}
        .alert.warn{background:#fffbeb;border:1px solid #fef3c7}
        .alert.error{background:#fef2f2;border:1px solid #fee2e2}
        .sr{position:absolute;left:-10000px;top:auto;width:1px;height:1px;overflow:hidden;}
        .count{font-size:.85rem;color:var(--muted)}
        .actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
        .hr{height:1px;background:var(--border);margin:12px 0}
      `}</style>

            <h2 className="wk-title">Base de Conocimiento</h2>
            <p className="wk-sub">Sube <b>.docx</b>, <b>.xlsx</b> o <b>.pdf</b>, o pega texto (noticias, comunicados, etc.).</p>

            {/* Tabs */}
            <div className="tabs" role="tablist" aria-label="Tipo de carga">
                <button className="tab" role="tab" aria-selected={tab === "file"} onClick={() => setTab("file")}>
                    📄 Archivo
                </button>
                <button className="tab" role="tab" aria-selected={tab === "text"} onClick={() => setTab("text")}>
                    ✍️ Texto
                </button>
            </div>

            <div className="grid" aria-live="polite">
                {/* === Archivo === */}
                <section className="card" hidden={tab !== "file"} aria-hidden={tab !== "file"}>
                    <h3 style={{ marginTop: 0 }}>Subir archivo</h3>

                    {/* Dropzone */}
                    <div
                        ref={dropRef}
                        className="dz"
                        onDragOver={onDragOver}
                        onDragLeave={onDragLeave}
                        onDrop={onDrop}
                        aria-label="Zona para arrastrar y soltar archivos"
                        role="region"
                    >
                        <strong>Arrastra aquí tu archivo</strong>
                        <small>o</small><br />
                        <label className="btn" style={{ marginTop: 8 }}>
                            <input
                                type="file"
                                onChange={(e) => setFile(e.target.files?.[0] || null)}
                                accept=".docx,.xlsx,.pdf"
                                style={{ display: "none" }}
                            />
                            Elegir archivo…
                        </label>
                        <div className="muted" style={{ marginTop: 8 }}>Formatos: .docx / .xlsx / .pdf — máx recomendado 20MB</div>
                    </div>

                    {/* Preview */}
                    {file && (
                        <div className="file-meta" aria-label="Archivo seleccionado">
                            <div><b>Nombre:</b> {file.name}</div>
                            <div><b>Tamaño:</b> {formatBytes(file.size)}</div>
                            <div><b>Tipo:</b> {file.type || "desconocido"}</div>
                        </div>
                    )}

                    <div className="hr" />

                    <label className="label" htmlFor="catF">Categoría</label>
                    <select id="catF" className="select" value={categoriaF} onChange={(e) => setCategoriaF(e.target.value)}>
                        {CATS.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>

                    <div className="row" style={{ marginTop: 10 }}>
                        <label className="switch">
                            <input
                                type="checkbox"
                                checked={publicarF}
                                onChange={(e) => setPublicarF(e.target.checked)}
                                aria-checked={publicarF}
                            />
                            <span>Publicar inmediatamente</span>
                        </label>
                    </div>

                    <div className="actions">
                        <button className="btn primary" onClick={onUpload} disabled={busy}>
                            {busy ? "Procesando…" : "Subir y procesar"}
                        </button>
                        <button className="btn ghost" type="button" onClick={resetFileForm} disabled={busy}>
                            Limpiar
                        </button>
                    </div>
                </section>

                {/* === Texto === */}
                <section className="card" hidden={tab !== "text"} aria-hidden={tab !== "text"}>
                    <h3 style={{ marginTop: 0 }}>Pegar texto</h3>

                    <label className="label" htmlFor="titulo">Título</label>
                    <input
                        id="titulo"
                        className="input"
                        placeholder="Título de la nota/noticia"
                        value={titulo}
                        onChange={(e) => setTitulo(e.target.value)}
                    />

                    <label className="label" htmlFor="contenido">Contenido</label>
                    <textarea
                        id="contenido"
                        className="textarea"
                        rows={10}
                        placeholder="Pega aquí el contenido…"
                        value={contenido}
                        onChange={(e) => setContenido(e.target.value)}
                    />
                    <div className="count">{contenido.length.toLocaleString()} caracteres</div>

                    <div className="hr" />

                    <label className="label" htmlFor="catT">Categoría</label>
                    <select id="catT" className="select" value={categoriaT} onChange={(e) => setCategoriaT(e.target.value)}>
                        {CATS.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>

                    <div className="row" style={{ marginTop: 10 }}>
                        <label className="switch">
                            <input
                                type="checkbox"
                                checked={publicarT}
                                onChange={(e) => setPublicarT(e.target.checked)}
                                aria-checked={publicarT}
                            />
                            <span>Publicar inmediatamente</span>
                        </label>
                    </div>

                    <div className="actions">
                        <button className="btn primary" onClick={onPasteText} disabled={busy}>
                            {busy ? "Procesando…" : "Guardar texto"}
                        </button>
                        <button className="btn ghost" type="button" onClick={resetTextForm} disabled={busy}>
                            Limpiar
                        </button>
                    </div>
                </section>
            </div>

            {/* Estado/Mensajes */}
            <div className={`alert ${msg.type}`} role="status" aria-live="polite">
                {msg.text}
                {lastResult?.doc_id && (
                    <>
                        {" "}
                        <button className="btn" style={{ marginLeft: 8 }} onClick={copyLastId} aria-label="Copiar identificador">
                            Copiar id
                        </button>
                    </>
                )}
            </div>

            {/* === Panel derecho: documentos subidos === */}
            <div className="hr" />
            <section className="card">
                <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
                    <h3 style={{ margin: 0 }}>Documentos subidos</h3>
                    <span className="count">{docs.length} docs</span>
                </div>

                {docs.length === 0 ? (
                    <p className="muted" style={{ marginTop: 8 }}>Aún no hay documentos en la base de conocimiento.</p>
                ) : (
                    <div className="doc-grid" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12, marginTop: 12 }}>
                        {docs.map((d) => {
                            const isNote = !d.filename;
                            return (
                                <article key={d.doc_id} className="doc-card" style={{ position: "relative", border: "1px solid var(--border)", borderRadius: 12, padding: 12, background: "var(--card)" }}>
                                    <button
                                        title="Eliminar"
                                        aria-label="Eliminar"
                                        onClick={() => handleDelete(d)}
                                        className="del-btn"
                                        style={{
                                            position: "absolute", top: 8, right: 8, width: 28, height: 28,
                                            borderRadius: 999, border: "1px solid var(--border)",
                                            background: "#fff", cursor: "pointer"
                                        }}
                                    >
                                        ✕
                                    </button>

                                    <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                                        <div style={{ fontSize: 28, lineHeight: 1 }}>{fileIcon(d.filename, isNote)}</div>
                                        <div style={{ minWidth: 0 }}>
                                            <div style={{ fontWeight: 700, fontSize: ".98rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                                                {d.titulo || d.filename || d.doc_id}
                                            </div>
                                            <div className="muted" style={{ fontSize: ".85rem" }}>
                                                {d.categoria} · {fileBadge(d)} · {d.chunks ?? 0} chunks
                                            </div>
                                        </div>
                                    </div>
                                </article>
                            );
                        })}
                    </div>
                )}
            </section>

        </div>
    );
}
