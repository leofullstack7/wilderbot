import React, { useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:10000";

export default function Conocimiento() {
    const [file, setFile] = useState(null);
    const [categoria, setCategoria] = useState("General");
    const [publicar, setPublicar] = useState(true);

    const [titulo, setTitulo] = useState("");
    const [contenido, setContenido] = useState("");

    const [log, setLog] = useState("");

    const onUpload = async (e) => {
        e.preventDefault();
        if (!file) return setLog("Selecciona un archivo (.docx / .xlsx / .pdf)");
        const fd = new FormData();
        fd.append("file", file);
        fd.append("categoria", categoria);
        fd.append("publicar", publicar ? "true" : "false");

        setLog("Subiendo y procesando...");
        try {
            const res = await fetch(`${API}/ingest/upload`, { method: "POST", body: fd });
            const data = await res.json();
            if (data.ok) {
                setLog(`✅ Subido: ${data.doc_id} | chunks: ${data.chunks}`);
            } else {
                setLog(`❌ Error: ${data.error || "falló"}`);
            }
        } catch (err) {
            setLog("❌ Error de red/subida");
        }
    };

    const onPasteText = async (e) => {
        e.preventDefault();
        if (!titulo || !contenido) return setLog("Completa título y contenido.");
        const fd = new FormData();
        fd.append("titulo", titulo);
        fd.append("contenido", contenido);
        fd.append("categoria", categoria);
        fd.append("publicar", publicar ? "true" : "false");

        setLog("Procesando texto...");
        try {
            const res = await fetch(`${API}/ingest/text`, { method: "POST", body: fd });
            const data = await res.json();
            if (data.ok) {
                setLog(`✅ Guardado: ${data.doc_id} | chunks: ${data.chunks}`);
            } else {
                setLog(`❌ Error: ${data.error || "falló"}`);
            }
        } catch (err) {
            setLog("❌ Error de red/procesamiento");
        }
    };

    return (
        <div style={{ maxWidth: 900, margin: "20px auto", padding: 16 }}>
            <h2 style={{ marginBottom: 8 }}>Base de Conocimiento</h2>
            <p style={{ marginTop: 0, color: "#555" }}>Sube archivos (.docx, .xlsx, .pdf) o pega texto (noticias, comunicados, etc.).</p>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                {/* Subir archivo */}
                <form onSubmit={onUpload} style={{ border: "1px solid #e5e7eb", borderRadius: 10, padding: 12 }}>
                    <h3>Subir archivo</h3>
                    <input type="file" onChange={e => setFile(e.target.files?.[0] || null)} accept=".docx,.xlsx,.pdf" />
                    <div style={{ marginTop: 8 }}>
                        <label>Categoría:&nbsp;</label>
                        <input value={categoria} onChange={e => setCategoria(e.target.value)} placeholder="General" />
                    </div>
                    <div style={{ marginTop: 8 }}>
                        <label><input type="checkbox" checked={publicar} onChange={e => setPublicar(e.target.checked)} /> Publicar</label>
                    </div>
                    <button type="submit" style={{ marginTop: 12 }}>Subir y procesar</button>
                </form>

                {/* Pegar texto */}
                <form onSubmit={onPasteText} style={{ border: "1px solid #e5e7eb", borderRadius: 10, padding: 12 }}>
                    <h3>Pegar texto</h3>
                    <input
                        style={{ width: "100%", marginBottom: 8 }}
                        placeholder="Título de la nota/noticia"
                        value={titulo}
                        onChange={e => setTitulo(e.target.value)}
                    />
                    <textarea
                        rows={10}
                        style={{ width: "100%" }}
                        placeholder="Pega aquí el contenido..."
                        value={contenido}
                        onChange={e => setContenido(e.target.value)}
                    />
                    <div style={{ marginTop: 8 }}>
                        <label>Categoría:&nbsp;</label>
                        <input value={categoria} onChange={e => setCategoria(e.target.value)} placeholder="General" />
                    </div>
                    <div style={{ marginTop: 8 }}>
                        <label><input type="checkbox" checked={publicar} onChange={e => setPublicar(e.target.checked)} /> Publicar</label>
                    </div>
                    <button type="submit" style={{ marginTop: 12 }}>Guardar texto</button>
                </form>
            </div>

            <pre style={{ marginTop: 16, background: "#f9fafb", padding: 12, borderRadius: 8, whiteSpace: "pre-wrap" }}>
                {log || "Estado: listo."}
            </pre>
        </div>
    );
}
