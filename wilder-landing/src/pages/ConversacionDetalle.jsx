// src/pages/ConversacionDetalle.jsx
import { useEffect, useState } from "react";
import {
    assignConversation,
    fetchConversation,
    formatDate,
    updateConversationState,
    firstOr,
    setPropuestaPotencial,
} from "../lib/firestoreQueries";

const ESTADOS = ["nuevo", "en_revision", "escalado", "resuelto", "cerrado"];

export default function ConversacionDetalle({ id, onClose }) {
    const [doc, setDoc] = useState(null);
    const [loading, setLoading] = useState(true);
    const [estado, setEstado] = useState("");
    const [asignado, setAsignado] = useState("");
    const [potencial, setPotencial] = useState(false);

    useEffect(() => {
        (async () => {
            setLoading(true);
            const d = await fetchConversation(id);
            setDoc(d);
            setEstado(d?.estado || "nuevo");
            setAsignado(d?.asignado_a || "");
            setPotencial(!!d?.propuesta_potencial);
            setLoading(false);
        })();
    }, [id]);

    async function handleEstado() {
        await updateConversationState(id, { estado });
    }

    async function handleAsignar() {
        await assignConversation(id, { asignado_a: asignado });
    }

    async function handleTogglePotencial() {
        const next = !potencial;
        setPotencial(next);
        await setPropuestaPotencial(id, next);
    }

    return (
        <div className="fixed inset-0 z-50">
            {/* overlay */}
            <div className="absolute inset-0 bg-black/40" onClick={onClose} />
            {/* drawer */}
            <div className="absolute right-0 top-0 h-full w-full max-w-xl bg-white shadow-2xl p-5 overflow-y-auto">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold">Conversación</h2>
                    <button onClick={onClose} className="text-gray-600 hover:text-gray-900">Cerrar ✕</button>
                </div>

                {loading ? (
                    <div className="text-gray-500">Cargando…</div>
                ) : !doc ? (
                    <div className="text-red-600">No encontrada</div>
                ) : (
                    <div className="space-y-4">
                        {/* Resumen */}
                        <section className="rounded-xl border bg-white p-4">
                            <h3 className="text-sm text-gray-500">Resumen</h3>
                            <div className="mt-1 text-gray-800">
                                {doc.last_topic_summary || "—"}
                            </div>
                            <div className="mt-2 text-xs text-gray-500">
                                Última actualización: {formatDate(doc.ultima_fecha)}
                            </div>
                        </section>

                        {/* Info principal */}
                        <section className="grid md:grid-cols-2 gap-3">
                            <Card label="Categoría">{firstOr(doc.categoria_general, "—")}</Card>
                            <Card label="Título">{firstOr(doc.titulo_propuesta, "—")}</Card>
                            <Card label="Tono" className="capitalize">{doc.tono_detectado || "—"}</Card>
                            <Card label="Canal">{doc.canal || "—"}</Card>
                            <Card label="Barrio (proyecto)">{doc.project_location || "—"}</Card>
                            <Card label="Barrio (residencia)">{doc?.contact_info?.barrio || "—"}</Card>
                        </section>

                        {/* Contacto (si existe) */}
                        <section className="rounded-xl border bg-white p-4">
                            <h3 className="text-sm text-gray-500 mb-2">Contacto</h3>
                            {doc.contact_collected ? (
                                <div className="text-sm">
                                    <div><b>Nombre:</b> {doc?.contact_info?.nombre || "—"}</div>
                                    <div><b>Teléfono:</b> {doc?.contact_info?.telefono || "—"}</div>
                                </div>
                            ) : (
                                <div className="text-gray-500 text-sm">No hay contacto registrado.</div>
                            )}
                        </section>

                        {/* Marcadores / Acciones rápidas */}
                        <section className="rounded-xl border bg-white p-4">
                            <h3 className="text-sm text-gray-500 mb-2">Marcadores</h3>
                            <label className="inline-flex items-center gap-2 text-sm">
                                <input
                                    type="checkbox"
                                    className="h-4 w-4 rounded border-gray-300 text-emerald-600 focus:ring-emerald-500"
                                    checked={potencial}
                                    onChange={handleTogglePotencial}
                                />
                                <span className="font-medium">Propuesta potencial</span>
                            </label>
                        </section>

                        {/* Acciones */}
                        <section className="rounded-xl border bg-white p-4 space-y-3">
                            <div className="grid md:grid-cols-2 gap-3">
                                <div>
                                    <label className="block text-xs text-gray-500 mb-1">Estado</label>
                                    <div className="flex gap-2">
                                        <select value={estado} onChange={(e) => setEstado(e.target.value)} className="border rounded-lg px-2 py-1 capitalize">
                                            {ESTADOS.map(s => <option key={s} value={s}>{s}</option>)}
                                        </select>
                                        <button onClick={handleEstado} className="px-3 py-1 rounded-lg text-white bg-gradient-to-r from-emerald-600 to-sky-600 hover:opacity-90">
                                            Guardar
                                        </button>
                                    </div>
                                </div>

                                <div>
                                    <label className="block text-xs text-gray-500 mb-1">Asignado a (email/uid)</label>
                                    <div className="flex gap-2">
                                        <input value={asignado} onChange={(e) => setAsignado(e.target.value)} className="border rounded-lg px-2 py-1 w-full" placeholder="usuario@equipo.com" />
                                        <button onClick={handleAsignar} className="px-3 py-1 rounded-lg text-white bg-gradient-to-r from-emerald-600 to-sky-600 hover:opacity-90">
                                            Asignar
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </section>
                    </div>
                )}
            </div>
        </div>
    );
}

function Card({ label, children, className = "" }) {
    return (
        <div className={`rounded-xl border bg-white p-4 ${className}`}>
            <div className="text-xs text-gray-500">{label}</div>
            <div className="mt-1 text-gray-800">{children}</div>
        </div>
    );
}
