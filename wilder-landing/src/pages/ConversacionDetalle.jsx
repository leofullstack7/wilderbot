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

    // ✅ NUEVO: Estado para controlar el modal de resumen completo
    const [showResumenCompleto, setShowResumenCompleto] = useState(false);

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
                        {/* ═══════════════════════════════════════════════════════════ */}
                        {/* RESUMEN MEJORADO CON BOTÓN "VER MÁS" */}
                        {/* ═══════════════════════════════════════════════════════════ */}
                        <section className="rounded-xl border bg-gradient-to-br from-emerald-50 to-sky-50 p-4">
                            <div className="flex items-start justify-between gap-2">
                                <div className="flex-1">
                                    <h3 className="text-sm font-semibold text-emerald-800 mb-1">
                                        Resumen
                                    </h3>
                                    <div className="text-sm text-gray-700 leading-relaxed">
                                        {(doc?.resumen && String(doc.resumen).trim())
                                            || (doc?.last_topic_summary ? String(doc.last_topic_summary).slice(0, 100) : "Sin resumen disponible")}
                                    </div>
                                </div>

                                {/* ✅ Botón "Ver más" solo si existe resumen_completo */}
                                {doc?.resumen_completo && (
                                    <button
                                        onClick={() => setShowResumenCompleto(true)}
                                        className="flex-shrink-0 px-3 py-1.5 rounded-lg text-xs font-medium text-white bg-gradient-to-r from-emerald-600 to-sky-600 hover:opacity-90 shadow-sm transition"
                                    >
                                        Ver más
                                    </button>
                                )}
                            </div>

                            <div className="mt-2 text-xs text-gray-500">
                                Última actualización: {formatDate(doc.resumen_updated_at || doc.ultima_fecha)}
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

            {/* ═══════════════════════════════════════════════════════════ */}
            {/* MODAL DE RESUMEN COMPLETO */}
            {/* ═══════════════════════════════════════════════════════════ */}
            {showResumenCompleto && doc?.resumen_completo && (
                <ResumenCompletoModal
                    resumen={doc.resumen_completo}
                    onClose={() => setShowResumenCompleto(false)}
                />
            )}
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

/* ═══════════════════════════════════════════════════════════════════════
   COMPONENTE: MODAL DE RESUMEN COMPLETO
   ═══════════════════════════════════════════════════════════════════════ */
function ResumenCompletoModal({ resumen, onClose }) {
    return (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
            {/* Overlay */}
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="relative bg-white rounded-2xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b bg-gradient-to-r from-emerald-50 to-sky-50">
                    <h3 className="text-lg font-semibold text-emerald-900">
                        📋 Resumen Completo
                    </h3>
                    <button
                        onClick={onClose}
                        className="text-gray-500 hover:text-gray-700 transition"
                        title="Cerrar"
                    >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto px-6 py-4">
                    <div className="space-y-4 text-sm">
                        {/* Tema Principal */}
                        {resumen.tema_principal && (
                            <Section title="Tema Principal" icon="🎯">
                                <p className="text-gray-800 font-medium">{resumen.tema_principal}</p>
                            </Section>
                        )}

                        {/* Consultas */}
                        {resumen.consultas && resumen.consultas.length > 0 && (
                            <Section title="Consultas" icon="❓">
                                <ul className="list-disc list-inside space-y-1 text-gray-700">
                                    {resumen.consultas.map((consulta, i) => (
                                        <li key={i}>{consulta}</li>
                                    ))}
                                </ul>
                            </Section>
                        )}

                        {/* Propuesta */}
                        {resumen.propuesta && (
                            <Section title="Propuesta" icon="💡">
                                <p className="text-gray-800 leading-relaxed">{resumen.propuesta}</p>
                            </Section>
                        )}

                        {/* Argumento */}
                        {resumen.argumento && (
                            <Section title="Argumento" icon="📝">
                                <p className="text-gray-700 leading-relaxed italic">{resumen.argumento}</p>
                            </Section>
                        )}

                        {/* Ubicación */}
                        {resumen.ubicacion && (
                            <Section title="Ubicación del Proyecto" icon="📍">
                                <p className="text-gray-800 font-medium">{resumen.ubicacion}</p>
                            </Section>
                        )}

                        {/* Contacto */}
                        {resumen.contacto && (resumen.contacto.nombre || resumen.contacto.telefono || resumen.contacto.barrio) && (
                            <Section title="Información de Contacto" icon="👤">
                                <div className="space-y-1 text-gray-700">
                                    {resumen.contacto.nombre && (
                                        <div><span className="font-medium">Nombre:</span> {resumen.contacto.nombre}</div>
                                    )}
                                    {resumen.contacto.barrio && (
                                        <div><span className="font-medium">Barrio (residencia):</span> {resumen.contacto.barrio}</div>
                                    )}
                                    {resumen.contacto.telefono && (
                                        <div><span className="font-medium">Teléfono:</span> {resumen.contacto.telefono}</div>
                                    )}
                                </div>
                            </Section>
                        )}

                        {/* Estado */}
                        {resumen.estado && (
                            <Section title="Estado Actual" icon="📊">
                                <span className="inline-block px-3 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800 capitalize">
                                    {resumen.estado}
                                </span>
                            </Section>
                        )}

                        {/* Historial Resumido */}
                        {resumen.historial_resumido && resumen.historial_resumido.length > 0 && (
                            <Section title="Últimos Intercambios" icon="💬">
                                <div className="space-y-2">
                                    {resumen.historial_resumido.map((msg, i) => (
                                        <div
                                            key={i}
                                            className={`p-3 rounded-lg ${msg.rol === "Usuario"
                                                    ? "bg-sky-50 border-l-4 border-sky-500"
                                                    : "bg-emerald-50 border-l-4 border-emerald-500"
                                                }`}
                                        >
                                            <div className="text-xs font-semibold text-gray-600 mb-1">
                                                {msg.rol}
                                            </div>
                                            <div className="text-gray-700 text-sm leading-relaxed">
                                                {msg.mensaje}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </Section>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-end gap-3 px-6 py-4 border-t bg-gray-50">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 rounded-lg text-white bg-gradient-to-r from-emerald-600 to-sky-600 hover:opacity-90 transition shadow"
                    >
                        Cerrar
                    </button>
                </div>
            </div>
        </div>
    );
}

/* Subcomponente para las secciones del modal */
function Section({ title, icon, children }) {
    return (
        <div className="border-l-4 border-emerald-500 pl-4">
            <h4 className="text-sm font-semibold text-emerald-900 mb-2 flex items-center gap-2">
                <span>{icon}</span>
                <span>{title}</span>
            </h4>
            <div>{children}</div>
        </div>
    );
}