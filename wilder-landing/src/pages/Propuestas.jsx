// src/pages/Propuestas.jsx
import { useEffect, useMemo, useState } from "react";
import dayjs from "dayjs";
import {
    listenConversationsByDate,
    formatDate,
    firstOr,
    // ⬇️ NUEVO: importa helpers
    archiveConversation,
    deleteConversation,
} from "../lib/firestoreQueries";
import ConversacionDetalle from "./ConversacionDetalle";
import { CATEGORIAS_CANON, normalizeCategoriaDocValue } from "../lib/categorias";

/* ------------------ constantes ------------------ */
const TONOS = ["positivo", "crítico", "preocupación", "propositivo"];
const CANALES = ["web", "whatsapp", "telegram"];
const ESTADOS = ["nuevo", "en_revision", "escalado", "resuelto", "cerrado"];

/* ------------------ hook: media query ------------------ */
function useMdUp() {
    const [md, setMd] = useState(() =>
        typeof window !== "undefined"
            ? window.matchMedia("(min-width: 768px)").matches
            : true
    );
    useEffect(() => {
        const mql = window.matchMedia("(min-width: 768px)");
        const h = (e) => setMd(e.matches);
        mql.addEventListener?.("change", h);
        return () => mql.removeEventListener?.("change", h);
    }, []);
    return md;
}

export default function Propuestas() {
    const isMdUp = useMdUp();

    // Filtros
    const [range, setRange] = useState("7d");
    const [fCategoria, setFCategoria] = useState("");
    const [fTitulo, setFTitulo] = useState("");
    const [fTono, setFTono] = useState("");
    const [fCanal, setFCanal] = useState("");
    const [fEstado, setFEstado] = useState("");
    const [fConContacto, setFConContacto] = useState(""); // "", "si", "no"
    const [fBarrioProyecto, setFBarrioProyecto] = useState("");
    const [fBarrioResidencia, setFBarrioResidencia] = useState("");

    // Datos
    const [rawRows, setRawRows] = useState([]);
    const [sel, setSel] = useState(null);
    const [loading, setLoading] = useState(true);

    // Busy para acciones
    const [busyId, setBusyId] = useState(null);

    // Suscripción Firestore por rango
    useEffect(() => {
        setLoading(true);
        const now = dayjs();
        const from =
            range === "7d"
                ? now.subtract(7, "day")
                : range === "30d"
                    ? now.subtract(30, "day")
                    : dayjs("2000-01-01");

        const unsub = listenConversationsByDate({ from, to: now, pageSize: 300 }, (rows) => {
            setRawRows(rows);
            setLoading(false);
        });
        return () => unsub();
    }, [range]);

    // Acciones
    const handleArchive = async (id) => {
        if (!id) return;
        const ok = window.confirm("¿Archivar este registro? Podrás verlo luego desde la base de datos.");
        if (!ok) return;
        try {
            setBusyId(`arch-${id}`);
            await archiveConversation(id);
        } finally {
            setBusyId(null);
        }
    };

    const handleDelete = async (id) => {
        if (!id) return;
        const ok = window.confirm("Esto eliminará definitivamente este registro. ¿Continuar?");
        if (!ok) return;
        try {
            setBusyId(`del-${id}`);
            await deleteConversation(id);
        } finally {
            setBusyId(null);
        }
    };

    // Aplicar filtros en cliente (y excluir sin categoría/título y archivados/eliminados)
    const rows = useMemo(() => {
        return rawRows.filter((r) => {
            // Ocultar archivados o marcados como eliminados si existiera soft flag
            if (r.archivado || r.eliminado) return false;

            const catRaw = firstOr(r.categoria_general, "");
            const titRaw = firstOr(r.titulo_propuesta, "");
            const hasCat = !!String(catRaw || "").trim();
            const hasTit = !!String(titRaw || "").trim();

            // SIEMPRE ocultar conversaciones sin categoría NI título
            if (!hasCat && !hasTit) return false;

            // (opcional) si tampoco quieres listar “Consulta”:
            // if (String(catRaw).toLowerCase() === "consulta") return false;

            const catNorm = normalizeCategoriaDocValue(r.categoria_general);

            if (fCategoria && catNorm !== fCategoria) return false;
            if (fTitulo && !tit.toLowerCase().includes(fTitulo.toLowerCase())) return false;
            if (fTono && (r.tono_detectado || "").toLowerCase() !== fTono) return false;
            if (fCanal && (r.canal || "").toLowerCase() !== fCanal) return false;
            if (fEstado && (r.estado || "") !== fEstado) return false;

            if (fConContacto === "si" && !r.contact_collected) return false;
            if (fConContacto === "no" && !!r.contact_collected) return false;

            if (fBarrioProyecto) {
                const bp = (r.project_location || "").toLowerCase();
                if (!bp.includes(fBarrioProyecto.toLowerCase())) return false;
            }
            if (fBarrioResidencia) {
                const br = (r?.contact_info?.barrio || "").toLowerCase();
                if (!br.includes(fBarrioResidencia.toLowerCase())) return false;
            }
            return true;
        });
    }, [
        rawRows,
        fCategoria,
        fTitulo,
        fTono,
        fCanal,
        fEstado,
        fConContacto,
        fBarrioProyecto,
        fBarrioResidencia,
    ]);

    // Paginación (10 en móvil, 20 en desktop)
    const [page, setPage] = useState(1);
    const pageSize = isMdUp ? 20 : 10;
    const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
    const pageRows = rows.slice((page - 1) * pageSize, page * pageSize);
    useEffect(() => setPage(1), [rows.length, isMdUp]);

    return (
        <div className="space-y-4">
            {/* Filtros */}
            <div className="rounded-2xl bg-white/80 backdrop-blur-sm p-3 sm:p-4 shadow border border-emerald-200/60">
                <div className="grid gap-3 grid-cols-2 md:grid-cols-3 xl:grid-cols-4 items-end">
                    <div>
                        <label className="block text-xs text-gray-600 mb-1">Rango</label>
                        <select
                            value={range}
                            onChange={(e) => setRange(e.target.value)}
                            className="input w-full focus:ring-emerald-500"
                        >
                            <option value="7d">Últimos 7 días</option>
                            <option value="30d">Últimos 30 días</option>
                            <option value="all">Todo</option>
                        </select>
                    </div>

                    <Field label="Categoría">
                        <select
                            value={fCategoria}
                            onChange={(e) => setFCategoria(e.target.value)}
                            className="input w-full focus:ring-emerald-500"
                        >
                            <option value="">Todas</option>
                            {CATEGORIAS_CANON.map((c) => (
                                <option key={c} value={c}>
                                    {c}
                                </option>
                            ))}
                        </select>
                    </Field>

                    <Field label="Título">
                        <input
                            value={fTitulo}
                            onChange={(e) => setFTitulo(e.target.value)}
                            className="input w-full focus:ring-emerald-500"
                            placeholder="baches, parque..."
                        />
                    </Field>

                    <Field label="Tono">
                        <select
                            value={fTono}
                            onChange={(e) => setFTono(e.target.value)}
                            className="input w-full focus:ring-emerald-500"
                        >
                            <option value="">Todos</option>
                            {TONOS.map((t) => (
                                <option key={t} value={t}>
                                    {t}
                                </option>
                            ))}
                        </select>
                    </Field>

                    <Field label="Canal">
                        <select
                            value={fCanal}
                            onChange={(e) => setFCanal(e.target.value)}
                            className="input w-full focus:ring-emerald-500"
                        >
                            <option value="">Todos</option>
                            {CANALES.map((c) => (
                                <option key={c} value={c}>
                                    {c}
                                </option>
                            ))}
                        </select>
                    </Field>

                    <Field label="Estado">
                        <select
                            value={fEstado}
                            onChange={(e) => setFEstado(e.target.value)}
                            className="input w-full focus:ring-emerald-500"
                        >
                            <option value="">Todos</option>
                            {ESTADOS.map((s) => (
                                <option key={s} value={s}>
                                    {s}
                                </option>
                            ))}
                        </select>
                    </Field>

                    <Field label="Con contacto">
                        <select
                            value={fConContacto}
                            onChange={(e) => setFConContacto(e.target.value)}
                            className="input w-full focus:ring-emerald-500"
                        >
                            <option value="">Todos</option>
                            <option value="si">Sí</option>
                            <option value="no">No</option>
                        </select>
                    </Field>

                    <Field label="Barrio proyecto">
                        <input
                            value={fBarrioProyecto}
                            onChange={(e) => setFBarrioProyecto(e.target.value)}
                            className="input w-full focus:ring-emerald-500"
                        />
                    </Field>

                    <Field label="Barrio residencia">
                        <input
                            value={fBarrioResidencia}
                            onChange={(e) => setFBarrioResidencia(e.target.value)}
                            className="input w-full focus:ring-emerald-500"
                        />
                    </Field>
                </div>
            </div>

            {/* ======= LISTA MOBILE (cards) ======= */}
            <div className="md:hidden space-y-3">
                {loading ? (
                    <div className="text-center text-gray-500 py-6">Cargando…</div>
                ) : pageRows.length ? (
                    pageRows.map((r) => (
                        <RowCard
                            key={r.id}
                            r={r}
                            onVer={() => setSel(r.id)}
                            onArchivar={() => handleArchive(r.id)}
                            onEliminar={() => handleDelete(r.id)}
                            busyId={busyId}
                        />
                    ))
                ) : (
                    <div className="text-center text-gray-500 py-6">Sin resultados</div>
                )}
            </div>

            {/* ======= TABLA DESKTOP ======= */}
            <div className="hidden md:block rounded-2xl bg-white shadow border border-gray-100 overflow-x-auto">
                <table className="min-w-full text-sm table-fixed">
                    <thead className="bg-gradient-to-r from-emerald-50 to-sky-50 text-emerald-800">
                        <tr>
                            <Th className="w-40">Fecha</Th>
                            <Th className="w-52">Categoría</Th>
                            <Th>Título</Th>
                            <Th className="w-36">Tono</Th>
                            <Th className="w-36">Canal</Th>
                            <Th className="w-56">Barrio (proyecto)</Th>
                            <Th className="w-40">Estado</Th>
                            <Th className="w-28">Contacto</Th>
                            <Th className="w-56">Acciones</Th> {/* ⬅️ NUEVO */}
                        </tr>
                    </thead>
                    <tbody>
                        {loading && (
                            <tr>
                                <td colSpan={10} className="p-4 text-center text-gray-500">
                                    Cargando…
                                </td>
                            </tr>
                        )}

                        {!loading && pageRows.length === 0 && (
                            <tr>
                                <td colSpan={10} className="p-4 text-center text-gray-500">
                                    Sin resultados
                                </td>
                            </tr>
                        )}

                        {!loading && pageRows.length > 0 && pageRows.map((r) => {
                            const categoria = normalizeCategoriaDocValue(r.categoria_general);
                            const archBusy = busyId === `arch-${r.id}`;
                            const delBusy = busyId === `del-${r.id}`;
                            return (
                                <tr key={r.id} className="border-t hover:bg-emerald-50/40 transition">
                                    <Td className="whitespace-nowrap">{formatDate(r.ultima_fecha || r.fecha_inicio)}</Td>
                                    <Td className="font-medium">{categoria}</Td>
                                    <Td title={firstOr(r.titulo_propuesta, "—")} className="truncate">
                                        {firstOr(r.titulo_propuesta, "—")}
                                    </Td>
                                    <Td><ToneChip tone={r.tono_detectado} /></Td>
                                    <Td><CanalBadge canal={r.canal} /></Td>
                                    <Td className="whitespace-nowrap">{r.project_location || "—"}</Td>
                                    <Td><EstadoBadge estado={r.estado || "nuevo"} /></Td>
                                    <Td>{r.contact_collected ? "Sí" : "No"}</Td>
                                    <Td>
                                        <div className="flex items-center gap-2">
                                            <button
                                                onClick={() => setSel(r.id)}
                                                className="px-3 py-1 rounded-lg text-white bg-gradient-to-r from-emerald-600 to-sky-600 hover:opacity-90 shadow"
                                            >
                                                Ver
                                            </button>
                                            <button
                                                onClick={() => handleArchive(r.id)}
                                                disabled={archBusy || delBusy}
                                                className="px-3 py-1 rounded-lg border border-amber-300 text-amber-700 hover:bg-amber-50 disabled:opacity-50"
                                                title="Archivar"
                                            >
                                                Archivar
                                            </button>
                                            <button
                                                onClick={() => handleDelete(r.id)}
                                                disabled={archBusy || delBusy}
                                                className="px-3 py-1 rounded-lg border border-rose-300 text-rose-700 hover:bg-rose-50 disabled:opacity-50"
                                                title="Eliminar"
                                            >
                                                Eliminar
                                            </button>
                                        </div>
                                    </Td>
                                </tr>
                            );
                        })}
                    </tbody>

                </table>
            </div>

            {/* Paginación */}
            <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3 justify-between text-sm">
                <div className="text-gray-600">
                    Total: {rows.length} • Página {page} / {totalPages} • Mostrar {pageSize} por página
                </div>
                <div className="flex items-center gap-2">
                    <button
                        className="btn hover:bg-emerald-50 disabled:opacity-50"
                        onClick={() => setPage((p) => Math.max(1, p - 1))}
                        disabled={page <= 1}
                    >
                        Anterior
                    </button>
                    <button
                        className="btn hover:bg-emerald-50 disabled:opacity-50"
                        onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                        disabled={page >= totalPages}
                    >
                        Siguiente
                    </button>
                </div>
            </div>

            {/* Modal de detalle */}
            {sel && <ConversacionDetalle id={sel} onClose={() => setSel(null)} />}
        </div>
    );
}

/** ---------- Subcomponentes visuales ---------- */
function Field({ label, children }) {
    return (
        <label className="text-xs text-gray-600">
            {label}
            <div className="mt-1">{children}</div>
        </label>
    );
}
function Th({ children, className = "" }) {
    return <th className={`text-left font-semibold px-3 py-2 ${className}`}>{children}</th>;
}
function Td({ children, className = "" }) {
    return <td className={`px-3 py-2 align-top ${className}`}>{children}</td>;
}

function ToneChip({ tone }) {
    const t = (tone || "").toLowerCase();
    const map = {
        positivo: "bg-emerald-100 text-emerald-700",
        "crítico": "bg-rose-100 text-rose-700",
        "preocupación": "bg-amber-100 text-amber-800",
        propositivo: "bg-emerald-100 text-emerald-700",
    };
    const cls = map[t] || "bg-gray-100 text-gray-700";
    return (
        <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${cls} capitalize`}>
            {tone || "—"}
        </span>
    );
}

function EstadoBadge({ estado }) {
    const map = {
        nuevo: "bg-gray-100 text-gray-700",
        en_revision: "bg-sky-100 text-sky-800",
        escalado: "bg-amber-100 text-amber-800",
        resuelto: "bg-emerald-100 text-emerald-700",
        cerrado: "bg-slate-100 text-slate-700",
    };
    const cls = map[estado] || "bg-gray-100 text-gray-700";
    return (
        <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${cls} capitalize`}>
            {estado}
        </span>
    );
}

function CanalBadge({ canal }) {
    const c = (canal || "").toLowerCase();
    const map = {
        web: "bg-slate-100 text-slate-700",
        whatsapp: "bg-emerald-100 text-emerald-700",
        telegram: "bg-sky-100 text-sky-800",
    };
    const label = c || "—";
    const cls = map[c] || "bg-gray-100 text-gray-700";
    return (
        <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${cls}`}>
            {label}
        </span>
    );
}

/* ---------- Card para móvil ---------- */
function RowCard({ r, onVer, onArchivar, onEliminar, busyId }) {
    const categoria = normalizeCategoriaDocValue(r.categoria_general);
    const archBusy = busyId === `arch-${r.id}`;
    const delBusy = busyId === `del-${r.id}`;

    return (
        <article className="rounded-xl border border-emerald-200/60 bg-white p-3 shadow-sm">
            <div className="flex items-start justify-between gap-2">
                <div className="text-xs text-slate-500">{formatDate(r.ultima_fecha || r.fecha_inicio)}</div>
                <EstadoBadge estado={r.estado || "nuevo"} />
            </div>

            <h4 className="mt-1 font-semibold text-slate-800 truncate">
                {firstOr(r.titulo_propuesta, "—")}
            </h4>

            <div className="mt-2 flex flex-wrap items-center gap-2">
                <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700">
                    {categoria}
                </span>
                <ToneChip tone={r.tono_detectado} />
                <CanalBadge canal={r.canal} />
                <span className="text-xs text-slate-600">{r.project_location || "—"}</span>
                <span className="text-xs">{r.contact_collected ? "📞 Con contacto" : "Sin contacto"}</span>
            </div>

            <div className="mt-3 flex justify-end gap-2">
                <button
                    onClick={onVer}
                    className="px-3 py-1 rounded-lg text-white bg-gradient-to-r from-emerald-600 to-sky-600 hover:opacity-90"
                >
                    Ver
                </button>
                <button
                    onClick={onArchivar}
                    disabled={archBusy || delBusy}
                    className="px-3 py-1 rounded-lg border border-amber-300 text-amber-700 hover:bg-amber-50 disabled:opacity-50"
                >
                    Archivar
                </button>
                <button
                    onClick={onEliminar}
                    disabled={archBusy || delBusy}
                    className="px-3 py-1 rounded-lg border border-rose-300 text-rose-700 hover:bg-rose-50 disabled:opacity-50"
                >
                    Eliminar
                </button>
            </div>
        </article>
    );
}
