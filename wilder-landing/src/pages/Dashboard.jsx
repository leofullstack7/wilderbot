// src/pages/Dashboard.jsx
import { useEffect, useMemo, useState } from "react";
import dayjs from "dayjs";
import { listenConversationsByDate, firstOr } from "../lib/firestoreQueries";
import {
    ResponsiveContainer,
    BarChart,
    Bar,
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    PieChart,
    Pie,
    Cell,
} from "recharts";
import { normalizeCategoriaDocValue } from "../lib/categorias";

const TONOS = ["positivo", "crítico", "preocupación", "propositivo"];

// Colores por canal (alineados con los badges que usamos en Propuestas)
const CHANNEL_COLORS = {
    web: "#64748B",       // slate-500
    whatsapp: "#10B981",  // emerald-500
    telegram: "#0EA5E9",  // sky-500
};

export default function Dashboard() {
    const [range, setRange] = useState("7d"); // 7d | 30d | all
    const [rows, setRows] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showAllCats, setShowAllCats] = useState(false);

    // Suscripción por rango
    useEffect(() => {
        setLoading(true);
        const now = dayjs();
        const from =
            range === "7d"
                ? now.subtract(7, "day")
                : range === "30d"
                    ? now.subtract(30, "day")
                    : dayjs("2000-01-01");

        const unsub = listenConversationsByDate(
            { from, to: now, pageSize: 500 },
            (docs) => {
                setRows(docs);
                setLoading(false);
            }
        );
        return () => unsub();
    }, [range]);

    // KPIs requeridos
    const { total, potenciales, contactOk } = useMemo(() => {
        const t = rows.length;
        const pot = rows.filter((r) => !!r.propuesta_potencial).length;
        const cOk = rows.filter((r) => !!r.contact_collected).length;
        return { total: t, potenciales: pot, contactOk: cOk };
    }, [rows]);

    const pctContactOk = total ? Math.round((contactOk / total) * 100) : 0;

    // Categorías (conteo total) – normalizadas – para cards y top3 de tendencia
    const allCats = useMemo(() => {
        const m = new Map();
        for (const r of rows) {
            const tit = (firstOr(r.titulo_propuesta, "") || "").trim();
            const cat = normalizeCategoriaDocValue(r.categoria_general);
            if (!tit || !cat) continue; // ignora sin título o sin categoría
            m.set(cat, (m.get(cat) || 0) + 1);
        }
        return [...m.entries()]
            .map(([categoria, count]) => ({ categoria, count }))
            .sort((a, b) => b.count - a.count);
    }, [rows]);

    const visibleCats = showAllCats ? allCats : allCats.slice(0, 4);

    // Tono por categoría (top 6) – usando categorías normalizadas
    const toneByCat = useMemo(() => {
        const catMap = new Map();
        for (const r of rows) {
            const tit = (firstOr(r.titulo_propuesta, "") || "").trim();
            const catNorm = normalizeCategoriaDocValue(r.categoria_general) || "Otras";
            if (!tit || !catNorm) continue;
            const tone = (r.tono_detectado || "").toLowerCase();
            if (!catMap.has(catNorm))
                catMap.set(catNorm, {
                    categoria: catNorm,
                    total: 0,
                    positivo: 0,
                    "crítico": 0,
                    "preocupación": 0,
                    "propositivo": 0,
                });
            const obj = catMap.get(catNorm);
            obj.total++;
            if (TONOS.includes(tone)) obj[tone] = (obj[tone] || 0) + 1;
        }
        return [...catMap.values()]
            .sort((a, b) => b.total - a.total)
            .slice(0, 6)
            .map(({ total, ...rest }) => rest);
    }, [rows]);

    // Tendencia semanal: Top 3 categorías normalizadas
    const trendTop3 = useMemo(() => {
        const top3 = allCats.slice(0, 3).map((c) => c.categoria);
        const weekSet = new Set();
        const weekMap = new Map(); // week -> { periodo, [cat]: count }

        for (const r of rows) {
            const tit = (firstOr(r.titulo_propuesta, "") || "").trim();
            const cat = normalizeCategoriaDocValue(r.categoria_general);
            if (!tit || !cat) continue;
            if (!top3.includes(cat)) continue;

            const d = r.ultima_fecha?.toDate ? r.ultima_fecha.toDate() : new Date(r.ultima_fecha);
            const key = dayjs(d).startOf("week").format("YYYY-[W]WW");
            weekSet.add(key);
            if (!weekMap.has(key)) {
                const base = { periodo: key };
                top3.forEach((c) => {
                    base[c] = 0;
                });
                weekMap.set(key, base);
            }
            const obj = weekMap.get(key);
            obj[cat] = (obj[cat] || 0) + 1;
        }

        return [...weekSet].sort().map((k) => weekMap.get(k) || { periodo: k });
    }, [rows, allCats]);

    // === NUEVO: Distribución por canal (donut + lista cantidades) ===
    const canalData = useMemo(() => {
        const counts = { web: 0, whatsapp: 0, telegram: 0, otras: 0 };
        for (const r of rows) {
            const c = (r.canal || "web").toLowerCase();
            if (c in counts) counts[c] += 1;
            else counts.otras += 1;
        }
        const totalC = Object.values(counts).reduce((a, b) => a + b, 0) || 0;

        const pie = [
            { name: "WhatsApp", key: "whatsapp", value: counts.whatsapp, color: CHANNEL_COLORS.whatsapp },
            { name: "Web", key: "web", value: counts.web, color: CHANNEL_COLORS.web },
            { name: "Telegram", key: "telegram", value: counts.telegram, color: CHANNEL_COLORS.telegram },
            // si quieres ocultar "otras", comenta la línea siguiente:
            // { name: "Otras", key: "otras", value: counts.otras, color: "#9CA3AF" },
        ].filter((d) => d.value > 0);

        // porcentajes (para la lista a la derecha)
        const list = pie.map((d) => ({
            ...d,
            pct: totalC ? Math.round((d.value / totalC) * 100) : 0,
        }));

        return { pie, list, total: totalC };
    }, [rows]);

    // etiqueta de % sobre cada porción
    const renderPctLabel = ({ percent }) =>
        percent > 0 ? `${Math.round(percent * 100)}%` : "";

    return (
        // 👇 fuerza este contenedor a estar bajo el menú móvil (que suele ser fixed z-50)
        <div className="relative z-0 space-y-4">
            {/* Filtro de rango */}
            <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3">
                <label className="text-sm text-gray-600" htmlFor="rango">
                    Rango:
                </label>
                <select
                    id="rango"
                    value={range}
                    onChange={(e) => setRange(e.target.value)}
                    className="input w-full sm:w-auto focus:ring-emerald-500"
                >
                    <option value="7d">Últimos 7 días</option>
                    <option value="30d">Últimos 30 días</option>
                    <option value="all">Todo</option>
                </select>
                {loading && (
                    <span role="status" aria-live="polite" className="text-sm text-gray-500">
                        Cargando…
                    </span>
                )}
            </div>

            {/* KPIs (colores del home) */}
            <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 md:grid-cols-3">
                <KPI title="Propuestas potenciales" value={potenciales} accent="emerald" />
                <KPI title="Conversaciones" value={total} accent="sky" />
                <KPI
                    title="% contacto recibido"
                    value={`${pctContactOk}%`}
                    sub={`${contactOk}/${total}`}
                    accent="emerald"
                />
            </div>

            {/* Cards de categorías */}
            <section
                aria-label="Categorías más mencionadas"
                className="relative z-0 rounded-2xl bg-white shadow p-3 sm:p-4"
            >
                <h3 className="text-sm text-gray-600 mb-3">Categorías más mencionadas</h3>
                {visibleCats.length ? (
                    <>
                        <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
                            {visibleCats.map((c) => (
                                <article
                                    key={c.categoria}
                                    className="rounded-xl ring-1 ring-emerald-200/60 bg-gradient-to-br from-emerald-50 to-sky-50 p-3 sm:p-4"
                                >
                                    <div className="text-xs sm:text-sm text-gray-500">Categoría</div>
                                    <h4 className="mt-1 font-semibold text-gray-800 truncate">{c.categoria}</h4>
                                    <div className="mt-3 text-xs text-gray-500">Cantidad</div>
                                    <div className="text-xl sm:text-2xl font-bold text-emerald-700">{c.count}</div>
                                </article>
                            ))}
                        </div>
                        {allCats.length > 4 && (
                            <div className="mt-3">
                                {!showAllCats ? (
                                    <button
                                        onClick={() => setShowAllCats(true)}
                                        className="w-full sm:w-auto px-3 py-2 rounded-lg text-white bg-gradient-to-r from-emerald-600 to-sky-600 hover:opacity-90"
                                        aria-expanded="false"
                                    >
                                        Ver más
                                    </button>
                                ) : (
                                    <button
                                        onClick={() => setShowAllCats(false)}
                                        className="w-full sm:w-auto px-3 py-2 rounded-lg text-white bg-gradient-to-r from-emerald-600 to-sky-600 hover:opacity-90"
                                        aria-expanded="true"
                                    >
                                        Ver menos
                                    </button>
                                )}
                            </div>
                        )}
                    </>
                ) : (
                    <div className="text-sm text-gray-500">Sin datos para este rango.</div>
                )}
            </section>

            {/* === NUEVO: Donut de canales === */}
            <Card title="Distribución por canal">
                {canalData.total ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
                        {/* Gráfico */}
                        <div className="h-56 sm:h-72">
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={canalData.pie}
                                        dataKey="value"
                                        nameKey="name"
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={60}
                                        outerRadius={90}
                                        labelLine={false}
                                        label={renderPctLabel}
                                    >
                                        {canalData.pie.map((entry) => (
                                            <Cell key={entry.key} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Lista de cantidades */}
                        <div>
                            <div className="text-sm text-gray-500 mb-2">
                                Total en rango: <span className="font-medium text-slate-800">{canalData.total}</span>
                            </div>
                            <ul className="space-y-2">
                                {canalData.list.map((d) => (
                                    <li key={d.key} className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <span
                                                className="inline-block h-3 w-3 rounded-full"
                                                style={{ backgroundColor: d.color }}
                                                aria-hidden
                                            />
                                            <span className="text-slate-700">{d.name}</span>
                                        </div>
                                        <div className="text-slate-900 font-semibold tabular-nums">
                                            {d.value} <span className="ml-2 text-xs text-slate-500">({d.pct}%)</span>
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                ) : (
                    <div className="text-sm text-gray-500">Sin datos para este rango.</div>
                )}
            </Card>

            {/* Tono por categoría (barras apiladas, verdes) */}
            <Card title="Tono por categoría (Top 6)">
                <div className="h-56 sm:h-72 lg:h-80">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={toneByCat} margin={{ left: 8, right: 8 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="categoria" />
                            <YAxis allowDecimals={false} domain={["auto", "auto"]} />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="positivo" stackId="a" fill="#34D399" />
                            <Bar dataKey="crítico" stackId="a" fill="#6EE7B7" />
                            <Bar dataKey="preocupación" stackId="a" fill="#A7F3D0" />
                            <Bar dataKey="propositivo" stackId="a" fill="#10B981" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </Card>

            {/* Tendencia semanal – Top 3 categorías */}
            <Card title="Tendencia semanal – Top 3 categorías">
                <div className="h-56 sm:h-72 lg:h-80">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={trendTop3} margin={{ left: 8, right: 8 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="periodo" />
                            <YAxis allowDecimals={false} domain={["auto", "auto"]} />
                            <Tooltip />
                            <Legend />
                            {allCats.slice(0, 3).map((c, idx) => {
                                const strokes = ["#059669", "#10B981", "#34D399"]; // emerald-600/500/400
                                return (
                                    <Line
                                        key={c.categoria}
                                        type="monotone"
                                        dataKey={c.categoria}
                                        stroke={strokes[idx % strokes.length]}
                                        strokeWidth={3}
                                        dot={false}
                                        activeDot={{ r: 5 }}
                                    />
                                );
                            })}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </Card>
        </div>
    );
}

/* ---------- UI helpers ---------- */

function KPI({ title, value, sub, accent = "emerald" }) {
    const grad = accent === "sky" ? "from-sky-600 to-emerald-600" : "from-emerald-600 to-sky-600";

    return (
        <div className="relative z-0 rounded-2xl ring-1 ring-emerald-200/60 bg-white p-3 sm:p-4">
            <div className={`h-1 w-full rounded-full bg-gradient-to-r ${grad} mb-2`} />
            <h3 className="text-sm text-emerald-800">{title}</h3>
            <div className="text-2xl sm:text-3xl font-semibold text-slate-800">{value}</div>
            {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
        </div>
    );
}

function Card({ title, children }) {
    return (
        <section className="relative z-0 rounded-2xl bg-white shadow p-3 sm:p-4 ring-1 ring-emerald-100">
            <h3 className="text-sm text-gray-600 mb-2">{title}</h3>
            {children}
        </section>
    );
}
