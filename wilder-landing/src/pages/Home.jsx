// src/pages/Home.jsx
import { useMemo, useState } from "react";
import ChatWidget from "../components/ChatWidget";
import logoUrl from "/vite.svg";

// helper para faq_origen
function slugify(s) {
    return s
        .toLowerCase()
        .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/(^-|-$)/g, "");
}

export default function Home() {
    const [chatOpen, setChatOpen] = useState(false);
    const [prefill, setPrefill] = useState({ text: "", faq: "" });

    const faqs = useMemo(
        () => [
            {
                q: "¿Cómo puedo presentar una propuesta ciudadana?",
                a: "Cuéntanos brevemente tu idea, el lugar y por qué es importante. El equipo de Wilder prioriza y canaliza las propuestas.",
            },
            {
                q: "¿Qué temas atiende el despacho de Wilder?",
                a: "Educación, salud, movilidad, seguridad y medio ambiente, entre otros. Si no estás seguro, igual cuéntanos y te orientamos.",
            },
            {
                q: "¿Tardan en responder?",
                a: "Siempre intentamos dar una primera orientación en poco tiempo. Si el caso requiere gestión, te lo indicaremos.",
            },
        ],
        []
    );

    function toggleChat() {
        setPrefill({ text: "", faq: "" });
        setChatOpen((v) => !v);
    }

    function talkAbout(questionText) {
        setPrefill({ text: questionText, faq: slugify(questionText) });
        setChatOpen(true);
    }

    return (
        <main className="min-h-dvh bg-gradient-to-b from-emerald-50 via-white to-sky-50 text-slate-800 selection:bg-emerald-200/60">
            {/* Hero / Logo centrado */}
            <section className="container mx-auto px-4 py-16 md:py-24 text-center">
                <div className="mx-auto mb-8 md:mb-12 h-20 w-20 md:h-28 md:w-28 rounded-3xl grid place-items-center
                        bg-gradient-to-br from-emerald-500 to-sky-600 text-white shadow-lg animate-[float_6s_ease-in-out_infinite]">
                    <img src={logoUrl} alt="Logo" className="h-12 w-12 opacity-90" />
                </div>

                <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight bg-clip-text text-transparent
                       bg-gradient-to-r from-emerald-600 to-sky-700">
                    Wilder Escobar · Ciudadanía en Acción
                </h1>
                <p className="mt-4 text-lg md:text-xl text-slate-600">
                    Un espacio para que tus propuestas, problemas y reconocimientos lleguen directo al equipo de Wilder.
                </p>
            </section>

            {/* Banner */}
            <section className="relative overflow-hidden">
                <div className="container mx-auto px-4">
                    <div className="rounded-3xl p-8 md:pb-12 md:pt-12
                          bg-gradient-to-r from-emerald-600 to-sky-700
                          text-white shadow-xl">
                        <h2 className="text-2xl md:text-3xl font-bold">Construyamos soluciones juntos</h2>
                        <p className="mt-2 md:mt-3 text-white/90">
                            Comparte tu situación o idea. Te acompañamos para canalizarla y darle seguimiento.
                        </p>

                        <div className="mt-6 flex flex-wrap gap-3">
                            <button
                                onClick={toggleChat}
                                className="rounded-xl bg-white text-slate-900 px-5 py-2 font-semibold
                           hover:scale-[1.02] active:scale-95 transition shadow"
                            >
                                {chatOpen ? "Cerrar conversación" : "Hablar con el asistente de Wilder"}
                            </button>
                            <a
                                href="#faqs"
                                className="rounded-xl bg-white/15 text-white px-5 py-2 font-medium border border-white/30
                           hover:bg-white/25 transition"
                            >
                                Ver preguntas frecuentes
                            </a>
                        </div>
                    </div>
                </div>
            </section>

            {/* Texto */}
            <section className="container mx-auto px-4 py-12 md:py-16">
                <div className="max-w-3xl mx-auto text-center">
                    <p className="text-lg leading-relaxed text-slate-700">
                        Nuestro compromiso es escuchar, priorizar y gestionar. Tu voz es clave para mejorar
                        nuestros barrios y ciudades. Empieza contándonos en qué podemos ayudarte.
                    </p>
                </div>
            </section>

            {/* 3 Cards */}
            <section className="container mx-auto px-4 pb-8 md:pb-16">
                <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                    {[
                        { t: "Propón una idea", d: "Describe la propuesta, su impacto y el lugar. Te orientamos para presentarla.", i: "💡" },
                        { t: "Reporta un problema", d: "Indícanos qué sucede y dónde. Escalamos al equipo o entidad correspondiente.", i: "📍" },
                        { t: "Reconoce a tu comunidad", d: "Cuéntanos buenas prácticas o liderazgos que debamos visibilizar.", i: "🤝" },
                    ].map((c, idx) => (
                        <div
                            key={idx}
                            className="group rounded-2xl border border-slate-200/60 bg-white p-6 shadow-sm
                         hover:shadow-xl hover:-translate-y-0.5 transition
                         bg-gradient-to-br from-white to-emerald-50/40"
                        >
                            <div className="text-3xl">{c.i}</div>
                            <h3 className="mt-3 text-xl font-bold text-slate-800">{c.t}</h3>
                            <p className="mt-2 text-slate-600">{c.d}</p>
                            <button
                                onClick={toggleChat}
                                className="mt-4 inline-block rounded-lg px-4 py-2 font-semibold
                           bg-gradient-to-r from-emerald-600 to-sky-600 text-white
                           hover:opacity-90 active:scale-95 transition"
                            >
                                {chatOpen ? "Volver al chat" : "Empezar ahora"}
                            </button>
                        </div>
                    ))}
                </div>
            </section>

            {/* Toggle + Chat */}
            <section className="container mx-auto px-4 pb-10 md:pb-16">
                <div className="text-center">
                    <button
                        onClick={toggleChat}
                        className="rounded-2xl px-6 py-3 font-semibold shadow-lg
                       bg-gradient-to-r from-emerald-600 to-sky-600 text-white
                       hover:opacity-90 active:scale-95 transition"
                    >
                        {chatOpen ? "Cerrar conversación" : "Hablar con el asistente de Wilder"}
                    </button>
                </div>
            </section>

            {/* FAQs */}
            <section id="faqs" className="container mx-auto px-4 pb-24">
                <h2 className="text-2xl md:text-3xl font-extrabold text-center mb-8
                       bg-clip-text text-transparent bg-gradient-to-r from-emerald-600 to-sky-700">
                    Preguntas frecuentes
                </h2>
                <div className="max-w-3xl mx-auto divide-y divide-slate-200/70 rounded-2xl border border-slate-200/60 bg-white">
                    {faqs.map((f, i) => (
                        <details key={i} className="group p-5 open:bg-emerald-50/40">
                            <summary className="cursor-pointer list-none text-lg font-semibold flex items-center justify-between">
                                <span className="pr-4">{f.q}</span>
                                <span className="text-emerald-700 group-open:rotate-180 transition">⌄</span>
                            </summary>
                            <p className="mt-3 text-slate-700">{f.a}</p>
                            <button
                                onClick={() => talkAbout(f.q)}
                                className="mt-4 inline-block rounded-lg px-4 py-2 font-semibold
                           bg-gradient-to-r from-emerald-600 to-sky-600 text-white
                           hover:opacity-90 active:scale-95 transition"
                            >
                                Hablar sobre esto con el asistente de Wilder
                            </button>
                        </details>
                    ))}
                </div>
            </section>

            {/* Chat flotante */}
            <ChatWidget
                open={chatOpen}
                onClose={() => setChatOpen(false)}
                initialMessage={prefill.text}
                initialFaqOrigin={prefill.faq}
            />
        </main>
    );
}
