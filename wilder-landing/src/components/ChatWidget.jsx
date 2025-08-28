// src/components/ChatWidget.jsx
import { useEffect, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL;
const CHAT_KEY = "wilder_chat_id";

export default function ChatWidget({ open, onClose, initialMessage, initialFaqOrigin }) {
    const [input, setInput] = useState("");
    const [sending, setSending] = useState(false);
    const [messages, setMessages] = useState([
        { role: "assistant", content: "¡Hola! Soy el asistente de Wilder. ¿Cómo puedo ayudarte?" },
    ]);

    // chat_id por sesión/pestaña
    const [chatId, setChatId] = useState("");
    const hasPrefilled = useRef(false);
    const endRef = useRef(null);

    // Recupera chatId si ya existe en esta pestaña
    useEffect(() => {
        const saved = sessionStorage.getItem(CHAT_KEY);
        if (saved) setChatId(saved);
    }, []);

    // Enviar automáticamente si llegó desde una FAQ
    useEffect(() => {
        if (open && initialMessage && !hasPrefilled.current) {
            hasPrefilled.current = true;
            setInput(initialMessage);
            handleSend(initialMessage, { faqOrigin: initialFaqOrigin });
        }
    }, [open, initialMessage, initialFaqOrigin]);

    // Auto-scroll al último mensaje
    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, open]);

    async function handleSend(forcedText, opts = {}) {
        const text = (forcedText ?? input).trim();
        if (!text || sending) return;

        setSending(true);
        setMessages((prev) => [...prev, { role: "user", content: text }]);
        setInput("");

        try {
            const payload = {
                mensaje: text,
                canal: "web", // 👈 importante: se guardará en Firestore
                chat_id: chatId || undefined, // si viene vacío, el backend crea uno
                faq_origen: opts.faqOrigin || undefined,
            };

            const res = await fetch(`${API_URL}/responder`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                const txt = await res.text().catch(() => "");
                throw new Error(txt || "Respuesta no válida del servidor");
            }

            const data = await res.json();

            // Guardar chat_id para conservar el hilo al reabrir
            if (data?.chat_id && data.chat_id !== chatId) {
                setChatId(data.chat_id);
                sessionStorage.setItem(CHAT_KEY, data.chat_id);
            }

            const bot = data?.respuesta || "Hubo un problema al responder. Inténtalo de nuevo.";
            setMessages((prev) => [...prev, { role: "assistant", content: bot }]);

            // Clasificar sin bloquear la UI
            const usedChatId = data?.chat_id || chatId;
            if (usedChatId) {
                fetch(`${API_URL}/clasificar`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ chat_id: usedChatId }),
                }).catch(() => { });
            }
        } catch (err) {
            setMessages((prev) => [
                ...prev,
                { role: "assistant", content: "No pude contactar el servidor. Revisa tu conexión." },
            ]);
        } finally {
            setSending(false);
        }
    }

    if (!open) return null;

    return (
        <div className="fixed bottom-4 right-4 z-50 w-[min(420px,92vw)] shadow-2xl rounded-2xl border border-white/10
                    bg-gradient-to-br from-emerald-600/95 to-sky-700/95 backdrop-blur text-white">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
                <div className="flex items-center gap-2">
                    <div className="h-8 w-8 rounded-full bg-white/20 grid place-items-center font-semibold">W</div>
                    <h3 className="font-semibold tracking-wide">Asistente de Wilder</h3>
                    {/* Indicador de envío (parpadea) */}
                    {sending && (
                        <span
                            className="ml-2 text-xs rounded-full px-2 py-0.5 bg-white/20 animate-pulse"
                            aria-live="polite"
                            role="status"
                        >
                            Enviando…
                        </span>
                    )}
                </div>
                <button
                    onClick={onClose}
                    className="px-2 py-1 text-sm rounded hover:bg-white/10 transition"
                    aria-label="Cerrar conversación"
                >
                    ✕
                </button>
            </div>

            {/* Mensajes */}
            <div className="max-h-[55vh] overflow-y-auto px-3 py-3 space-y-2" aria-live="polite">
                {messages.map((m, i) => (
                    <div
                        key={i}
                        className={
                            m.role === "user"
                                ? "ml-auto max-w-[85%] bg-white text-slate-800 rounded-xl px-3 py-2 shadow"
                                : "max-w-[90%] bg-white/10 rounded-xl px-3 py-2"
                        }
                    >
                        {m.content}
                    </div>
                ))}
                <div ref={endRef} />
            </div>

            {/* Input */}
            <div className="p-3 flex gap-2 border-t border-white/10">
                <textarea
                    className="flex-1 rounded-xl px-3 py-2 bg-white/90 text-slate-900 outline-none min-h-[44px] max-h-40 resize-y"
                    placeholder="Escribe tu mensaje…"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            handleSend();
                        }
                    }}
                />
                <button
                    onClick={() => handleSend()}
                    disabled={sending}
                    className={`rounded-xl px-4 py-2 active:scale-95 transition disabled:opacity-60 ${sending ? "bg-white/20 animate-pulse" : "bg-white/20 hover:bg-white/30"
                        }`}
                    aria-live="polite"
                    aria-busy={sending ? "true" : "false"}
                >
                    {sending ? <span className="animate-pulse">Enviando…</span> : "Enviar"}
                </button>
            </div>
        </div>
    );
}
