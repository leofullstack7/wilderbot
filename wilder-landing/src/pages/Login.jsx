// src/pages/Login.jsx
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { onAuthStateChanged, signInWithEmailAndPassword, signOut } from "firebase/auth";
import {
    collection,
    getDocs,
    query,
    where,
    doc,
    getDoc,
    limit as qLimit,
} from "firebase/firestore";
import { auth, db } from "../lib/firebase";

const ALLOWLIST_COLLECTION = "admin_usuarios";

export default function Login() {
    const navigate = useNavigate();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    // Si ya está logueado y llega a /login, mándalo a /admin
    useEffect(() => {
        const unsub = onAuthStateChanged(auth, (u) => {
            if (u) navigate("/admin", { replace: true });
        });
        return () => unsub();
    }, [navigate]);

    async function handleSubmit(e) {
        e.preventDefault();
        setError("");
        setLoading(true);
        try {
            // 1) Login
            const emailInput = email.trim();
            const { user } = await signInWithEmailAndPassword(auth, emailInput, password);

            // 2) Allowlist en admin_usuarios (UID → email → email_lc → correo)
            const colName = ALLOWLIST_COLLECTION;
            const emailNorm = (user.email || "").trim();
            const emailLc = emailNorm.toLowerCase();

            let allowedDoc = null;

            // A) por UID
            const uidRef = doc(db, colName, user.uid);
            const uidSnap = await getDoc(uidRef);
            if (uidSnap.exists()) allowedDoc = { id: uidSnap.id, ...uidSnap.data() };

            // B) por email
            if (!allowedDoc) {
                const s1 = await getDocs(query(collection(db, colName), where("email", "==", emailNorm)));
                if (!s1.empty) allowedDoc = { id: s1.docs[0].id, ...s1.docs[0].data() };
            }

            // C) por email_lc
            if (!allowedDoc) {
                const s2 = await getDocs(query(collection(db, colName), where("email_lc", "==", emailLc)));
                if (!s2.empty) allowedDoc = { id: s2.docs[0].id, ...s2.docs[0].data() };
            }

            // D) por 'correo' (tu campo actual)
            if (!allowedDoc) {
                const s3 = await getDocs(query(collection(db, colName), where("correo", "==", emailNorm)));
                if (!s3.empty) allowedDoc = { id: s3.docs[0].id, ...s3.docs[0].data() };
            }
            if (!allowedDoc) {
                const s4 = await getDocs(query(collection(db, colName), where("correo", "==", emailLc)));
                if (!s4.empty) allowedDoc = { id: s4.docs[0].id, ...s4.docs[0].data() };
            }

            // E) Fallback pequeño
            if (!allowedDoc) {
                const s = await getDocs(query(collection(db, colName), qLimit(100)));
                s.forEach((d) => {
                    const data = d.data() || {};
                    const candidates = [data.email, data.email_lc, data.correo, data.usuario]
                        .filter(Boolean)
                        .map((x) => String(x).toLowerCase());
                    if (candidates.includes(emailLc)) allowedDoc = { id: d.id, ...data };
                });
            }

            if (!allowedDoc || allowedDoc.active === false) {
                await signOut(auth);
                throw new Error("Tu cuenta no está autorizada para acceder al panel.");
            }

            // 3) ✅ Todo bien → ir a /admin
            navigate("/admin", { replace: true });
        } catch (err) {
            const msg =
                err?.code === "auth/invalid-credential" ? "Credenciales inválidas."
                    : err?.code === "auth/invalid-email" ? "Email inválido."
                        : err?.message || "Error al iniciar sesión";
            setError(msg);
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-white to-gray-50">
            <div className="w-full max-w-md bg-white shadow-lg rounded-2xl p-6">
                <h1 className="text-2xl font-semibold text-gray-800 mb-2">Panel – Wilder Escobar</h1>
                <p className="text-sm text-gray-500 mb-6">Acceso exclusivo para administradores</p>

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label className="block text-sm text-gray-600 mb-1">Email</label>
                        <input
                            type="email"
                            className="w-full rounded-xl border border-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                            placeholder="ciudadaniaenmarcha@demo.local"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            autoComplete="username"
                            required
                        />
                    </div>

                    <div>
                        <label className="block text-sm text-gray-600 mb-1">Contraseña</label>
                        <input
                            type="password"
                            className="w-full rounded-xl border border-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                            placeholder="••••••••"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            autoComplete="current-password"
                            required
                        />
                    </div>

                    {error && (
                        <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl p-2">
                            {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full rounded-xl bg-purple-600 text-white py-2 font-medium hover:bg-purple-700 disabled:opacity-60 transition"
                    >
                        {loading ? "Ingresando…" : "Ingresar"}
                    </button>
                </form>

                <div className="mt-6 text-xs text-gray-400">
                    <p>Demo sugerido: <b>ciudadaniaenmarcha@demo.local</b> / <b>escobar%</b></p>
                </div>
            </div>
        </div>
    );
}
