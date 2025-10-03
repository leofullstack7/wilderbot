// src/pages/AdminLayout.jsx
import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { signOut } from "firebase/auth";
import { auth } from "../lib/firebase";

export default function AdminLayout() {
    const [open, setOpen] = useState(false);

    return (
        <div className="min-h-screen bg-slate-50">
            {/* Topbar */}
            <header className="sticky top-0 z-40 bg-white/95 border-b border-slate-200">
                <div className="flex items-center justify-between px-3 md:px-6 py-3">
                    <div className="flex items-center gap-3">
                        {/* Hamburger solo móvil */}
                        <button
                            onClick={() => setOpen(true)}
                            className="md:hidden inline-flex items-center justify-center h-9 w-9 rounded-lg border border-slate-200 text-slate-700"
                            aria-label="Abrir menú"
                        >
                            ☰
                        </button>
                        <h1 className="text-sm sm:text-base font-semibold text-slate-800">
                            Panel Wilder
                        </h1>
                    </div>

                    {/* Nav en desktop */}
                    <nav className="hidden md:flex items-center gap-2">
                        <TopLink to="/admin">Dashboard</TopLink>
                        <TopLink to="/admin/propuestas">Propuestas</TopLink>
                        <TopLink to="/admin/conocimiento">Conocimiento</TopLink>
                        <button
                            onClick={() => signOut(auth)}
                            className="ml-2 rounded-lg px-3 py-1.5 text-sm font-medium text-white bg-gradient-to-r from-emerald-600 to-sky-600 hover:opacity-90"
                        >
                            Cerrar sesión
                        </button>
                    </nav>
                </div>
            </header>

            {/* Drawer móvil */}
            {open && (
                <div className="fixed inset-0 z-[60] md:hidden">
                    <div
                        className="absolute inset-0 bg-black/30"
                        onClick={() => setOpen(false)}
                        aria-hidden="true"
                    />
                    <aside
                        className="absolute left-0 top-0 h-full w-72 bg-white shadow-xl p-4 flex flex-col"
                        role="dialog"
                        aria-label="Menú de navegación"
                    >
                        <div className="flex items-center justify-between mb-2">
                            <span className="font-semibold text-slate-800">Menú</span>
                            <button
                                onClick={() => setOpen(false)}
                                className="h-8 w-8 grid place-items-center rounded-lg border border-slate-200"
                                aria-label="Cerrar menú"
                            >
                                ✕
                            </button>
                        </div>
                        <MobileLink to="/admin" onClick={() => setOpen(false)}>
                            Dashboard
                        </MobileLink>
                        <MobileLink to="/admin/propuestas" onClick={() => setOpen(false)}>
                            Propuestas
                        </MobileLink>
                        <MobileLink to="/admin/conocimiento" onClick={() => setOpen(false)}>
                            Conocimiento
                        </MobileLink>
                        <button
                            onClick={() => { setOpen(false); signOut(auth); }}
                            className="mt-4 rounded-lg px-3 py-2 text-left text-white bg-gradient-to-r from-emerald-600 to-sky-600 hover:opacity-90"
                        >
                            Cerrar sesión
                        </button>

                    </aside>
                </div>
            )}

            {/* Layout principal */}
            <div className="mx-auto max-w-7xl">
                <div className="flex">
                    {/* Sidebar desktop */}
                    <aside className="hidden md:flex md:w-60 md:flex-col md:gap-2 md:border-r md:border-slate-200 md:bg-white">
                        <nav className="p-4 space-y-1">
                            <SideLink to="/admin">Dashboard</SideLink>
                            <SideLink to="/admin/propuestas">Propuestas</SideLink>
                            <SideLink to="/admin/conocimiento">Conocimiento</SideLink>
                        </nav>
                        <div className="p-4 mt-auto">
                            <button
                                onClick={() => signOut(auth)}
                                className="w-full rounded-lg px-3 py-2 text-sm font-medium text-white bg-gradient-to-r from-emerald-600 to-sky-600 hover:opacity-90"
                            >
                                Cerrar sesión
                            </button>
                        </div>
                    </aside>

                    {/* Contenido */}
                    <main className="flex-1 p-3 md:p-6">
                        <Outlet />
                    </main>
                </div>
            </div>
        </div>
    );
}

/* ----- UI helpers de navegación ----- */

function TopLink({ to, children }) {
    return (
        <NavLink
            to={to}
            className={({ isActive }) =>
                `px-3 py-1.5 rounded-lg text-sm font-medium ${isActive
                    ? "text-emerald-700 bg-emerald-50 ring-1 ring-emerald-200"
                    : "text-slate-700 hover:bg-slate-100"
                }`
            }
        >
            {children}
        </NavLink>
    );
}

function SideLink({ to, children }) {
    return (
        <NavLink
            to={to}
            className={({ isActive }) =>
                `block px-3 py-2 rounded-lg text-sm font-medium ${isActive
                    ? "text-emerald-700 bg-emerald-50 ring-1 ring-emerald-200"
                    : "text-slate-700 hover:bg-slate-100"
                }`
            }
        >
            {children}
        </NavLink>
    );
}

function MobileLink({ to, children, onClick }) {
    return (
        <NavLink
            to={to}
            onClick={onClick}
            className={({ isActive }) =>
                `block px-3 py-2 rounded-lg text-base ${isActive
                    ? "text-emerald-700 bg-emerald-50 ring-1 ring-emerald-200"
                    : "text-slate-700 hover:bg-slate-100"
                }`
            }
        >
            {children}
        </NavLink>
    );
}
