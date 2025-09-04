// src/App.jsx
import { BrowserRouter, Routes, Route, Navigate, Outlet } from "react-router-dom";
import { onAuthStateChanged } from "firebase/auth";
import { useEffect, useState } from "react";
import { auth } from "./lib/firebase";

import Home from "./pages/Home";
import Login from "./pages/Login";
import AdminLayout from "./pages/AdminLayout";
import Dashboard from "./pages/Dashboard";
import Propuestas from "./pages/Propuestas";

import Conocimiento from "./pages/Conocimiento.jsx";

function ProtectedRoute() {
  const [ready, setReady] = useState(false);
  const [user, setUser] = useState(null);

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, (u) => {
      setUser(u);
      setReady(true);
    });
    return () => unsub();
  }, []);

  if (!ready) return <div className="p-6 text-gray-600">Cargando…</div>;
  if (!user) return <Navigate to="/login" replace />;
  return <Outlet />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Público */}
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />

        {/* Protegido */}
        <Route path="/admin" element={<ProtectedRoute />}>
          <Route element={<AdminLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="propuestas" element={<Propuestas />} />
          </Route>
        </Route>

        <Route path="/admin/conocimiento" element={
          <AdminLayout><Conocimiento /></AdminLayout>
        } />

        

        {/* 404 simple */}
        <Route path="*" element={<div className="p-6">Página no encontrada</div>} />
      </Routes>
    </BrowserRouter>
  );
}
