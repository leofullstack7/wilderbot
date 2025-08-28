// src/lib/firestoreQueries.js
import {
  collection,
  doc,
  getDoc,
  getDocs,
  onSnapshot,
  orderBy,
  query,
  updateDoc,
  where,
  limit as qLimit,
} from "firebase/firestore";
import dayjs from "dayjs";
import { db } from "./firebase";

/**
 * Suscribe conversaciones por rango de fechas.
 * Server-side: solo filtramos por ultima_fecha para evitar índices compuestos.
 * Otros filtros se aplican en cliente dentro del callback.
 */

export async function setPropuestaPotencial(id, value) {
  const ref = doc(db, "conversaciones", id);
  await updateDoc(ref, { propuesta_potencial: !!value });
}

export function listenConversationsByDate({ from, to, pageSize = 100 }, callback) {
  const col = collection(db, "conversaciones");
  // Firestore almacena timestamps; asumimos campo 'ultima_fecha' como Timestamp/ISO.
  // Filtro mínimo por fecha (>= from). 'to' lo cortamos en cliente para evitar más índices.
  const q = query(
    col,
    where("ultima_fecha", ">=", from.toDate ? from.toDate() : new Date(from)),
    orderBy("ultima_fecha", "desc"),
    qLimit(pageSize)
  );

  const unsub = onSnapshot(q, (snap) => {
    const rows = [];
    snap.forEach((d) => rows.push({ id: d.id, ...d.data() }));
    // Enviamos al callback para que aplique filtros de UI.
    callback(rows);
  });

  return unsub;
}

export async function fetchConversation(id) {
  const ref = doc(db, "conversaciones", id);
  const snapshot = await getDoc(ref);
  if (!snapshot.exists()) return null;
  return { id: snapshot.id, ...snapshot.data() };
}

export async function updateConversationState(id, { estado }) {
  const ref = doc(db, "conversaciones", id);
  await updateDoc(ref, { estado });
}

export async function assignConversation(id, { asignado_a }) {
  const ref = doc(db, "conversaciones", id);
  await updateDoc(ref, { asignado_a });
}

export async function getPanelResumen() {
  // documento único: panel_resumen/global
  const ref = doc(db, "panel_resumen", "global");
  const snapshot = await getDoc(ref);
  return snapshot.exists() ? snapshot.data() : null;
}

/** Utilidades */
export const firstOr = (arr, fallback = "—") =>
  Array.isArray(arr) && arr.length ? arr[0] : fallback;

export const formatDate = (ts) => {
  try {
    // ts puede ser Timestamp de Firestore o ISO/string
    const d = ts?.toDate ? ts.toDate() : new Date(ts);
    return dayjs(d).format("YYYY-MM-DD HH:mm");
  } catch {
    return "—";
  }
};
