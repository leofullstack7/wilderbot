// src/lib/firestoreQueries.js
import dayjs from "dayjs";
import {
  collection,
  doc,
  getDoc,
  getDocs,
  deleteDoc,
  onSnapshot,
  orderBy,
  query,
  updateDoc,
  where,
  limit as qLimit,
  serverTimestamp,
  Timestamp,
} from "firebase/firestore";
import { db } from "./firebase";

/* ====================== Helpers visibles en UI ====================== */
export function firstOr(val, fallback = "—") {
  if (Array.isArray(val)) return val.length ? val[0] : fallback;
  if (typeof val === "string") return val || fallback;
  return fallback;
}

export function formatDate(v) {
  try {
    const d =
      v instanceof Date
        ? v
        : v?.toDate
        ? v.toDate()
        : typeof v === "string"
        ? new Date(v)
        : v;
    return dayjs(d).format("YYYY-MM-DD HH:mm");
  } catch {
    return "—";
  }
}

/* ====================== Normalizadores internos ====================== */
function toJsDate(x, def = null) {
  if (!x) return def;
  if (x instanceof Date) return x;
  if (x?.toDate) return x.toDate(); // Firestore Timestamp
  if (x?.$d) return x.$d; // dayjs
  if (typeof x === "string" || typeof x === "number") return new Date(x);
  return def;
}

function normalizeRow(id, data = {}) {
  // Tomar ultima_fecha o, en su defecto, fecha_inicio
  const lf =
    data?.ultima_fecha != null ? data.ultima_fecha : data?.fecha_inicio ?? null;
  const lastDate = toJsDate(lf, null);

  const cg = data?.categoria_general;
  const tp = data?.titulo_propuesta;

  return {
    id,
    ...data,
    // normalizamos a Date para facilitar UI
    ultima_fecha: lastDate,
    categoria_general: Array.isArray(cg) ? cg : cg ? [cg] : [],
    titulo_propuesta: Array.isArray(tp) ? tp : tp ? [tp] : [],
  };
}

/* ====================== Acciones ====================== */
export async function setPropuestaPotencial(id, value) {
  const ref = doc(db, "conversaciones", id);
  await updateDoc(ref, { propuesta_potencial: !!value });
}

export async function fetchConversation(id) {
  const ref = doc(db, "conversaciones", id);
  const snapshot = await getDoc(ref);
  if (!snapshot.exists()) return null;
  return normalizeRow(snapshot.id, snapshot.data() || {});
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
  const ref = doc(db, "panel_resumen", "global");
  const snapshot = await getDoc(ref);
  return snapshot.exists() ? snapshot.data() : null;
}

/** Marca la conversación como archivada (soft delete) */
export async function archiveConversation(id) {
  const ref = doc(db, "conversaciones", id);
  await updateDoc(ref, {
    archivado: true,
    archivado_fecha: serverTimestamp(),
  });
}

/** Elimina definitivamente la conversación (hard delete) */
export async function deleteConversation(id) {
  const ref = doc(db, "conversaciones", id);
  await deleteDoc(ref);
}

/* ====================== Conocimiento (listar / eliminar) ====================== */
export function listenKnowledgeDocs(callback) {
  const col = collection(db, "knowledge_docs");

  // Intento 1: ordenar por fecha desc (si hay null puede fallar según reglas)
  let triedTitle = false;
  let triedPlain = false;

  const attach = (q) =>
    onSnapshot(
      q,
      (snap) => {
        const rows = snap.docs.map((d) => ({ id: d.id, ...(d.data() || {}) }));
        callback(rows);
      },
      (err) => {
        console.warn("[listenKnowledgeDocs] order attempt failed:", err?.message || err);
        if (!triedTitle) {
          triedTitle = true;
          // Intento 2: ordenar por título asc
          unsub = attach(query(col, orderBy("titulo", "asc")));
        } else if (!triedPlain) {
          triedPlain = true;
          // Intento 3: sin orderBy
          unsub = attach(col);
        } else {
          // Último recurso: one-shot
          getDocs(col)
            .then((snap) => {
              const rows = snap.docs.map((d) => ({ id: d.id, ...(d.data() || {}) }));
              callback(rows);
            })
            .catch((e) => {
              console.error("[listenKnowledgeDocs] getDocs failed:", e);
              callback([]);
            });
        }
      }
    );

  let unsub = attach(query(col, orderBy("fecha", "desc")));
  return () => unsub && unsub();
}


export async function deleteKnowledgeDoc(docId) {
  const ref = doc(db, "knowledge_docs", docId);
  await deleteDoc(ref);
}


/* ====================== Suscripción con fallback robusto ====================== */
/**
 * Suscribe conversaciones por rango en 'ultima_fecha' (o 'fecha_inicio' si no hay).
 * - Intenta query por fecha (>= from) + orderBy(desc).
 * - Si falla (índice/reglas) cae a orderBy(desc) sin where.
 * - Si aun así no llegan docs, hace un getDocs one-shot y filtra en cliente.
 * Evita loaders infinitos.
 */
export function listenConversationsByDate(
  { from, to, pageSize = 200 },
  callback
) {
  const col = collection(db, "conversaciones");

  const fromDate =
    toJsDate(from, dayjs().subtract(30, "day").toDate()) || new Date(0);
  const toDate = toJsDate(to, new Date());

  const qPrimary = query(
    col,
    where("ultima_fecha", ">=", Timestamp.fromDate(fromDate)),
    orderBy("ultima_fecha", "desc"),
    qLimit(pageSize)
  );

  const qFallback = query(col, orderBy("ultima_fecha", "desc"), qLimit(pageSize));

  let unsub = () => {};
  let triedFallback = false;
  let triedOneShot = false;

  function filterTo(rowsRaw) {
    return rowsRaw
      .map((d) => normalizeRow(d.id, d.data ? d.data() : d))
      .filter((r) => {
        // Aseguramos fecha (ultima_fecha normalizada arriba)
        if (!r.ultima_fecha) return false;
        return r.ultima_fecha <= toDate;
      });
  }

  const attach = (q) => {
    unsub = onSnapshot(
      q,
      (snap) => {
        const rows = filterTo(snap.docs);
        callback(rows);

        // Si no llegó nada y aún no probamos one-shot, probemos traer todo y filtrar
        if (!rows.length && !triedOneShot) {
          triedOneShot = true;
          getDocs(col)
            .then((all) => {
              const allRows = filterTo(all.docs);
              if (allRows.length) callback(allRows);
            })
            .catch((e) => {
              console.error("[listenConversationsByDate] getDocs error:", e);
              // al menos devolvemos []
              callback([]);
            });
        }
      },
      (err) => {
        console.error("[listenConversationsByDate] onSnapshot error:", err);
        if (!triedFallback) {
          triedFallback = true;
          try {
            unsub();
          } catch {}
          // engancha fallback simple
          unsub = onSnapshot(
            qFallback,
            (snap2) => {
              const rows = filterTo(snap2.docs);
              callback(rows);
            },
            (err2) => {
              console.error("[listenConversationsByDate] fallback error:", err2);
              // último intento: one-shot
              if (!triedOneShot) {
                triedOneShot = true;
                getDocs(col)
                  .then((all) => callback(filterTo(all.docs)))
                  .catch((e) => {
                    console.error(
                      "[listenConversationsByDate] final getDocs error:",
                      e
                    );
                    callback([]);
                  });
              } else {
                callback([]);
              }
            }
          );
        } else {
          callback([]);
        }
      }
    );
  };

  attach(qPrimary);
  return () => unsub();
}
