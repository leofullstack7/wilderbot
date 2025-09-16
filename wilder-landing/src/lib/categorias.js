// src/lib/categorias.js

// Lista canónica (ordenada de forma general). Añade/quita según tu contexto.
export const CATEGORIAS_CANON = [
  "Consulta",
  "Infraestructura y urbanismo",
  "Movilidad y transporte",
  "Seguridad ciudadana",
  "Salud pública",
  "Educación",
  "Medio ambiente y sostenibilidad",
  "Servicios públicos (agua, energía, aseo)",
  "Espacio público y parques",
  "Vivienda y hábitat",
  "Empleo y desarrollo económico",
  "Cultura y patrimonio",
  "Deporte y recreación",
  "Participación ciudadana y gobierno",
  "Inclusión social y diversidad",
  "Juventud",
  "Mujer y equidad de género",
  "Adulto mayor",
  "Discapacidad y accesibilidad",
  "Tecnología y gobierno digital",
  "Gestión del riesgo y emergencias",
  "Ordenamiento territorial",
  "Turismo y promoción",
  "Transparencia y control social",
  "Otras",
];

// Sinónimos / palabras clave por categoría (en minúsculas, sin tildes).
// El normalizador hará matching por "includes" sobre el texto limpio.
const KEYWORDS = {
  "Infraestructura y urbanismo": [
    "infraestructura", "urbanismo", "infraestructura urbana", "vias", "vías",
    "calle", "bache", "andén", "anden", "vereda", "malla vial", "puente", "obra civil"
  ],
  "Movilidad y transporte": [
    "movilidad", "transporte", "trafico", "tráfico", "transito", "tránsito",
    "semaforo", "semáforo", "parqueadero", "bicicleta", "cicloruta", "bus", "metro",
    "taxi", "peatonal", "vehicular"
  ],
  "Seguridad ciudadana": [
    "seguridad", "hurto", "robo", "atraco", "violencia", "convivencia",
    "policia", "policía", "delincuencia", "microtrafico", "microtráfico"
  ],
  "Salud pública": [
    "salud", "hospital", "clinica", "clínica", "eps", "ips", "vacuna",
    "vacunacion", "vacunación", "salubridad", "ambulancia"
  ],
  "Educación": [
    "educacion", "educación", "colegio", "escuela", "universidad",
    "jardin", "jardín", "beca", "icetex"
  ],
  "Medio ambiente y sostenibilidad": [
    "medio ambiente", "ambiente", "sostenibilidad", "contaminacion", "contaminación",
    "ruido", "aire", "arbol", "árbol", "reforestacion", "reforestación",
    "reciclaje", "rio", "río", "quebrada"
  ],
  "Servicios públicos (agua, energía, aseo)": [
    "servicios publicos", "servicio publico", "agua", "acueducto", "alcantarillado",
    "energia", "energía", "gas", "aseo", "alumbrado", "recoleccion", "recolección", "basuras"
  ],
  "Espacio público y parques": [
    "espacio publico", "espacio público", "parque", "plaza", "alameda",
    "zona verde", "jardin", "jardín", "parque lineal"
  ],
  "Vivienda y hábitat": [
    "vivienda", "habitat", "hábitat", "mejoramiento de vivienda", "arriendo",
    "invasion", "invasión", "titulacion", "titulación"
  ],
  "Empleo y desarrollo económico": [
    "empleo", "trabajo", "emprendimiento", "negocio", "empresa", "economia",
    "economía", "mercados", "comercio", "informalidad"
  ],
  "Cultura y patrimonio": [
    "cultura", "patrimonio", "biblioteca", "museo", "teatro", "festival",
    "arte", "historia"
  ],
  "Deporte y recreación": [
    "deporte", "recreacion", "recreación", "gimnasio", "cancha", "polideportivo",
    "torneo", "actividad fisica", "actividad física"
  ],
  "Participación ciudadana y gobierno": [
    "participacion", "participación", "cabildo", "veeduria", "veeduría",
    "junta", "edil", "alcaldia", "alcaldía", "gobierno", "audiencia"
  ],
  "Inclusión social y diversidad": [
    "inclusion", "inclusión", "diversidad", "lgbti", "lgbtiq", "migrantes",
    "poblacion vulnerable", "equidad social"
  ],
  "Juventud": [
    "juventud", "jovenes", "jóvenes", "adolescentes"
  ],
  "Mujer y equidad de género": [
    "mujer", "mujeres", "equidad de genero", "género", "violencia de genero",
    "igualdad", "embarazo adolescente"
  ],
  "Adulto mayor": [
    "adulto mayor", "anciano", "vejez", "personas mayores", "centro dia", "centro día"
  ],
  "Discapacidad y accesibilidad": [
    "discapacidad", "personas con discapacidad", "accesibilidad", "rampa",
    "inclusivo", "lengua de senas", "señas"
  ],
  "Tecnología y gobierno digital": [
    "tecnologia", "tecnología", "internet", "conectividad", "transformacion digital",
    "transformación digital", "gobierno digital", "datos abiertos", "app", "software", "wifi", "fibra"
  ],
  "Gestión del riesgo y emergencias": [
    "riesgo", "desastre", "inundacion", "inundación", "deslizamiento",
    "emergencia", "bomberos", "simulacro", "prevencion", "prevención"
  ],
  "Ordenamiento territorial": [
    "pot", "plan de ordenamiento", "uso del suelo", "ordenamiento",
    "expansion urbana", "expansión urbana", "lotes", "control urbano", "licencia"
  ],
  "Turismo y promoción": [
    "turismo", "visitantes", "promocion", "promoción", "rutas turisticas",
    "rutas turísticas", "hotel", "gastronomia", "gastronomía"
  ],
  "Transparencia y control social": [
    "transparencia", "corrupcion", "corrupción", "control social",
    "rendicion de cuentas", "rendición de cuentas", "acceso a la informacion",
    "informacion publica", "información pública"
  ],
  "Otras": [
    // catch-all; no pongas palabras aquí
  ],
};

// Limpia texto: minúsculas, sin tildes, sin signos, espacios comprimidos.
export function limpiarTexto(s = "") {
  return String(s)
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// Normaliza un nombre de categoría libre a la canónica.
// Devuelve una cadena de CATEGORIAS_CANON (u "Otras" si no matchea).
export function normalizeCategoria(input) {
  if (!input) return "Otras";
  const clean = limpiarTexto(input);

  // 1) match directo contra nombres canónicos
  for (const canon of CATEGORIAS_CANON) {
    const c = limpiarTexto(canon);
    if (clean === c) return canon;
  }

  // 2) match por keywords (primer hit gana; si quieres, haz scoring)
  for (const canon of Object.keys(KEYWORDS)) {
    const kws = KEYWORDS[canon];
    if (!kws || !kws.length) continue;
    for (const kw of kws) {
      const k = limpiarTexto(kw);
      if (k && clean.includes(k)) return canon;
    }
  }

  // 3) fallback
  return "Otras";
}

// Intenta inferir una categoría desde un texto largo (resumen, mensaje, etc.)
// Retorna la mejor coincidencia o "Otras".
export function inferCategoriaDesdeTexto(texto = "") {
  const clean = limpiarTexto(texto);
  if (!clean) return "Otras";

  // scoring por apariciones de keywords
  const scores = {};
  for (const canon of Object.keys(KEYWORDS)) {
    scores[canon] = 0;
    for (const kw of KEYWORDS[canon]) {
      const k = limpiarTexto(kw);
      if (!k) continue;
      if (clean.includes(k)) scores[canon] += 1;
    }
  }
  // mejor puntaje
  const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
  if (!best || best[1] === 0) return "Otras";
  return best[0];
}

// Normaliza una categoría que venga como array o string (como en tus docs)
export function normalizeCategoriaDocValue(value) {
  const v = Array.isArray(value) ? (value[0] || "") : (value || "");
  const s = String(v).trim();
  if (!s) return "Otras";
  if (s.toLowerCase() === "consulta") return "Consulta";
  return normalizeCategoria(s); // 👈 antes decía "raw" (variable inexistente)
}
