import firebase_admin
from firebase_admin import credentials, firestore
import json

# Cargar credenciales
cred = credentials.Certificate("credenciales/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# Cargar archivo JSON
with open("frases_wilder.json", "r", encoding="utf-8") as f:
    frases = json.load(f)

# Subir a Firestore con IDs numéricos
for idx, frase in enumerate(frases, start=1):
    doc_id = str(idx)  # IDs numéricos como string
    db.collection("frases_wilder").document(doc_id).set(frase)

print("✅ Frases cargadas correctamente")
