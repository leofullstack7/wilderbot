import os
from dotenv import load_dotenv
load_dotenv()

# --- OpenAI
from openai import OpenAI
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
openai_client   = OpenAI(api_key=OPENAI_API_KEY)

# --- Pinecone
from pinecone import Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "wilder-frases")
pc = Pinecone(api_key=PINECONE_API_KEY)
pine_index = pc.Index(PINECONE_INDEX)

# --- Firestore
import firebase_admin
from firebase_admin import credentials, firestore
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "/etc/secrets/firebase.json"

try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate(GOOGLE_CREDS)
    firebase_admin.initialize_app(cred)

db = firestore.client()