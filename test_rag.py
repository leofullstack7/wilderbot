"""
TEST DE DIAGNÓSTICO - RAG FUNCIONAL
====================================
Este archivo prueba si los documentos están correctamente en Pinecone
y si el RAG los puede encontrar.

INSTRUCCIONES:
1. Copia este archivo a la raíz de tu proyecto (al lado de main.py)
2. Ejecuta: python test_rag.py
3. Envíame la salida completa
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# Configuración
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "wilder-frases")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Clientes
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def test_rag(query: str, top_k: int = 5):
    """Función idéntica a la de main.py"""
    print(f"\n🔍 Buscando: '{query}'")
    print("-" * 60)
    
    # Crear embedding
    emb = client.embeddings.create(model=EMBEDDING_MODEL, input=query).data[0].embedding
    print(f"✅ Embedding creado: {len(emb)} dimensiones")
    
    # Buscar en Pinecone
    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    print(f"✅ Pinecone respondió: {len(res.matches)} resultados")
    
    # Procesar resultados
    hits = []
    for i, m in enumerate(res.matches):
        metadata = m.metadata or {}
        texto = metadata.get("texto", "")
        
        hit = {
            "id": m.id,
            "score": float(m.score),
            "texto": texto,
            "doc_id": metadata.get("doc_id", "N/A"),
            "categoria": metadata.get("categoria", "N/A"),
        }
        hits.append(hit)
        
        print(f"\n📄 Resultado #{i+1}:")
        print(f"   ID: {hit['id']}")
        print(f"   Score: {hit['score']:.4f}")
        print(f"   Doc ID: {hit['doc_id']}")
        print(f"   Categoría: {hit['categoria']}")
        print(f"   Texto: {hit['texto'][:150]}{'...' if len(hit['texto']) > 150 else ''}")
    
    return hits

def test_index_stats():
    """Ver estadísticas del índice"""
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DEL ÍNDICE PINECONE")
    print("="*60)
    
    try:
        stats = index.describe_index_stats()
        print(f"✅ Total de vectores: {stats.total_vector_count}")
        print(f"✅ Dimensiones: {stats.dimension}")
        
        if hasattr(stats, 'namespaces'):
            print(f"\n📁 Namespaces:")
            for ns, data in stats.namespaces.items():
                count = data.vector_count if hasattr(data, 'vector_count') else data
                print(f"   - '{ns}': {count} vectores")
    except Exception as e:
        print(f"❌ Error al obtener stats: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 INICIANDO TEST DE DIAGNÓSTICO RAG")
    print("="*60)
    
    # 1. Verificar índice
    test_index_stats()
    
    # 2. Pruebas de búsqueda
    print("\n" + "="*60)
    print("🔬 PRUEBAS DE BÚSQUEDA")
    print("="*60)
    
    # Prueba 1: Búsqueda genérica
    test_rag("educación", top_k=3)
    
    # Prueba 2: Búsqueda específica
    test_rag("salud y hospitales", top_k=3)
    
    # Prueba 3: Búsqueda sobre Wilder
    test_rag("Wilder Escobar propuestas", top_k=3)
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETADO")
    print("="*60)
    print("\n📋 INSTRUCCIONES:")
    print("1. Si ves resultados con texto → El RAG funciona ✅")
    print("2. Si los resultados están vacíos → Problema de ingesta ❌")
    print("3. Si hay error → Envíame el mensaje de error ⚠️")
    print("\nEnvíame TODO lo que salió en pantalla.")
