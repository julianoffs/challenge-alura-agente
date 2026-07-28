"""
Crea la base vectorial con los documentos de Mercado Central 24h.
Cubre las etapas 1, 2 y 3 del challenge (colecta, extraccion e indexacion).

Se ejecuta una sola vez:
    python crear_base.py
"""

import os
import time

import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

CARPETA_DATOS = "data"
CARPETA_BASE = "vectorstore"

# Etapa 1: organizo los documentos por categoria del negocio.
# Esta categoria la guardo como metadato para saber de donde sale cada respuesta.
CATEGORIAS = {
    "Politica_Atencion_Cliente_Devoluciones_Mercado_Central_24h.pdf": "Atencion al Cliente",
    "FAQ_Mercado_Central_24h.pdf": "Preguntas Frecuentes",
    "Reglamento_Interno_Mercado_Central_24h.pdf": "Recursos Humanos",
    "Manual_Proveedores_Mercado_Central_24h.pdf": "Compras y Proveedores",
    "inventario_de_supermercado_latam.xlsx": "Inventario",
}


def leer_pdfs():
    """Lee los PDFs con PyPDF. Cada pagina queda como un documento aparte."""
    documentos = []

    for archivo in sorted(os.listdir(CARPETA_DATOS)):
        if not archivo.endswith(".pdf"):
            continue

        print("Leyendo", archivo)
        loader = PyPDFLoader(os.path.join(CARPETA_DATOS, archivo))
        paginas = loader.load()

        for pagina in paginas:
            pagina.metadata["archivo"] = archivo
            pagina.metadata["categoria"] = CATEGORIAS.get(archivo, "General")
            # PyPDF cuenta las paginas desde 0, le sumo 1 para que se entienda
            pagina.metadata["pagina"] = pagina.metadata.get("page", 0) + 1

        documentos += paginas

    return documentos


def leer_inventario():
    """
    Lee el Excel con pandas. Como una tabla no es texto corrido, convierto
    cada fila en una frase repitiendo el nombre de cada columna.
    """
    archivo = "inventario_de_supermercado_latam.xlsx"
    print("Leyendo", archivo)

    df = pd.read_excel(os.path.join(CARPETA_DATOS, archivo))
    documentos = []

    for _, fila in df.iterrows():
        texto = ", ".join(f"{columna}: {fila[columna]}" for columna in df.columns)

        documentos.append(
            Document(
                page_content="Producto del inventario. " + texto,
                metadata={
                    "archivo": archivo,
                    "categoria": CATEGORIAS[archivo],
                    "pagina": 0,  # el Excel no tiene paginas
                },
            )
        )

    return documentos


print("--- Etapa 1 y 2: leyendo los documentos ---")
documentos = leer_pdfs() + leer_inventario()
print("Documentos leidos:", len(documentos))

# Etapa 2: corto el texto en pedazos mas chicos.
# Si mando el documento entero el modelo se pierde, asi funciona mejor.
print("\n--- Cortando el texto en chunks ---")
divisor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = divisor.split_documents(documentos)
print("Chunks:", len(chunks))

# Etapa 3: convierto cada chunk en un vector y lo guardo en FAISS
print("\n--- Etapa 3: creando los embeddings (tarda un poco) ---")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# El plan gratuito de Gemini deja hacer 100 embeddings por minuto, y yo tengo
# muchos mas chunks que eso. Asi que los mando por tandas y espero un minuto
# entre una y otra, si no me devuelve error 429.
TANDA = 90
base = None

for i in range(0, len(chunks), TANDA):
    grupo = chunks[i:i + TANDA]
    print(f"  procesando {i + len(grupo)} de {len(chunks)} chunks...")

    if base is None:
        base = FAISS.from_documents(grupo, embeddings)
    else:
        base.add_documents(grupo)

    # si todavia quedan chunks, espero para no pasarme del limite
    if i + TANDA < len(chunks):
        time.sleep(62)

base.save_local(CARPETA_BASE)

print("\nListo. La base quedo guardada en la carpeta", CARPETA_BASE)
