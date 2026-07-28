"""
Interfaz del agente con Streamlit.
Cubre las etapas 4, 5, 6 y 8 (recuperacion, respuesta, interfaz y registro).

Para levantarla:
    streamlit run app.py
"""

import json
import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

ARCHIVO_LOG = "registro_preguntas.jsonl"

# Etapa 5: el prompt le dice al modelo que use SOLO los documentos.
# Esto es para que no invente respuestas (alucinaciones).
PROMPT = """Eres el asistente virtual de Mercado Central 24h.
Responde la pregunta usando solamente la informacion del contexto de abajo.
Si la respuesta no aparece en el contexto, responde exactamente:
"No encontre esa informacion en los documentos de Mercado Central 24h."
No inventes datos. Responde en espanol, claro y breve.

Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:"""


def buscar_clave():
    """
    Busca la clave de Groq. En mi PC viene del archivo .env, y cuando la app
    esta publicada en Streamlit Cloud viene de los "Secrets".
    """
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


@st.cache_resource
def cargar():
    """Carga la base vectorial y el modelo. Solo se hace una vez."""
    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    base = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    # Etapa 4: el retriever busca los 6 chunks mas parecidos a la pregunta.
    # Probe con 4 y a veces se quedaba corto, con 6 responde mejor.
    retriever = base.as_retriever(search_kwargs={"k": 6})
    clave = buscar_clave()
    if not clave:
        st.error(
            "Falta la clave GROQ_API_KEY. Si estas corriendo la app en tu "
            "computadora, ponla en el archivo .env. Si esta publicada en "
            "Streamlit Cloud, ponla en Settings > Secrets."
        )
        st.stop()

    modelo = ChatGroq(
        model="llama-3.3-70b-versatile", temperature=0.2, api_key=clave
    )
    return retriever, modelo


def guardar_log(pregunta, respuesta, fuentes, segundos):
    """
    Etapa 8: guardo cada pregunta en un archivo .jsonl (una linea por pregunta).
    Sirve para revisar despues como se esta comportando el agente.
    """
    registro = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "pregunta": pregunta,
        "respuesta": respuesta,
        "fuentes": fuentes,
        "segundos": round(segundos, 2),
    }
    with open(ARCHIVO_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def responder(pregunta, retriever, modelo):
    """Busca los fragmentos, arma el prompt y pide la respuesta al modelo."""
    inicio = time.time()

    # 1. Recuperar los fragmentos relevantes
    encontrados = retriever.invoke(pregunta)
    contexto = "\n\n".join(d.page_content for d in encontrados)

    # 2. Generar la respuesta
    salida = modelo.invoke(PROMPT.format(contexto=contexto, pregunta=pregunta))

    # 3. Armar la lista de fuentes (archivo y pagina) para poder verificar
    fuentes = []
    for d in encontrados:
        archivo = d.metadata.get("archivo", "?")
        pagina = d.metadata.get("pagina", 0)
        fuente = f"{archivo} (pag. {pagina})" if pagina else archivo
        if fuente not in fuentes:
            fuentes.append(fuente)

    tiempo = time.time() - inicio
    guardar_log(pregunta, salida.content, fuentes, tiempo)

    return salida.content, fuentes, tiempo


# ------------------- Interfaz -------------------

st.set_page_config(page_title="Agente Mercado Central 24h", page_icon="🛒")

st.title("Agente Mercado Central 24h")

# Etapa 6: hay que avisar que se esta hablando con una IA, no con una persona
st.info(
    "Estas conversando con un agente de IA. Responde solo con la informacion "
    "de los documentos internos de la empresa y siempre muestra la fuente."
)

with st.sidebar:
    st.subheader("Ejemplos de preguntas")
    st.write("- Cual es la politica de devoluciones para clientes?")
    st.write("- Que beneficios tiene el programa Cliente VIP Central?")
    st.write("- Cuantas unidades de Arroz Integral 1kg hay?")
    st.write("- Como doy de alta un proveedor nuevo?")

retriever, modelo = cargar()

# Guardo la conversacion para que se vea el historial
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for m in st.session_state.mensajes:
    with st.chat_message(m["rol"]):
        st.write(m["texto"])
        if m.get("fuentes"):
            st.caption("Fuentes: " + " | ".join(m["fuentes"]))

pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta})
    with st.chat_message("user"):
        st.write(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en los documentos..."):
            respuesta, fuentes, tiempo = responder(pregunta, retriever, modelo)
        st.write(respuesta)
        st.caption("Fuentes: " + " | ".join(fuentes))
        st.caption(f"Respondido en {tiempo:.1f} segundos")

    st.session_state.mensajes.append(
        {"rol": "assistant", "texto": respuesta, "fuentes": fuentes}
    )
