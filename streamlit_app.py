import streamlit as st
from utils import *
from agent import *

# Configuracion de la interfaz de Streamlit
st.title("Chatbot sobre curriculums")
st.write("El asistente respondera usando los curriculums cargados.")

# Entrada del usuario
pregunta = st.text_input("Escribe tu pregunta:")

if st.button("Enviar"):
    if pregunta:
        agente_cv = GroqAgent()
        with st.spinner("Consultando al agente..."):
            respuesta, total_tokens = query_with_agent(agente_cv, pregunta)
            # Obtener los mensajes del agente después de la consulta
            logs_groq = agente_cv.messages
        st.success("Consulta completada!")

        st.info(f"🔢 Tokens usados en esta consulta: {total_tokens}")

        # Mostrar logs en un expander
        with st.expander("📜 Historial completo de mensajes con Groq"):
            for msg in logs_groq:
                emoji = "👤" if msg["role"] == "user" else "🤖"
                # Formateo especial para mensajes del sistema
                if msg["role"] == "system":
                    st.markdown(f"{emoji} **{msg['role']}:**")
                    st.text_area("", value=msg["content"], height=200, label_visibility="collapsed")
                else:
                    # Para otros mensajes, mostramos sin formato de código
                    content = str(msg.get("content", "")).replace("`", "'")
                    st.markdown(f"{emoji} **{msg['role']}:** {content}")

        st.write(f"✅ Respuesta final: {respuesta}")
    else:
        st.warning("Por favor, escribe una pregunta.")
