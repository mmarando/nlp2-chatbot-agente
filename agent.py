from groq import Groq
import os
import re
from utils import *

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class GroqAgent:
    def __init__(self, system=None):
        self.messages = []
        if system is None:
            system = """
            Corres en un ciclo de Pensamiento, Acción, PAUSA, Observación.
            Al final del ciclo, das una Respuesta.
            Después de recibir una observación, siempre debes responder con una línea que comience con "Respuesta:".
            Si no tienes información, responde con "Respuesta: No tengo la información del curriculum disponible."
            Si tienes información, resume los puntos clave comenzando la línea con "Respuesta:".
            Si no se nombra a ninguna persona, debes usar el curriculum de Emma Johnson.

            Tus acciones disponibles son:

            buscar_cv:
            Ejemplo: buscar_cv: Juan Perez
            Devuelve la información relevante del curriculum de esa persona.

            buscar_multi_cv:
            Ejemplo: buscar_multi_cv: Juan Perez, Maria Garcia
            Devuelve la información relevante de los curriculums de ambas personas.

            Ejemplo de sesión:

            Pregunta: ¿Qué experiencia tiene Juan Perez?
            Pensamiento: Debería buscar el curriculum de Juan Perez.
            Acción: buscar_cv: Juan Perez
            PAUSA

            (Esperar observación)

            Observación: Juan Perez trabajó como ingeniero en Google y Tesla.

            Respuesta: Juan Perez tiene experiencia como ingeniero en Google y Tesla.
            """.strip()
        self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        print(f"[GroqAgent] Usuario: {message}")
        result, tokens_info = self.execute()
        print(f"[GroqAgent] Asistente: {result}")
        self.messages.append({"role": "assistant", "content": result})
        return result, tokens_info

    def execute(self):
        # print("[GroqAgent] Enviando la consulta a Groq...")
        # print("[GroqAgent] Mensajes enviados:")
        # print(json.dumps(self.messages, indent=2, ensure_ascii=False))
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=self.messages,
            temperature=0.7,
        )
        result = response.choices[0].message.content

        # Capturar información de tokens
        tokens_info = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return result, tokens_info


# Ejecutar un ciclo de interaccion con un agente
def query_with_agent(agent, pregunta, max_turns=5):
    i = 0
    next_prompt = pregunta
    all_tokens = []

    acciones_disponibles = {
        "buscar_cv": buscar_cv,
        "buscar_multi_cv": buscar_multi_cv
    }
    action_re = re.compile(r'^Acción: (\w+): (.*)$')
    respuesta_re = re.compile(r'^Respuesta: (.*)$')

    while i < max_turns:
        i += 1
        print(f"-----------")
        print(f"Iteracion {i}")
        print(f"-----------")
        result, tokens_info = agent(next_prompt)
        all_tokens.append(tokens_info)

        # Buscar las acciones en la respuesta
        acciones = [
            action_re.match(a)
            for a in result.split('\n')
            if action_re.match(a)
        ]

        respuestas = [
            respuesta_re.match(a)
            for a in result.split('\n')
            if respuesta_re.match(a)
        ]

        if respuestas:
            total_tokens = sum(t['total_tokens'] for t in all_tokens)
            return respuestas[0].group(1), total_tokens

        if acciones:
            accion, accion_input = acciones[0].groups()

            if accion not in acciones_disponibles:
                print(f"Acción desconocida: {accion}")
                next_prompt = f"Observación: Acción '{accion}' no reconocida. Fin"
                continue

            # Ejecutar la acción correspondiente
            observacion = acciones_disponibles[accion](accion_input)
            time.sleep(2)
            next_prompt = f"Observación: {observacion}"
        else:
            break
    total_tokens = sum(t['total_tokens'] for t in all_tokens)
    return f"No se obtuvo respuesta satisfactoria después de {max_turns} iteraciones.", total_tokens
