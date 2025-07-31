from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai
import os

app = FastAPI()

# Tu API key
openai.api_key = "sk-proj-..."  # ← Reemplaza por tu key real

class Entrada(BaseModel):
    mensaje: str
    usuario: str

@app.post("/responder")
async def responder(data: Entrada):
    prompt_sistema = (
        "Actúa como Wilder Escobar, Representante a la Cámara en Colombia. "
        "Eres empático, directo, conectado con la comunidad. Agradeces cada mensaje, "
        "respondes como ser humano real, y motivas a seguir participando."
    )

    respuesta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": data.mensaje}
        ]
    )

    contenido = respuesta['choices'][0]['message']['content']
    return {"respuesta": contenido}
