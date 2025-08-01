from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

# Inicializar cliente OpenAI con clave desde variable de entorno
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Entrada(BaseModel):
    mensaje: str
    usuario: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/responder")
async def responder(data: Entrada):
    prompt_sistema = (
        "Actúa como Wilder Escobar, Representante a la Cámara en Colombia. "
        "Eres empático, directo, conectado con la comunidad. Agradeces cada mensaje, "
        "respondes como ser humano real, y motivas a seguir participando."
    )

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": data.mensaje}
            ]
        )
        return {"respuesta": chat_completion.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
