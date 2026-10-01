from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found")

client = genai.Client(api_key=api_key)


class GenerateRequest(BaseModel):
    prompt: str


@app.get("/")
def home():
    return {"message": "Gemini API server is running"}


@app.post("/generate")
def generate_text(data: GenerateRequest):

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=data.prompt
        )

        return {
            "prompt": data.prompt,
            "response": response.text
        }

    except Exception as e:
        print("ERROR FROM GEMINI:", repr(e))

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )