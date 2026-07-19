
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI # this library for Groq
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client 
client = OpenAI(
    api_key=os.getenv("groq_api_key"),
    base_url="https://api.groq.com/openai/v1"
)

class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[
            {"role": "system", "content": "You are a professional business jargon translator. Transform the user's casual sentence (even if it's nonsense) into a visionary, motivational corporate sentence using business jargon (just like we see in Linkedin). Also, don't forget to recognize social media abbreviations if user uses any. Your response shouldn't include social media abbreviations. Your response ONLY should be translated corporate text directly, you shouldn't add what you are going to do or your personal comments into your response. If the user is complaining or rejecting a task, phrase it as 'optimizing priorities' or 're-evaluating bandwidth' instead of pretending they are excited to do it. Response MUST be in English."},
            {"role": "user", "content": request.text}
        ]
    )
    return {"translated_text": response.choices[0].message.content}

if __name__ == "__main__":
    import uvicorn
    import os
    # Render portu dinamik verir, bulamazsa 8000 açar
    port = int(os.getenv("PORT", 8000)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port)
