from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatbotRequest(BaseModel):
    domain: str

@app.post("/create-chatbot")
async def create_chatbot(req: ChatbotRequest):
    url = f"https://yourchatbotsite.com/{req.domain.replace('.', '-')}"
    return {"chatbot_url": url}
