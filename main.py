from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import sys
import os

# --- Add Services folder to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services'))
from Services import RAG

app = FastAPI()

# --- Setup ---
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
templates = Jinja2Templates(directory="templates")


# --- Pydantic Models ---

class InitialChatPayload(BaseModel):
    age: str
    emotions: List[str]
    intensity: str

class ChatTurnPayload(BaseModel):
    user_data: InitialChatPayload
    user_query: str
    history: List[Dict[str, str]]


# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_landing_page(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})


# **FIXED**: Changed the route from "/chat" to "/chat.html" to match the browser's request.
@app.get("/chat.html", response_class=HTMLResponse)
async def serve_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/start-chat")
async def start_chat(payload: InitialChatPayload):
    """
    Handles the form submission. No initial message is expected.
    """
    print("Received initial data. Generating first response...")
    try:
        initial_history = []
        rag_response = RAG.get_rag_response(
            user_data=payload.dict(),
            history=initial_history,
            user_query=None
        )
        return JSONResponse({
            "ai_response": rag_response,
            "user_data": payload.dict()
        })
    except Exception as e:
        print(f"Error during /start-chat: {e}")
        raise HTTPException(status_code=500, detail="Error generating initial AI response.")


@app.post("/chat-turn")
async def chat_turn(payload: ChatTurnPayload):
    """
    Handles each subsequent message from the chat interface.
    """
    print("Received new message. Generating next response...")
    try:
        rag_response = RAG.get_rag_response(
            user_data=payload.user_data.dict(),
            history=payload.history,
            user_query=payload.user_query
        )
        return JSONResponse({"ai_response": rag_response})
    except Exception as e:
        print(f"Error during /chat-turn: {e}")
        raise HTTPException(status_code=500, detail="Error generating AI response.")


# --- To run the app ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)









