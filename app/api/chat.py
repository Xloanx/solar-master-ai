from fastapi import APIRouter
from pydantic import BaseModel
from app.services.ai_advisor import query_advisor

router = APIRouter()

class ChatInput(BaseModel):
    question: str
    user_id: str | None = None  # Optional if not logged in

@router.post("/")
def ask_ai(data: ChatInput):
    response = query_advisor(data.question, user_id=data.user_id)
    return {"response": response}
