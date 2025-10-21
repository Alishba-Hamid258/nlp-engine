from fastapi import APIRouter
from pydantic import BaseModel
from models.nlp_models import process_text

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/analyze")
def analyze_text(input: TextInput):
    result = process_text(input.text)
    return result