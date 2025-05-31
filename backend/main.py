from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from googletrans import Translator
import cgi

from processor import SignLanguageProcessor

app = FastAPI(title="Sign Language Translation API")

# Allow local dev React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = SignLanguageProcessor()
translator = Translator()


class FramePayload(BaseModel):
    image: str  # Base64 data URL string


class TranslateRequest(BaseModel):
    text: str
    target_lang: str  # e.g., 'hi', 'es'


@app.post("/start")
def api_start():
    """Reset the current recording session."""
    processor.reset_session()
    return {"status": "recording_started"}


@app.post("/frame")
def api_frame(payload: FramePayload):
    """Endpoint to receive a single frame in base64 format."""
    try:
        processor.add_frame(payload.image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "frame_received"}


@app.post("/stop")
def api_stop():
    """Stop recording and run translation."""
    translation = processor.translate()
    return {"translation": translation}


@app.post("/translate")
def translate_text(req: TranslateRequest):
    try:
        result = translator.translate(req.text, dest=req.target_lang)
        return {"translated_text": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 