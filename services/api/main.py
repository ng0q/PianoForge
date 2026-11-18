from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response

SCORE_PATH = Path(__file__).resolve().parent / "scores" / "demo-piano.musicxml"

app = FastAPI(title="PianoForge API")


@app.post("/demo/transcriptions")
async def create_demo_transcription(audio: UploadFile = File(...)) -> Response:
    """Return a prepared piano score. The uploaded audio is not transcribed."""
    await audio.read()
    return Response(content=SCORE_PATH.read_bytes(), media_type="application/xml")
