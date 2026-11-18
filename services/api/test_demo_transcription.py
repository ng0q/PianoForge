import io
import wave
import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)
    return buffer.getvalue()


def test_demo_transcription_returns_musicxml() -> None:
    response = client.post(
        "/demo/transcriptions",
        files={"audio": ("demo.wav", _wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 200
    media_type = response.headers["content-type"].split(";", 1)[0].strip()
    assert media_type == "application/xml"

    root = ET.fromstring(response.content)
    assert root.tag == "score-partwise"
    notes_with_pitch = [
        note for note in root.iter("note") if note.find("pitch") is not None
    ]
    assert notes_with_pitch
