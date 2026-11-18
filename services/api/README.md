# PianoForge API

A local demo service for the first roadmap scenario. `POST /demo/transcriptions` accepts an audio file and returns a prepared piano score. It does not transcribe the upload.

## Install

```bash
cd services/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

## Check

```bash
curl -sS -D - -o /tmp/demo-score.musicxml \
  -F "audio=@/path/to/audio.wav" \
  http://127.0.0.1:8000/demo/transcriptions
```

A successful response is HTTP 200 with `Content-Type: application/xml`. The body is MusicXML whose root element is `score-partwise`.
