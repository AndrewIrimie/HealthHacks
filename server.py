# server.py
import io
import os
import time
import wave
from datetime import datetime

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

app = FastAPI()
os.makedirs("uploads", exist_ok=True)

@app.get("/")
def root():
    return PlainTextResponse("POST a WAV file to /upload with form field name 'file'.")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    t0 = time.time()
    raw = await file.read()
    size = len(raw)

    # Inspect WAV header (will throw if not a valid WAV)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.getnframes()
        duration = frames / float(sr) if sr else 0.0

    # -------- Save block (fixed .wav.wav issue) --------
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    orig = file.filename or "audio.wav"
    root, ext = os.path.splitext(orig)
    if not ext:          # ensure it ends with .wav if no extension
        ext = ".wav"
    out_path = os.path.join("uploads", f"{ts}-{root}{ext}")
    with open(out_path, "wb") as f:
        f.write(raw)
    # ---------------------------------------------------

    print(
        f"[UPLOAD] file={file.filename} size={size}B "
        f"sr={sr} ch={ch} sampwidth={sw} frames={frames} "
        f"duration={duration:.3f}s -> saved={out_path}"
    )

    return JSONResponse({
        "ok": True,
        "filename": file.filename,
        "bytes": size,
        "sample_rate": sr,
        "channels": ch,
        "sample_width": sw,
        "frames": frames,
        "duration_sec": round(duration, 3),
        "saved": out_path,
        "elapsed_ms": int((time.time() - t0) * 1000),
    })

