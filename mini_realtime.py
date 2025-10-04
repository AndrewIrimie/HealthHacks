# mini_realtime.py
import asyncio
from loguru import logger
import aiohttp
import os

# --- Pipecat core ---
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner

# Local audio I/O (mic + speakers)
from pipecat.transports.local.audio import (
    LocalAudioTransport, LocalAudioTransportParams
)

# Frames (STT out / TTS in / start)
from pipecat.frames.frames import (
    StartFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,   # text we want TTS to speak
)

# STT: Ultravox (runs locally; see docs for model setup)
from pipecat.services.ultravox.stt import UltravoxSTTService

# TTS: Piper over HTTP (you run a local Piper HTTP server)
from pipecat.services.piper.tts import PiperTTSService


# ---------- Simple “assistant” logic ----------
def make_reply(user_text: str) -> str:
    """
    Keep this stupid-simple for now: parrot back with a tiny flourish.
    Swap this out for your LLM later.
    """
    user_text = user_text.strip()
    if not user_text:
        return "I didn't quite catch that. Please try again."
    if user_text.endswith("?"):
        return f"You asked: {user_text}"
    return f"You said: {user_text}"


async def main():
    # ----------- config -----------
    # Piper HTTP server base URL; set PIPER_URL if yours differs
    piper_url = os.environ.get("PIPER_URL", "http://localhost:5002")
    # Sample rates:
    # - Ultravox STT prefers 16000 Hz mic input
    # - Piper voice models vary; 22050 is a common default
    audio_in_sr = 16000
    audio_out_sr = int(os.environ.get("AUDIO_OUT_SR", "22050"))

    # ----------- local audio -----------
    transport = LocalAudioTransport(LocalAudioTransportParams())

    # ----------- STT (Ultravox) -----------
    stt = UltravoxSTTService(
        sample_rate=audio_in_sr,
        enable_interim=True,  # change to False if you don't want live partials
    )

    # ----------- TTS (Piper over HTTP) -----------
    # PiperTTSService needs an aiohttp session
    session = aiohttp.ClientSession()
    tts = PiperTTSService(base_url=piper_url, aiohttp_session=session)
    # NOTE: Voice/language is chosen by your Piper server config.
    # If your server lets you select a voice via query params, configure it there.

    # ----------- pipeline -----------
    # Order matters: mic input -> STT -> TTS -> speaker output
    pipeline = Pipeline([
        transport.input(),
        stt,
        tts,
        transport.output(),  # plays TTSAudioRawFrame out the speakers
    ])

    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    # ----------- events -----------
    @task.event_handler("on_frame_reached_downstream")
    async def on_frames(_task, frame):
        # show partials quietly
        if isinstance(frame, InterimTranscriptionFrame):
            logger.debug(f"[interim] {frame.text}")
            return

        # when we get a final transcript, synthesize a spoken reply
        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if not text:
                return
            logger.info(f"[final] {text}")

            reply = make_reply(text)
            logger.info(f"[bot  ] {reply}")

            # Send text to TTS (Piper) → audio frames → transport.output()
            await task.queue_frame(TTSSpeakFrame(reply))

    # Kick off with audio format details
    await task.queue_frame(StartFrame(
        audio_in_sample_rate=audio_in_sr,
        audio_out_sample_rate=audio_out_sr
    ))

    logger.info("🎙️ Speak into your mic. I will talk back. Press Ctrl+C to quit.")
    try:
        await runner.run(task)
    except KeyboardInterrupt:
        pass
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
