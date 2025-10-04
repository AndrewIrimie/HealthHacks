# file: stt_to_llm_local.py
import asyncio
from pipecat.audio.devices import Microphone
from pipecat.runtime import Pipeline, Event, on
from pipecat.services.ultravox.stt import UltravoxSTTService

# ---- Local LLM (Transformers) ----
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, pipeline

# Load a small local instruct model you have (replace with your local path or HF id).
# Examples: "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" (GPU) or a small 3B–8B instruct model.
LLM_ID = "/path/to/your/local-instruct-model"   # or "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(LLM_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    torch_dtype="auto",
    device_map="auto"   # CPU or GPU; adjust to your hardware
)
llm: TextGenerationPipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.2,
    do_sample=False
)

SYSTEM_PROMPT = (
    "You are a concise reasoning assistant. "
    "Given the user's last utterance, think briefly and reply with a clear, helpful answer."
)

# ---- Pipecat STT (Ultravox local) + Mic ----
mic = Microphone(sample_rate=16000, channels=1)
stt = UltravoxSTTService(   # runs Ultravox locally; no external API
    sample_rate=16000,
    enable_interim=False,   # set True if you want partials; we'll use finalized segments
)

pipeline = Pipeline(sources=[mic], stages=[stt])

@on(pipeline, Event.TRANSCRIPT)  # fired when STT yields a transcript segment
async def handle_transcript(evt: Event):
    text = evt.data.get("text", "").strip()
    is_final = evt.data.get("final", True)
    if not text:
        return
    if not is_final:
        # if you enabled interim, you could show partials here
        return

    print(f"\n[User]: {text}")

    prompt = f"{SYSTEM_PROMPT}\n\nUser: {text}\nAssistant:"
    out = llm(prompt)[0]["generated_text"]
    # Some models echo the prompt; strip it:
    reply = out.split("Assistant:")[-1].strip()
    print(f"[LLM ]: {reply}\n")

async def main():
    async with pipeline:
        print("🎙️  Speak into the mic. Ctrl+C to stop.")
        await pipeline.start()
        # keep running until interrupted
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
