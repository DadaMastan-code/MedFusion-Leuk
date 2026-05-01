"""FastAPI application entry point for MedFusion-Leuk."""

import asyncio
import ollama
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.api.image_route import router as image_router
from app.api.pdf_route import router as pdf_router
from app.api.nlp_route import router as nlp_router
from app.api.predict_route import router as predict_router
from config import API_HOST, API_PORT, STATIC_DIR, OLLAMA_MODEL, OLLAMA_FALLBACK_MODEL

load_dotenv()

app = FastAPI(
    title="MedFusion-Leuk API",
    description="Multimodal Fusion Architecture for Leukemia Clinical Decision Support",
    version="1.0.0",
)

# Serve static files (SHAP plots, Grad-CAM overlays)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(image_router)
app.include_router(pdf_router)
app.include_router(nlp_router)
app.include_router(predict_router)


@app.on_event("startup")
async def startup_event():
    """Load all ML models at startup so first request has no lazy-init delay."""
    # Ollama check
    try:
        available = [m["model"] for m in ollama.list().get("models", [])]
        available_flat = " ".join(available)
        if OLLAMA_MODEL in available_flat:
            print(f"[Ollama] '{OLLAMA_MODEL}' is ready.")
        elif OLLAMA_FALLBACK_MODEL in available_flat:
            print(f"[Ollama] Fallback '{OLLAMA_FALLBACK_MODEL}' active.")
        else:
            print(f"[Ollama] WARNING: No model found. Run: python scripts/setup_ollama.py")
    except Exception as e:
        print(f"[Ollama] ERROR: {e}")

    # Models load lazily on first request — nothing to do here.
    print("[Models] Lazy loading enabled — models initialize on first request.")


@app.get("/health")
def health():
    """Health check endpoint including Ollama model status."""
    try:
        available = [m["model"] for m in ollama.list().get("models", [])]
        llm_status = "ready" if any(OLLAMA_MODEL in m for m in available) else "model_missing"
    except Exception:
        llm_status = "ollama_offline"
    return {
        "status": "ok",
        "system": "MedFusion-Leuk",
        "llm_backend": "ollama (local)",
        "llm_model": OLLAMA_MODEL,
        "llm_status": llm_status,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=True)
