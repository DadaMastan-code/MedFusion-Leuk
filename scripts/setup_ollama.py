"""
One-time setup script: downloads required Ollama models for MedFusion-Leuk.
Run once after installing Ollama.

Usage:
    python scripts/setup_ollama.py

Models downloaded:
    meditron  — Medical-domain fine-tuned LLaMA (~7B, ~4GB)
    llama3    — General fallback LLM (~8B, ~4.7GB)
"""

import sys
import subprocess


MODELS = [
    ("meditron", "Medical-focused LLM — primary model (~4GB)"),
    ("llama3",   "General LLM — fallback model (~4.7GB)"),
]


def check_ollama_installed() -> bool:
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def pull_model(name: str) -> bool:
    print(f"  Pulling {name} (this may take several minutes on first run) …")
    try:
        subprocess.run(["ollama", "pull", name], check=True)
        print(f"  {name} downloaded successfully.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed to download {name}: {e}\n")
        return False


def list_available() -> list[str]:
    try:
        import ollama
        return [m["model"] for m in ollama.list().get("models", [])]
    except Exception:
        return []


def setup_ollama():
    print("=" * 55)
    print("  MedFusion-Leuk — Ollama Model Setup")
    print("=" * 55)

    if not check_ollama_installed():
        print("\nOllama is not installed or not on PATH.")
        print("Install it first:")
        print("  Mac/Linux : curl -fsSL https://ollama.ai/install.sh | sh")
        print("  Windows   : https://ollama.ai/download")
        sys.exit(1)

    print("\nOllama detected. Checking for required models …\n")
    already_have = list_available()

    all_ok = True
    for model_name, description in MODELS:
        if any(model_name in m for m in already_have):
            print(f"  [{model_name}] already present — skipping.")
        else:
            print(f"\n  [{model_name}] {description}")
            ok = pull_model(model_name)
            if not ok:
                all_ok = False

    print("\n" + "=" * 55)
    if all_ok:
        print("All models ready.")
        print("\nNext steps:")
        print("  1. ollama serve          (keep running in background)")
        print("  2. uvicorn app.main:app --reload")
        print("  3. streamlit run frontend/streamlit_app.py")
    else:
        print("Some models failed to download. Check your internet connection.")
        print("You can retry by running this script again.")
    print("=" * 55)


if __name__ == "__main__":
    setup_ollama()
