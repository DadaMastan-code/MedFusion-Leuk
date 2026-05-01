#!/usr/bin/env bash
# MedFusion-Leuk — Full system startup
# Usage: ./scripts/start_system.sh
# Starts FastAPI (port 8000) + Streamlit UI (port 8501)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "==========================================="
echo " MedFusion-Leuk — Clinical Decision Support"
echo "==========================================="
echo ""

# Verify model files exist
if [ ! -f "models/efficientnet_vit/best_model.pth" ]; then
    echo "ERROR: Image model not found. Run: python scripts/train_image_model.py"
    exit 1
fi
if [ ! -f "models/xgboost/fusion_classifier.json" ]; then
    echo "ERROR: Fusion classifier not found. Run: python scripts/train_fusion_classifier.py"
    exit 1
fi

# Check Ollama is running
if ! ollama list &>/dev/null; then
    echo "WARNING: Ollama not responding. LLM explanations will use fallback text."
    echo "         To enable: run 'ollama serve' and 'ollama pull meditron' in a separate terminal."
fi

# Start FastAPI in background
echo "Starting FastAPI backend (http://localhost:8000)..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
FASTAPI_PID=$!
echo "FastAPI started (PID $FASTAPI_PID)"

# Wait for API to be ready
echo "Waiting for API startup..."
for i in {1..15}; do
    if curl -sf http://localhost:8000/health &>/dev/null; then
        echo "API ready: $(curl -s http://localhost:8000/health | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d[\"llm_status\"])')"
        break
    fi
    sleep 1
done

echo ""
echo "Starting Streamlit UI (http://localhost:8501)..."
echo ""
echo "-------------------------------------------"
echo "  FastAPI:    http://localhost:8000"
echo "  API docs:   http://localhost:8000/docs"
echo "  Streamlit:  http://localhost:8501"
echo "-------------------------------------------"
echo ""

# Run Streamlit in foreground — Ctrl+C here stops everything
trap "echo ''; echo 'Shutting down...'; kill $FASTAPI_PID 2>/dev/null; exit 0" INT TERM

streamlit run frontend/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false

kill $FASTAPI_PID 2>/dev/null
