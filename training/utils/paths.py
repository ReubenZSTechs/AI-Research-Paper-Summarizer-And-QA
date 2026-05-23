from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TRAINING_DIR = ROOT / "training" / "outputs"

OLLAMA_MODELS_DIR = ROOT / "deployment" / "ollama" / "modelfiles"

LLAMA_CPP_DIR = ROOT / "llama.cpp"

GGUF_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"