"""
Helper script to download trained model artifacts from Hugging Face:
- models/classifier_v1/ (DeBERTa backbone, head.pth, config.json)
- models/policy_network_v1.pth (DQN PyTorch policy weights)
"""

import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is not installed. Please run:")
    print("pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "Sambarlasagna/cognitive-offloading-models"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

def main():
    print("=" * 60)
    print(f"Downloading model weights from https://huggingface.co/{REPO_ID}")
    print(f"Destination: {MODELS_DIR}")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
        ignore_patterns=[".git*", "README.md"]
    )

    print("\nModel download complete!")
    print(f"Classifier artifacts: {MODELS_DIR / 'classifier_v1'}")
    print(f"Policy weights:       {MODELS_DIR / 'policy_network_v1.pth'}")
    print("\nYou can now start the server with: python fastapi_app.py")

if __name__ == "__main__":
    main()
