"""
Automated Model & Database Downloader from Hugging Face:
- models/classifier_v1/ (DeBERTa backbone, head.pth, config.json)
- models/policy_network_v1.pth (DQN PyTorch policy weights)
- data/tutor_memory.db (Pre-populated SQLite database with 11 concepts and analytics)
"""

import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("Error: huggingface_hub is not installed. Please run:")
    print("pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "Sambarlasagna/cognitive-offloading-models"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

def main():
    print("=" * 60)
    print(f"Downloading model & database artifacts from:")
    print(f"https://huggingface.co/{REPO_ID}")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Downloading Classifier & DQN Policy Weights into models/...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
        ignore_patterns=[".git*", "README.md", "data/*"]
    )

    db_path = DATA_DIR / "tutor_memory.db"
    print("\n[2/2] Checking SQLite Database...")
    if not db_path.exists():
        print(f"Downloading pre-populated tutor_memory.db to {db_path}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename="data/tutor_memory.db",
            local_dir=str(PROJECT_ROOT),
            local_dir_use_symlinks=False
        )
    else:
        print(f"Local database {db_path} already exists. Kept local version.")

    print("\n" + "=" * 60)
    print("Setup verified successfully!")
    print(f"Classifier: {MODELS_DIR / 'classifier_v1'}")
    print(f"Policy:     {MODELS_DIR / 'policy_network_v1.pth'}")
    print(f"Database:   {db_path}")
    print("=" * 60)
    print("\nYou can now launch the mentor with:")
    print("  python fastapi_app.py")

if __name__ == "__main__":
    main()
