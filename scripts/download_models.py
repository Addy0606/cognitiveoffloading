"""Download pre-trained models from Hugging Face into the local models/ directory.

Usage:
    python scripts/download_models.py
"""

from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "addy0606/cognitive-offloading"
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


def main():
    print(f"Downloading model artifacts from Hugging Face ({REPO_ID})...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(MODELS_DIR),
        ignore_patterns=["*.git*", "README.md"],
    )

    print(f"\nAll models downloaded successfully into: {local_path}")
    print("Files available:")
    for path in sorted(MODELS_DIR.rglob("*")):
        if path.is_file():
            print(f"  - {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
