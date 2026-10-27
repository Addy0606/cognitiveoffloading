import os
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "addy0606/cognitive-offloading"
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

README_CONTENT = """---
license: mit
library_name: transformers
tags:
- reinforcement-learning
- cognitive-offloading
- education
- dqn
- deberta
- pytorch
---

# Cognitive Offloading Detection & Adaptive Pedagogical Policy Models

Model artifacts for the [Cognitive Offloading Prevention System](https://github.com/Addy0606/cognitiveoffloading).

## Contents
- **`classifier_v1/`**: `microsoft/deberta-base` fine-tuned to classify student messages for cognitive offloading / answer extraction behavior.
  - `backbone/`: Tokenizer files and transformer base weights (`model.safetensors`).
  - `head.pth`: 2-layer MLP classification head (`768 -> 256 -> 64 -> 2`).
  - `config.json`: Model configuration and hyperparameters.
- **`policy_network_v1.pth`**: Deep Q-Network (DQN) policy trained via reinforcement learning to select dynamic pedagogical intervention actions (`redirect`, `scaffold`, `socratic`, `delay`) balancing learning gains against engagement friction.

## Usage
To download these models locally:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="addy0606/cognitive-offloading",
    local_dir="models",
)
```

## GitHub Repository
Source code, evaluation pipeline, and Flask interface:
https://github.com/Addy0606/cognitiveoffloading
"""

def main():
    api = HfApi()
    print(f"Checking/creating repo: {REPO_ID}...")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print("Repo ready.")

    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=README_CONTENT.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add model card README"
    )

    print("Uploading policy_network_v1.pth...")
    policy_path = MODELS_DIR / "policy_network_v1.pth"
    if policy_path.exists():
        api.upload_file(
            path_or_fileobj=str(policy_path),
            path_in_repo="policy_network_v1.pth",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload RL DQN policy network weights"
        )
        print("policy_network_v1.pth uploaded.")
    else:
        print(f"Warning: {policy_path} not found.")

    print("Uploading classifier_v1 folder (including backbone and head)...")
    classifier_dir = MODELS_DIR / "classifier_v1"
    if classifier_dir.exists():
        api.upload_folder(
            folder_path=str(classifier_dir),
            path_in_repo="classifier_v1",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload classifier_v1 backbone, tokenizer, and head weights"
        )
        print("classifier_v1 folder uploaded.")
    else:
        print(f"Warning: {classifier_dir} not found.")

    print(f"\nUpload complete! Model URL: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
