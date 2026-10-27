# Cognitive Offloading: RL-Based Adaptive Tutoring System

An adaptive AI tutoring system that detects whether a student is seeking direct answers (cognitive offloading / extraction) rather than trying to understand a concept, then uses a trained reinforcement learning policy to choose an appropriate pedagogical intervention.

## Architecture

```
Student Message
    ↓
[DeBERTa Classifier] → extraction_score (0-1)
    ↓
[Engagement Scorer] → engagement_score (0-1)
    ↓
[State Encoder] → 771-dim vector (768 embedding + 3 scalar features)
    ↓
[Q-Learning Policy Network] → 4 Q-values
    ↓
[Action Selection] → REDIRECT | SCAFFOLD | SOCRATIC | DELAY
    ↓
[Prompt Generator + LLM] → Tutoring Response
```

### Components

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **Extraction Classifier** | DeBERTa-base fine-tuned for extraction vs. learning detection | `src/classifier.py` |
| **Engagement Scorer** | Heuristic scorer combining reasoning, length, questions, and context signals | `src/engagement_scorer.py` |
| **State Encoder** | Constructs 771-dim state vector from embedding + scalar features | `src/utils.py` |
| **RL Policy** | Q-learning network selecting from 4 intervention actions | `src/rl_policy_network.py` |
| **Adaptive Tutor** | End-to-end inference pipeline combining all components | `src/adaptive_tutor.py` |
| **Prompt System** | Four pedagogical intervention prompts + baseline | `src/prompts.py` |

### Intervention Actions

| Action | ID | When Used |
|--------|----|-----------|
| **REDIRECT** | 0 | Student seeking direct answers — redirect toward independent thinking |
| **SCAFFOLD** | 1 | Student showing effort but needs structured guidance |
| **SOCRATIC** | 2 | Student capable of reasoning with guided questions |
| **DELAY** | 3 | Student frustrated or fatigued — suggest a strategic pause |

## Models & Weights (Hugging Face)

Pretrained model weights (fine-tuned DeBERTa classifier and trained RL DQN policy network) are hosted on Hugging Face:
🤗 **[addy0606/cognitive-offloading](https://huggingface.co/addy0606/cognitive-offloading)**

You can download them with a single command without needing to retrain from scratch:
```bash
python scripts/download_models.py
```

---

## Setup & Running on Another PC

Follow these steps to set up and run the complete project on any machine (Windows, macOS, Linux).

### 1. Prerequisites
- **Python 3.10+** (tested on 3.10 and 3.11)
- **Git**
- *(Optional for AI response generation)*: **[Ollama](https://ollama.com/)** running locally (e.g. `qwen2.5:3b`) or an **OpenAI API key**. If neither is provided, the system automatically uses built-in offline pedagogical response templates.

---

### 2. Clone the Repository
```bash
git clone https://github.com/Addy0606/cognitiveoffloading.git
cd cognitiveoffloading
```

---

### 3. Create & Activate a Virtual Environment

**Using Python `venv`:**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**Or using Conda:**
```bash
conda create -n rl python=3.10 -y
conda activate rl
```

---

### 4. Install Dependencies

```bash
# If you have an NVIDIA GPU (optional, for CUDA acceleration):
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install required dependencies:
pip install -r requirements.txt
```

---

### 5. Download Pre-trained Models

Download the DeBERTa extraction classifier backbone, classification head, and RL DQN policy network directly from Hugging Face:

```bash
python scripts/download_models.py
```

---

### 6. Configure Environment Variables

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` if desired:
* To use free local LLM: ensure Ollama is running (`ollama run qwen2.5:3b`) and keep `LLM_PROVIDER=ollama`.
* To use OpenAI: set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=your_key`.
* To run completely offline without an LLM: set `LLM_PROVIDER=fallback`.

---

### 7. Run the Application

#### A. Interactive Web Interface (Flask)
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser to chat with the adaptive tutor, view real-time cognitive offloading probability bars, inspect state vectors, and observe RL action selections live.

#### B. Terminal Interactive Demo
```bash
python scripts/demo.py
```

#### C. Run the Evaluation Benchmark
```bash
python scripts/5_evaluate.py
```

---

### 8. Running the Full Training Pipeline (Optional)

If you wish to retrain all models from scratch:

```bash
# Run complete end-to-end training and evaluation:
python scripts/run_all.py

# Or step-by-step:
python scripts/0_prepare_data.py       # Generate synthetic dialogue data
python scripts/1b_create_splits.py     # Create train/val/test splits
python scripts/2_train_classifier.py   # Fine-tune DeBERTa extraction classifier
python scripts/3_causal_analysis.py    # Run causal analysis on tutor transitions
python scripts/4_train_rl_policy.py    # Train RL DQN policy
python scripts/5_evaluate.py           # Evaluate against baselines
```

---

### 9. Running Tests

```bash
# Run complete test suite (162 tests)
pytest tests/ -v
```

## Project Structure

```
cognitiveoffloading/
├── app.py                      # Flask interactive web UI
├── static/                     # Web UI styles & frontend client
│   ├── app.js
│   └── style.css
├── templates/                  # Web UI HTML templates
│   └── index.html
├── src/                        # Core source modules
│   ├── __init__.py
│   ├── adaptive_tutor.py       # End-to-end inference pipeline
│   ├── classifier.py           # DeBERTa extraction/learning classifier
│   ├── engagement_scorer.py    # Heuristic engagement scoring
│   ├── prompts.py              # Pedagogical intervention prompts
│   ├── rl_policy_network.py    # Q-learning policy network + trainer
│   └── utils.py                # Data loading, state encoding, LLM, helpers
├── scripts/                    # Pipeline and utility scripts
│   ├── download_models.py      # Download models from Hugging Face Hub
│   ├── upload_to_hf.py         # Upload models & card to Hugging Face
│   ├── 0_prepare_data.py       # Data preparation (synthetic or real)
│   ├── 1_label_data.py         # Interactive annotation interface
│   ├── 1b_create_splits.py     # Stratified train/val/test splitting
│   ├── 2_train_classifier.py   # Classifier fine-tuning
│   ├── 3_causal_analysis.py    # Tutor→student pattern analysis
│   ├── 4_train_rl_policy.py    # RL policy training
│   ├── 5_evaluate.py           # Evaluation metrics
│   ├── demo.py                 # Interactive terminal demo
│   └── run_all.py              # Full pipeline orchestration
├── tests/                      # Test suite (162 unit & integration tests)
│   ├── conftest.py             # Shared fixtures
│   ├── test_engagement.py      # Engagement scorer tests
│   ├── test_classifier.py      # Classifier tests
│   ├── test_policy.py          # RL policy tests
│   ├── test_state_encoder.py   # State encoder tests
│   ├── test_utils.py           # Utility function tests
│   └── test_integration.py     # Integration tests
├── data/                       # Dataset splits (train/val/test)
├── models/                     # Model weights (DeBERTa backbone + RL policy)
├── results/                    # Evaluation metrics and causal analysis
├── notebooks/                  # Jupyter exploration notebooks
├── logs/                       # Training logs
├── .env.example                # Environment variable template
├── .gitignore
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
├── IMPLEMENTATION_PLAN.md      # Development plan
├── REPORT.md                   # Technical report
└── IMPLEMENTATION_STATUS.md    # Current implementation status
```

## Data

### Synthetic Data

The system generates synthetic student-tutor dialogues for pipeline testing. Synthetic data is clearly labeled and should **never** be presented as real data.

- **Extraction messages**: Short, direct requests for answers (e.g., "Just give me the code")
- **Learning messages**: Longer messages showing reasoning, context, and conceptual questions

### Real Data

To use real data, place CSV files with columns `dialogue_id`, `turn_num`, `speaker`, `message` in the `data/` directory and re-run the pipeline. The interactive labeling tool (`scripts/1_label_data.py`) can be used to annotate real data.

## Key Technical Decisions

1. **Synthetic data**: The spec references "StudyChat" dataset but no verified public URL exists. Synthetic data is used for pipeline testing with clear warnings.
2. **Offline RL**: The policy learns from observed dialogue transitions. Rewards cannot be causally attributed to hypothetical actions — this is a fundamental limitation clearly documented throughout.
3. **CPU training**: All training runs on CPU. Batch sizes and epochs are tuned accordingly.
4. **LLM fallback**: Template-based responses are available when no OpenAI API key is configured. Fallback responses are clearly labeled as non-LLM-generated.

## Known Limitations

- Classifier trained on synthetic data will not generalize to real student dialogues without real training data
- Offline RL reward attribution is associative, not causal
- Small dataset (500 examples) limits classifier generalization
- CPU-only training constrains DeBERTa fine-tuning speed

## License

This project is for educational and research purposes.
