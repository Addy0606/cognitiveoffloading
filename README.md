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

## Quick Start

### Prerequisites

- Python 3.10+ (tested with 3.10.12)
- PyTorch 2.x
- conda environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/Addy0606/cognitiveoffloading.git
cd cognitiveoffloading

# Create and activate conda environment
conda activate rl

# Install dependencies
pip install -r requirements.txt

# Copy environment variables template
cp .env.example .env
# Edit .env to add your OpenAI API key (optional - fallback responses work without it)
```

### Running the Pipeline

```bash
# Step 0: Prepare data (generates synthetic data for testing)
python scripts/0_prepare_data.py

# Step 1b: Create train/val/test splits
python scripts/1b_create_splits.py

# Step 2: Train the classifier
python scripts/2_train_classifier.py

# Step 3: Run causal analysis
python scripts/3_causal_analysis.py

# Step 4: Train the RL policy
python scripts/4_train_rl_policy.py

# Step 5: Evaluate
python scripts/5_evaluate.py

# Or run everything at once:
python scripts/run_all.py
```

### Interactive Demo

```bash
python scripts/demo.py
```

### Running Tests

```bash
# Run fast tests (no model download required)
python -m pytest tests/ -v

# Run all tests including slow tests (requires DeBERTa download)
python -m pytest tests/ -v -m "slow"

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

## Project Structure

```
cognitiveoffloading/
├── src/                        # Core source modules
│   ├── __init__.py
│   ├── adaptive_tutor.py       # End-to-end inference pipeline
│   ├── classifier.py           # DeBERTa extraction/learning classifier
│   ├── engagement_scorer.py    # Heuristic engagement scoring
│   ├── prompts.py              # Pedagogical intervention prompts
│   ├── rl_policy_network.py    # Q-learning policy network + trainer
│   └── utils.py                # Data loading, state encoding, LLM, helpers
├── scripts/                    # Pipeline and analysis scripts
│   ├── 0_prepare_data.py       # Data preparation (synthetic or real)
│   ├── 1_label_data.py         # Interactive annotation interface
│   ├── 1b_create_splits.py     # Stratified train/val/test splitting
│   ├── 2_train_classifier.py   # Classifier fine-tuning
│   ├── 3_causal_analysis.py    # Tutor→student pattern analysis
│   ├── 4_train_rl_policy.py    # RL policy training
│   ├── 5_evaluate.py           # Evaluation metrics
│   ├── demo.py                 # Interactive demo
│   └── run_all.py              # Full pipeline orchestration
├── tests/                      # Test suite (162 tests)
│   ├── conftest.py             # Shared fixtures
│   ├── test_engagement.py      # Engagement scorer tests
│   ├── test_classifier.py      # Classifier tests
│   ├── test_policy.py          # RL policy tests
│   ├── test_state_encoder.py   # State encoder tests
│   ├── test_utils.py           # Utility function tests
│   └── test_integration.py     # Integration tests
├── data/                       # Data directory (generated)
├── models/                     # Saved model checkpoints
├── results/                    # Evaluation results and plots
├── notebooks/                  # Jupyter notebooks
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
