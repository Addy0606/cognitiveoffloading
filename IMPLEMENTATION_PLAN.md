# Implementation Plan: RL-Based Adaptive Tutoring System

## Architecture Overview

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

## Environment Assessment

- **Python**: 3.10.12 (conda `rl` env)
- **PyTorch**: 2.8.0 (CPU only)
- **CUDA**: Not available — all training will be CPU-based
- **DeBERTa hidden size**: 768 (verified; matches spec's 771 = 768 + 3)
- **Existing code**: None — fresh workspace with only `PROJECT_SPEC_COMPLETE.md`

## Implementation Stages

### Phase 1: Project Structure & Configuration
- Create directory layout (src/, scripts/, data/, models/, results/, tests/)
- Write requirements.txt with compatible pinned versions
- Create .gitignore, .env.example
- Set up logging configuration

### Phase 2: Core Source Modules
- `src/__init__.py` — package init
- `src/prompts.py` — four intervention system prompts
- `src/engagement_scorer.py` — heuristic engagement scoring
- `src/classifier.py` — DeBERTa extraction/learning classifier
- `src/rl_policy_network.py` — Q-learning policy network + trainer
- `src/adaptive_tutor.py` — end-to-end inference pipeline
- `src/utils.py` — shared helpers (data loading, state encoding, LLM calls)

### Phase 3: Data Pipeline Scripts
- `scripts/0_prepare_data.py` — download/convert StudyChat data
- `scripts/1_label_data.py` — interactive annotation interface
- `scripts/1b_create_splits.py` — stratified train/val/test splitting

### Phase 4: Training & Analysis Scripts
- `scripts/2_train_classifier.py` — classifier fine-tuning
- `scripts/3_causal_analysis.py` — tutor→student pattern analysis
- `scripts/4_train_rl_policy.py` — RL policy training
- `scripts/5_evaluate.py` — classifier metrics + baseline vs adaptive comparison
- `scripts/demo.py` — interactive demo
- `scripts/run_all.py` — orchestration script

### Phase 5: Tests
- `tests/test_engagement.py`
- `tests/test_classifier.py`
- `tests/test_policy.py`
- `tests/test_state_encoder.py`
- `tests/test_utils.py`
- `tests/test_integration.py`

### Phase 6: Documentation & Reporting
- `README.md`
- `REPORT.md`
- `IMPLEMENTATION_STATUS.md`

## Key Technical Decisions

### 1. StudyChat Dataset
The spec references a "StudyChat" dataset but provides no verified URL. Research indicates this is not a standard public dataset with a known repository. **Decision**: Implement a synthetic data generator that creates realistic student-tutor dialogues for pipeline testing, clearly labeled as synthetic. Provide instructions for users to supply their own real data.

### 2. DeBERTa Embedding for State Encoder
DeBERTa-base produces 768-dimensional [CLS] token embeddings. The state vector is: `[768 embedding] + [extraction_score] + [engagement_score] + [turn_number_normalized] = 771 dims`. This matches the spec exactly.

### 3. Offline RL Training
The spec's RL training loop assumes we can observe what happens after an intervention, but in historical data, we don't control which intervention was applied. **Decision**: 
- Implement a heuristic tutor-response categorizer to label historical tutor responses as one of the 4 action types
- Use behavioral cloning (supervised) as a warm-start  
- Apply offline Q-learning with importance-aware updates
- Clearly document that observed rewards cannot be attributed causally to actions the policy would have chosen

### 4. LLM Integration
- Primary: OpenAI API (GPT-4/GPT-4o)
- Fallback: Deterministic template-based responses for development/testing
- API key loaded from `.env` file, never hardcoded

### 5. Checkpoint Format
For the classifier, we save using HuggingFace `save_pretrained()` into a directory. For the policy network, standard PyTorch `.pth` state dict files.

## Key Technical Risks

| Risk | Mitigation |
|------|-----------|
| No real StudyChat dataset available | Synthetic data generator + clear documentation |
| Small labeled dataset (500 examples) overfitting | Use class weighting, early stopping, dropout; honestly report limitations |
| CPU-only training is slow for DeBERTa | Reduce batch size, use gradient accumulation, limit epochs |
| Offline RL reward attribution is invalid | Document limitation, use conservative offline strategy |
| No LLM API key available | Template fallback that's clearly labeled |

## Acceptance Criteria

1. All source modules importable and functional
2. Data pipeline creates valid splits from synthetic or real data
3. Classifier trains, saves, loads, and predicts with correct output shape
4. State encoder produces exactly 771-dim vectors
5. Policy network takes 771-dim input, outputs 4 Q-values
6. Adaptive tutor integrates all components end-to-end
7. All automated tests pass
8. Demo script runs interactively
9. No fabricated metrics or hidden synthetic substitutions
