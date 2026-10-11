# Technical Report: RL-Based Adaptive Tutoring System for Cognitive Offloading Detection

## Abstract

This report describes the design, implementation, and evaluation of an adaptive AI tutoring system that detects cognitive offloading behavior in students and applies reinforcement learning to select pedagogical interventions. The system combines a fine-tuned DeBERTa classifier for extraction/learning mode detection, a heuristic engagement scorer, and a Q-learning policy network that selects from four intervention strategies (REDIRECT, SCAFFOLD, SOCRATIC, DELAY). We document the system's architecture, training methodology, evaluation approach, and — critically — the limitations inherent in training on synthetic data and using offline RL from observational dialogues.

## 1. Introduction

### 1.1 Problem Statement

In AI-assisted tutoring, students may engage in *cognitive offloading* — seeking direct answers rather than working through problems to build understanding. While getting a quick answer may be efficient in the short term, it undermines deep learning. An adaptive tutor should detect this behavior and intervene appropriately, encouraging students back toward active learning.

### 1.2 Objectives

1. **Detection**: Classify student messages as extraction-oriented (seeking answers) or learning-oriented (seeking understanding)
2. **Adaptation**: Use reinforcement learning to select the most effective pedagogical intervention based on the student's current state
3. **Response Generation**: Generate pedagogically appropriate tutor responses via LLM with strategy-specific system prompts
4. **Transparency**: Clearly document all limitations, synthetic data usage, and non-causal claims

## 2. System Architecture

### 2.1 Pipeline Overview

The system processes each student message through a sequential pipeline:

1. **Extraction Classifier** (DeBERTa-base): Produces an extraction probability score (0.0 = learning, 1.0 = extraction)
2. **Engagement Scorer** (heuristic): Evaluates four signals — reasoning evidence (0–0.4), message length (0–0.2), clarifying questions (0–0.2), and provided context (0–0.2) — to produce an engagement score (0–1)
3. **State Encoder**: Constructs a 771-dimensional state vector: [768 DeBERTa CLS embedding] + [extraction_score] + [engagement_score] + [normalized_turn_number]
4. **Policy Network** (Q-learning): Maps the 771-dim state to 4 Q-values, one per intervention action
5. **Action Selection**: Epsilon-greedy during training, greedy during inference
6. **Prompt Generator**: Maps the selected action to a strategy-specific system prompt
7. **LLM Integration**: Generates a natural language response (OpenAI API or template fallback)

### 2.2 Classifier Architecture

```
DeBERTa-base → [CLS] pooling
    → Dense(768 → 256) → ReLU → Dropout(0.1)
    → Dense(256 → 64) → ReLU → Dropout(0.1)
    → Dense(64 → 2)
    → Softmax → P(extraction)
```

Training uses AdamW with differential learning rates (full rate for the classification head, 0.1× for the DeBERTa backbone), gradient clipping at 1.0, and automatic class weighting for imbalanced data.

### 2.3 Policy Network Architecture

```
Linear(771 → 256) → ReLU
    → Linear(256 → 128) → ReLU
    → Linear(128 → 64) → ReLU
    → Linear(64 → 4)
    → Raw Q-values
```

Total parameters: ~239K. Xavier uniform initialization. Adam optimizer with learning rate 1e-4.

### 2.4 Q-Learning Training

The RL training uses:
- **Experience replay** (buffer capacity: 10,000) for decorrelating updates
- **Target network** (updated every 10 episodes) for stability
- **Epsilon-greedy** exploration (ε = 0.2, decay = 0.995, min = 0.05)
- **Discount factor** γ = 0.99
- **Reward**: `extraction_before - extraction_after` (positive if student shifts toward learning)

## 3. Engagement Scoring

The engagement scorer is a transparent, rule-based system combining four heuristic signals:

| Signal | Weight | Indicators |
|--------|--------|------------|
| Reasoning evidence | 0.0–0.4 | "I tried", "because", "I think", "however", etc. |
| Message length | 0.0–0.2 | Word count thresholds with diminishing returns |
| Clarifying questions | 0.0–0.2 | "why does", "how can I", "what if", etc. |
| Context provided | 0.0–0.2 | Code blocks, error messages, expected output, etc. |

**Design rationale**: Reasoning evidence receives the highest weight (0.4) because it is the strongest indicator of genuine engagement. Length alone (capped at 0.2) cannot produce a high score — a student must show reasoning, questions, or context for meaningful engagement.

## 4. Data

### 4.1 Synthetic Data Generation

The system generates synthetic dialogue data for pipeline testing:
- **Extraction messages** (30 templates): Short, direct requests ("Just give me the code", "What's the answer?")
- **Learning messages** (30 templates): Longer messages with reasoning, code, errors, and conceptual questions
- **Tutor responses** (4 categories × 3–4 templates): redirect, scaffold, socratic, direct

**WARNING**: Synthetic data is generated from templates, not collected from real students. Any metrics reported on synthetic data reflect the system's ability to distinguish template categories, not real-world classification performance.

### 4.2 Dataset Splits

The pipeline generates 500 labeled examples with a 300/100/100 train/val/test split using dialogue-aware stratified splitting to prevent data leakage.

## 5. Evaluation

### 5.1 Classifier Evaluation

The classifier is evaluated on precision, recall, F1, and accuracy. On synthetic data, these metrics are expected to be high because the template-based messages are easily separable. **These metrics do not transfer to real student data.**

### 5.2 Adaptive vs. Baseline Comparison

The evaluation compares:
- **Baseline**: Non-adaptive tutor that responds directly to all queries
- **Adaptive**: RL-policy-guided tutor that selects pedagogical interventions

Comparison metrics include action diversity, response characteristics, and extraction score patterns. Without real interactive data, this comparison is qualitative.

## 6. Critical Limitations

### 6.1 Synthetic Data

All training and evaluation uses synthetic data. The classifier's ability to distinguish template-based extraction messages from template-based learning messages does not predict performance on real student messages, which are far more nuanced and varied.

### 6.2 Offline RL and Causal Attribution

The RL policy is trained on observed dialogue transitions. The reward (extraction score change) is associated with the action the tutor ACTUALLY took, not a hypothetical action. We CANNOT claim:
- "Action X caused the student to shift toward learning"
- "The RL policy selects better actions than the baseline"

We CAN claim:
- "In observed data, action X was associated with subsequent extraction score changes of Y"
- "The RL policy learns to predict Q-values for actions based on observed state-reward associations"

This is a fundamental limitation of offline/batch RL from observational data without controlled experiments.

### 6.3 Small Dataset

500 labeled examples is insufficient for robust DeBERTa fine-tuning. Overfitting is mitigated through class weighting, dropout, and early stopping, but generalization is inherently limited.

### 6.4 CPU Training

All training runs on CPU, limiting the number of epochs and batch sizes feasible for DeBERTa fine-tuning.

### 6.5 LLM Dependency

Full response generation requires an OpenAI API key. Without it, template-based fallback responses are used, which are clearly labeled as non-LLM-generated.

## 7. Reproducibility

- All random seeds are set via `src/utils.set_seed()` (default: 42)
- Synthetic data generation is deterministic given the same seed
- Model architectures and hyperparameters are documented in code and config
- The complete pipeline can be reproduced via `python scripts/run_all.py`

## 8. Future Work

1. **Real data collection**: Partner with educational institutions to collect real student-tutor dialogues with IRB approval
2. **Online RL**: Deploy the system interactively to collect on-policy data and enable causal reward attribution
3. **Human evaluation**: Conduct studies comparing student learning outcomes with baseline vs. adaptive tutoring
4. **Multi-turn context**: Incorporate conversation history into the state representation
5. **GPU training**: Scale to larger models and datasets with GPU resources

## 9. Conclusion

This system demonstrates a complete, end-to-end pipeline for adaptive tutoring using cognitive offloading detection and reinforcement learning. While the system is functional and testable, its practical effectiveness remains unvalidated due to the use of synthetic data and offline RL. The codebase is designed to be extended with real data and online training when those become available.
