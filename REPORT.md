# Technical Report: Reinforcement Learning-Driven Adaptive Tutoring System for Cognitive Offloading Mitigation

---

## Abstract

This technical report details the complete design, engineering, and evaluation of an adaptive AI tutoring system developed to detect and mitigate **cognitive offloading** in students learning Data Structures and Algorithms (DSA). The architecture pairs a fine-tuned **DeBERTa-v3** transformer model for real-time extraction intent detection with a multi-signal reasoning engagement scorer and a PyTorch **Deep Q-Network (DQN)** policy that selects from four pedagogical intervention strategies (`REDIRECT`, `SCAFFOLD`, `SOCRATIC`, `DELAY`). The system incorporates neuro-symbolic safety guardrails, persistent SQLite memory tracking for concept mastery, teacher-specified spaced retention check-ins, and automated mid-chat topic transitions. Trained on dialogue sequences from the **StudyChat dataset**, the system achieves a 94.2% F1-score in offloading detection and significantly curtails dependency on code dumps while fostering active algorithmic reasoning.

---

## 1. Introduction & Problem Statement

### 1.1 Cognitive Offloading in Programming Education
Generative AI tools provide instant, syntactic code generation. While highly efficient for production workflows, in computer science education this immediacy facilitates **cognitive offloading**—where students outsource difficult analytical thinking to the AI rather than constructing mental algorithmic models. Habitual offloading impairs deep learning, creates an illusion of competence, and degrades retention on foundational concepts (e.g., node manipulation in linked lists, tree balancing, dynamic programming memoization).

### 1.2 System Objectives
1. **Accurate Detection**: Real-time identification of student prompts intended to extract answers without reasoning.
2. **Pedagogical Action Selection**: Using Deep Reinforcement Learning to choose the optimal pedagogical strategy for each interaction.
3. **Adaptive Safeguards**: Deterministic safety guardrails to prevent infinite redirect loops, handle cognitive fatigue, and gracefully process non-pedagogical greetings.
4. **Long-Term Memory & Spaced Retention**: Tracking mastery across DSA topics in SQLite and administering spaced recall checks on struggled concepts.
5. **Full-Stack Deployment**: A real-time FastAPI backend coupled with a glassmorphic dashboard featuring live terminal telemetry and student topic analytics.

---

## 2. System Architecture & Pipeline

```
                               ┌──────────────────────────────────────────────┐
                               │          Incoming Student Message            │
                               └──────────────────────┬───────────────────────┘
                                                      │
                              ┌───────────────────────┴───────────────────────┐
                              ▼                                               ▼
              ┌───────────────────────────────┐               ┌───────────────────────────────┐
              │    DeBERTa-v3 NLU Engine      │               │   Heuristic Engagement        │
              │  - [CLS] Embedding (768-d)    │               │   Scorer (0.0 to 1.0)         │
              │  - P(Extraction) (0.0 to 1.0) │               │   - Reasoning / Inquiry       │
              └───────────────┬───────────────┘               └───────────────┬───────────────┘
                              │                                               │
                              └───────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                                      ┌───────────────────────────────┐
                                      │ State Encoder (771-dim Vector)│
                                      │ [Embedding + Scores + Turns]  │
                                      └───────────────┬───────────────┘
                                                      │
                                                      ▼
                                      ┌───────────────────────────────┐
                                      │   PyTorch Deep Q-Network      │
                                      │  Outputs 4 Action Q-Values:   │
                                      │  [RED, SCAF, SOC, DEL]        │
                                      └───────────────┬───────────────┘
                                                      │
                                                      ▼
                                      ┌───────────────────────────────┐
                                      │ Adaptive Pedagogical          │
                                      │ Guardrails & Safety Bounds    │
                                      │ - Pre-routing (Greetings)     │
                                      │ - Struggle / Fatigue Escalate │
                                      └───────────────┬───────────────┘
                                                      │
                                                      ▼
                                      ┌───────────────────────────────┐
                                      │ Prompt Construction Engine    │
                                      │ + Topic Context               │
                                      │ + Learning History Context    │
                                      └───────────────┬───────────────┘
                                                      │
                                                      ▼
                                      ┌───────────────────────────────┐
                                      │ LLM Response Generator        │
                                      │ (Local Ollama / Qwen2.5:7b)   │
                                      └───────────────┬───────────────┘
                                                      │
                                                      ▼
                                      ┌───────────────────────────────┐
                                      │ SQLite Persistent Memory      │
                                      │ & Real-Time Terminal Telemetry│
                                      └───────────────────────────────┘
```

The pipeline processes each student interaction sequentially:
1. **NLU Offloading Inference**: Fine-tuned DeBERTa-v3 generates an extraction score $P(\text{Extraction}) \in [0.0, 1.0]$ and a 768-dimensional contextual representation.
2. **Engagement Quantification**: A multi-signal heuristic evaluator analyzes reasoning markers, message length, context blocks, and clarifying questions ($E \in [0.0, 1.0]$).
3. **State Representation**: Concatenates features into a 771-dimensional state vector $\mathbf{s}_t$.
4. **Q-Value Computation**: The PyTorch DQN policy evaluates $\mathbf{s}_t$ and outputs expected cumulative rewards for all 4 pedagogical actions.
5. **Guardrail Mediation**: Rules intervene when safety thresholds (e.g., cognitive fatigue, persistent struggle) override policy recommendations.
6. **LLM Generation**: Prompt templates incorporate the selected intervention, DSA problem context, topic transition directives, and past mastery status.
7. **Telemetry & Persistence**: Saves turns, Q-values, and metrics to SQLite and streams ASCII telemetry to the server console.

---

## 3. Dataset: StudyChat Dialogue Corpus

### 3.1 Corpus Specification
The system is trained and evaluated on the **StudyChat dataset**, a curated corpus of multi-turn student-tutor dialogues in computer science and algorithmic problem-solving. It captures authentic problem-solving workflows, syntax errors, conceptual struggles, and attempts to solicit direct answers.

### 3.2 Annotation & Categorization
Exchanges in the StudyChat dataset are categorized by student intent:
- **Extraction Mode**: Seeking direct code solutions, complete answers, or syntax fixes without demonstrating reasoning (e.g., *"Just give me the code for Dijkstra"*, *"Write the answer"*).
- **Learning Mode**: Demonstrating active engagement, explaining partial logic, sharing debug traces, or posing conceptual inquiries (e.g., *"I tried setting two pointers at the ends, but how do I handle duplicates?"*).

### 3.3 Dialogue-Aware Stratification
To prevent information leakage across splits, conversations are partitioned into **60% Train**, **20% Validation**, and **20% Test** sets at the dialogue level (i.e., all turns from a given conversation belong strictly to one split).

---

## 4. Extraction Classifier (DeBERTa-v3)

### 4.1 Model Topology
The classification engine leverages **DeBERTa-v3-base** with disentangled attention:

$$\mathbf{h}_0 = \text{DeBERTa}_{\text{backbone}}(\text{tokens})_{[\text{CLS}]} \in \mathbb{R}^{768}$$
$$\mathbf{h}_1 = \text{Dropout}(0.1)(\text{ReLU}(\mathbf{W}_1 \mathbf{h}_0 + \mathbf{b}_1)), \quad \mathbf{W}_1 \in \mathbb{R}^{256 \times 768}$$
$$\mathbf{h}_2 = \text{Dropout}(0.1)(\text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)), \quad \mathbf{W}_2 \in \mathbb{R}^{64 \times 256}$$
$$\mathbf{z} = \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3, \quad \mathbf{W}_3 \in \mathbb{R}^{2 \times 64}$$
$$P(\text{Extraction}) = \text{Softmax}(\mathbf{z})_1$$

### 4.2 Fine-Tuning Optimization
- **Optimizer**: AdamW with differential learning rates:
  - Backbone: $2\text{e-}5$
  - Classification Head: $2\text{e-}4$
- **Loss**: Weighted Cross-Entropy Loss to balance class frequencies.
- **Regularization**: Weight decay of $0.01$, gradient clipping at max norm $1.0$.

---

## 5. Multi-Signal Engagement Scorer

The engagement evaluator computes a continuous score $E \in [0.0, 1.0]$ across four transparent dimensions:

$$E = S_{\text{reasoning}} + S_{\text{length}} + S_{\text{clarifying}} + S_{\text{context}}$$

| Signal Dimension | Max Weight | Heuristic Mechanism | Pedagogical Justification |
| :--- | :---: | :--- | :--- |
| **Reasoning Evidence** | **0.40** | Regular expressions for causal reasoning (*"because"*, *"I think"*, *"however"*, *"my assumption"*, etc.). | Strongest indicator of cognitive effort. |
| **Message Length** | **0.20** | Word count thresholding with diminishing returns ($S_{\text{length}} = \min(0.20, \frac{\text{words}}{50} \times 0.20)$). | Penalizes one-word answers; caps verbosity. |
| **Clarifying Questions** | **0.20** | Detection of conceptual interrogative syntax (*"why does"*, *"how come"*, *"what if"*). | Identifies active conceptual inquiry. |
| **Context Provided** | **0.20** | Detection of code snippets, tracebacks, or expected vs. actual outputs. | Verifies engagement with the coding environment. |

---

## 6. Reinforcement Learning Action Space & Policy Network

### 6.1 State Space Formulation ($\mathcal{S}$)
The state vector $\mathbf{s}_t \in \mathbb{R}^{771}$ represents the student's dynamic learning context:

$$\mathbf{s}_t = \Big[ \mathbf{e}_{\text{CLS}} \;\Vert\; P(\text{Extraction}) \;\Vert\; E \;\Vert\; \tau_{\text{norm}} \Big]$$

Where:
- $\mathbf{e}_{\text{CLS}} \in \mathbb{R}^{768}$: Contextual embedding from DeBERTa-v3.
- $P(\text{Extraction}) \in [0.0, 1.0]$: Offloading probability.
- $E \in [0.0, 1.0]$: Engagement score.
- $\tau_{\text{norm}} = \min(1.0, \frac{\text{struggle turns}}{10}) \in [0.0, 1.0]$: Normalized struggle duration.

### 6.2 Pedagogical Action Space ($\mathcal{A}$)
| Action | ID | Strategy |
| :--- | :---: | :--- |
| **REDIRECT** | 0 | Refuses direct code solutions; refocuses student on underlying logic and conceptual invariants. |
| **SCAFFOLD** | 1 | Decomposes complex logic into manageable sub-goals, offering targeted structural hints. |
| **SOCRATIC** | 2 | Asks targeted open-ended questions that lead students to identify bugs or deduce the next step. |
| **DELAY** | 3 | Advises a cognitive break or pauses the interaction to prevent burnout during sustained struggle. |

### 6.3 Deep Q-Network (DQN) Architecture
- **Layer Dimensions**: $771 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 4$ with ReLU activations.
- **Parameters**: 239,044 trainable weights (Xavier uniform initialization).
- **Experience Replay Buffer**: 10,000 transitions.
- **Target Network**: Synchronized every 10 episodes to guarantee training stability:
  $$\mathcal{L}(\theta) = \mathbb{E}_{(\mathbf{s}, a, r, \mathbf{s}') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(\mathbf{s}', a'; \theta^{-}) - Q(\mathbf{s}, a; \theta) \right)^2 \right]$$
- **Reward Function**: $R_t = P(\text{Extraction})_t - P(\text{Extraction})_{t+1} + \lambda \Delta E_t$, rewarding actions that reduce offloading while promoting active reasoning.
- **GPU Retraining**: Retrained on CUDA GPU with class-weighted loss to ensure balanced Q-value distributions across all actions.

---

## 7. Neuro-Symbolic Safety Guardrails & Anti-Loop Safeguards

1. **Conversational Pre-Routing**: Greetings (*"hi"*, *"hello"*) and gratitude (*"thanks"*) bypass the RL pipeline and receive natural conversational replies without incurring offloading penalties.
2. **Anti-Loop Escalation**: If a student repeatedly insists on getting code ($P(\text{Extraction}) > 0.70$) over $10+$ turns, repeating `REDIRECT` causes frustration. The escalation guardrail intervenes and shifts the action to `DELAY` (Action ID: 3), recommending a cognitive break.
3. **Cognitive Fatigue Detection**: Explicit burnout statements (*"my brain is fried"*, *"head hurts"*, *"exhausted"*) trigger immediate `DELAY` interventions.

---

## 8. Persistent Knowledge Engine & Teacher Requirements

### 8.1 SQLite Database Schema (`tutor_memory.db`)
- **`conversations`**: Stores unique sessions, auto-generated titles, and active DSA concept mappings.
- **`messages`**: Records conversation history, role, content, RL actions, Q-values, and student feedback.
- **`student_knowledge`**: Tracks student mastery status (`not_started`, `in_progress`, `struggling`, `mastered`), turns practiced, intervention counts, and confidence scores (1–10) across 11 core DSA concepts.

### 8.2 Spaced Retention & Recall Check-In (Teacher Requirement 3)
When a student starts a new session, `get_retention_checkin()` searches the knowledge table for concepts previously studied where the student struggled or reported low confidence ($< 7/10$). The UI presents an interactive **Spaced Practice Banner** with options to take a 1-minute recall quiz, rate confidence, or dismiss.

### 8.3 Mid-Chat Topic Transition Detection
- An expanded DSA keyword lexicon identifies when a student shifts topics mid-chat (e.g., from *AVL Trees* to *Reverse a Linked List*).
- When a transition is recognized:
  1. The conversation's topic tag in SQLite is updated, and the tutor's struggle counter is reset.
  2. The LLM prompt receives a mandatory directive to verbally acknowledge the transition and check on the student's satisfaction with the previous topic.
  3. The UI renders an **Interactive Topic Transition Feedback Card** in the chat stream, allowing the student to record their confidence rating for the completed topic.

---

## 9. Full-Stack Web Application & Telemetry

1. **FastAPI Server (`fastapi_app.py`)**: Asynchronous, high-throughput REST API supporting chat streaming, concept queries, and feedback logging with automatic reload (`StatReload`).
2. **Terminal Telemetry Console**: Formatted ASCII telemetry cards stream live to the server terminal showing message text, offloading probability, reasoning engagement, Q-value distributions, and chosen RL action.
3. **Interactive Concept Mastery Dashboard**: Clicking the bottom-left **Concepts Explored** card opens an analytics modal displaying overall syllabus mastery, problem-solving turns, cumulative study time, average confidence, engagement levels, topic filters, and a *"Practice with Tutor"* quick-start button.

---

## 10. Empirical Evaluation & Technical Validation

### 10.1 Extraction Classifier Performance (StudyChat Test Split)
- **Accuracy**: **94.2%**
- **Precision (Extraction)**: **92.8%**
- **Recall (Extraction)**: **95.6%**
- **F1-Score**: **94.2%**
- **AUC-ROC**: **0.981**

### 10.2 Reinforcement Learning Convergence & Policy Impact
- **Convergence**: Deep Q-Network training demonstrates stable loss minimization and steady positive reward growth across 400 episodes.
- **Action Selection Profile**:
  - `SOCRATIC` (42%): Primary response for conceptual inquiries.
  - `SCAFFOLD` (34%): Applied when partial understanding is evident.
  - `REDIRECT` (16%): Applied on overt solution extraction requests.
  - `DELAY` (8%): Applied during prolonged struggle or student fatigue.
- **Behavioral Impact**: Compared against a standard, non-adaptive direct-code baseline, the adaptive tutor achieves a **63% reduction in immediate code offloading attempts**, successfully redirecting students into active dialogue.

### 10.3 Execution-Time Measurements
Benchmarked on the deployed FastAPI server with warm production-loaded models:

| Measurement | Value |
| :--- | :---: |
| **Classifier inference time per message** | **~280 ms** |
| **DQN forward pass per turn** | **~0.06 ms** |
| **End-to-end response time including LLM generation** | **~13.9 s** |
| **DQN training time (400 episodes, CUDA GPU)** | **~2 min** |

*Classifier time (mean 279.67 ms, p95 289.66 ms) reflects full DeBERTa-v3 forward pass on CPU. DQN forward pass (mean 0.0642 ms, p95 0.0966 ms) measured over 5,000 iterations. End-to-end latency dominated by Qwen2.5-7b LLM generation.*

### 10.4 Comparative Multi-Transformer Benchmark Study
To validate DeBERTa against alternate transformer architectures on the StudyChat dataset:

| Transformer Backbone | Params | Accuracy | Macro F1 | Extraction Recall | Latency | Core Observations |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **DeBERTa-base (Ours)** | **86M** | **94.2%** | **94.2%** | **95.6%** | **~280 ms** | Disentangled attention captures nuanced offloading patterns. |
| **RoBERTa-base** | 125M | 91.8% | 91.6% | 92.4% | ~265 ms | Solid generalization; higher memory footprint. |
| **ELECTRA-base** | 110M | 90.5% | 90.3% | 91.1% | ~250 ms | Fast pre-training; slightly lower recall on conversational shortcuts. |
| **BERT-base-uncased** | 110M | 88.4% | 88.1% | 87.5% | ~240 ms | Standard baseline; struggles with short colloquial prompts. |
| **DistilBERT-base** | 66M | 86.2% | 85.9% | 84.8% | **~145 ms** | Lightweight & fastest; suitable for resource-constrained edge deployments. |

### 10.5 Active Cognitive Engagement Time (ACET)
To eliminate runaway timers from blurred tabs or idle sessions, the platform tracks **Active Cognitive Engagement Time (ACET)**:
- **HTML5 Page Visibility API**: Pauses active tracking when tab is hidden or minimized.
- **Activity Watchdog**: Disables accumulation after 60 seconds of student inactivity (no typing, clicks, or scrolling).
- **Periodic Focus Heartbeat**: Sends 25-second telemetry pulses to `/api/heartbeat`.
- **Bounded Dwell Time**: Clamps per-turn student reading/problem-solving time between 5s and 180s.

### 10.6 Adaptive Feedback-Driven Diagnostic & Challenge Question Engine
Student comprehension ratings dynamically trigger targeted cognitive interventions:
- **Tier 1 (Rating 1–4 / 'confused')**: 4-option diagnostic MCQ isolating foundational misconceptions without cognitive overload.
- **Tier 2 (Rating 5–7)**: Step-by-step logic and code tracing challenge.
- **Tier 3 (Rating 8–10 / 'clear')**: Advanced asymptotic bound & proof-of-correctness challenge.
- **Topic Transition Persistence**: Mid-chat topic transition cards remain visible and interactive until rated, with automatic 12-hour cooldowns on resolved check-ins.

---

## 11. Automated Test Suite Validation

The system is validated through an automated `pytest` test suite:
- `tests/test_classifier.py`: Validates model weights, inference latency, and extraction probability boundaries.
- `tests/test_engagement.py`: Tests regex accuracy, edge cases, and weight bounds for engagement signals.
- `tests/test_policy.py`: Verifies DQN forward passes, replay buffer memory transitions, and Q-value tensor formatting.
- `tests/test_state_encoder.py`: Tests 771-dimensional state vector construction and normalization.
- `tests/test_integration.py`: Validates end-to-end multi-turn chat sessions, topic transitions, guardrail overrides, and SQLite persistence.

**Current Test Status**: **162 passed tests** across all modules.
