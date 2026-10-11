# COMPLETE PROJECT SPECIFICATION
# Detecting and Mitigating Cognitive Offloading: An RL-Based Adaptive Tutoring System

**Project Status:** Capstone, 14 weeks  
**Last Updated:** October 2026  
**Target Build Platform:** Python 3.10+

---

## TABLE OF CONTENTS

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Technical Specifications](#technical-specifications)
5. [Data Requirements](#data-requirements)
6. [Implementation Timeline](#implementation-timeline)
7. [Code Structure](#code-structure)
8. [Model Architectures](#model-architectures)
9. [Training Procedures](#training-procedures)
10. [Evaluation Methodology](#evaluation-methodology)
11. [Deliverables](#deliverables)
12. [Dependencies](#dependencies)
13. [Key Metrics](#key-metrics)
14. [Running Instructions](#running-instructions)

---

## PROJECT OVERVIEW

### What We're Building

An AI tutoring system that:
1. **Detects** when students are asking for answers (extraction-mode) vs. trying to learn (learning-mode)
2. **Uses RL** to learn which pedagogical intervention works best for each student query
3. **Adapts** its responses in real-time based on what the RL policy learned
4. **Evaluates** using both classifier accuracy and side-by-side response comparison

### Key Innovation

Instead of static "if-else" rules, we train a **policy network via reinforcement learning** that learns which intervention (REDIRECT, SCAFFOLD, SOCRATIC, DELAY) maximizes the reward (student shifts from extraction to learning-mode).

### Final Deliverables

- Trained classifier (extraction-mode detector)
- Trained RL policy network (intervention selector)
- Evaluation report with metrics
- Side-by-side comparison of baseline vs. adaptive tutor on 20 test queries
- Demo script for live testing

---

## PROBLEM STATEMENT

### The Gap

Students often ask for direct answers instead of trying to learn. This is called "cognitive offloading."

**Current state:** Tutors give the answer, enabling offloading.

**Unsolved problem:** No system detects offloading in real-time and adapts its response.

**Research gap:** Papers flag this as unsolved. Everyone says "this is a problem," but nobody built a technical solution.

### Why It Matters

- Offloading reduces learning outcomes
- Students become dependent on AI
- No one has automated detection + intervention

### Our Solution

Train an RL agent to learn which intervention works best for which student query type.

---

## SOLUTION ARCHITECTURE

### High-Level Flow (Inference)

```
Student Message
    ↓
[CLASSIFIER] → Extract/Learning Score (0-1)
    ↓
[STATE ENCODER] → 771-dim vector (embedding + features)
    ↓
[RL POLICY NETWORK] → Q-values for 4 actions
    ↓
[ACTION SELECTOR] → argmax(Q-values) → best action
    ↓
[SYSTEM PROMPT GENERATOR] → instruction for LLM
    ↓
[LLM CALL] → GPT-4 / Claude with system prompt
    ↓
Response to Student
```

### High-Level Flow (Training)

```
Student Message
    ↓
[CLASSIFIER + RL POLICY] → choose action
    ↓
[GENERATE RESPONSE] → tutor responds
    ↓
[WAIT FOR NEXT MESSAGE] → student replies
    ↓
[CALCULATE REWARD] → did extraction score decrease?
    ↓
[Q-LEARNING UPDATE] → improve policy network
    ↓
Repeat on all 450+ StudyChat dialogues
```

### Components

| Component | Purpose | Input | Output |
|---|---|---|---|
| **Classifier** | Detect if student is in extraction or learning mode | Student message | Extraction score (0-1) |
| **Engagement Scorer** | Measure student engagement from message patterns | Student message | Engagement score (0-1) |
| **State Encoder** | Convert message + features to RL state vector | Message + scores | 771-dim vector |
| **Policy Network** | RL agent that learns best action for each state | State vector | 4 Q-values |
| **Action Selector** | Choose which action to take | Q-values | Action ID (0-3) |
| **System Prompt Generator** | Create LLM system prompt based on action | Action ID | System prompt text |
| **LLM** | Generate tutor response | System prompt + student message | Tutor response |

---

## TECHNICAL SPECIFICATIONS

### The 4 Actions

```
0: REDIRECT
   Purpose: Stop lazy requests for answers
   Prompt: "Tell student no, redirect to thinking"
   When: High extraction, low engagement
   
1: SCAFFOLD  
   Purpose: Provide hints without solving
   Prompt: "Give worked example + let them finish"
   When: Mid extraction, showing some effort
   
2: SOCRATIC
   Purpose: Ask questions, force thinking
   Prompt: "Ask clarifying questions, no answers"
   When: Low extraction, capable of thinking
   
3: DELAY
   Purpose: Suggest break when exhausted
   Prompt: "Recommend rest, not pushing harder"
   When: High extraction + high frustration
```

### State Representation

```python
state = {
    "message_embedding": [768-dim BERT embedding],
    "extraction_score": float (0-1),
    "engagement_score": float (0-1),
    "turn_number": int,
    "conversation_history": list of prior messages (optional)
}

# Combined state vector: 768 + 3 = 771 dimensions
```

### Reward Function

```python
reward = extraction_score_before - extraction_score_after

# If student shifts from high extraction to low extraction: positive reward
# If student gets lazier: negative reward
# Range: -1.0 to +1.0

# Quality tiers:
# > +0.50: Excellent (major shift to learning)
# +0.20 to +0.50: Good (some improvement)
# -0.05 to +0.20: Neutral (no change)
# < -0.05: Bad (student got worse)
```

### Hyperparameters

```python
# Classifier training
CLASSIFIER_MODEL = "microsoft/deberta-base"
CLASSIFIER_EPOCHS = 3
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LEARNING_RATE = 2e-5

# RL Policy Network
POLICY_HIDDEN_DIMS = [256, 128, 64]
POLICY_LEARNING_RATE = 1e-4
POLICY_GAMMA = 0.99  # Discount factor
POLICY_EPSILON = 0.2  # Exploration rate (20% random actions)
POLICY_ALPHA = 0.01  # Learning rate for Q-updates

# Training
RL_EPISODES = 400  # Number of dialogues to train on
RL_MAX_STEPS = 20  # Max turns per dialogue
```

---

## DATA REQUIREMENTS

### Data Source 1: StudyChat Dataset

**What:** Real student-LLM dialogue logs from university AI course  
**Size:** ~15,000 turns across ~450 dialogues  
**Duration:** 15 weeks of actual course  
**Source:** Public (GitHub or Hugging Face)  
**Usage:**
- 500 turns hand-labeled → split 60/20/20 (train/val/test classifier)
- 450+ dialogues used for RL training

**Format:**
```csv
dialogue_id,turn_num,speaker,message,timestamp,context
SC_001,1,student,"How do I start?",2024-01-15,"Week 3 sorting"
SC_001,2,tutor,"What have you tried?",2024-01-15,
SC_001,3,student,"Nothing, just give code",2024-01-15,
```

### Data Source 2: Hand-Labeled Extraction/Learning

**What:** 500 student turns labeled by you  
**Labels:** 
- 0 = Learning-mode (trying to understand)
- 1 = Extraction-mode (trying to get answer)

**Format:**
```csv
dialogue_id,turn_num,message,label,confidence,notes
SC_001,3,"Just give code",1,high,"Direct request"
SC_005,2,"I tried X but got Y",0,high,"Shows reasoning"
```

**Labeling Process:**
1. Sample 500 random turns from StudyChat
2. For each, decide: extraction or learning?
3. Use rubric (see Labeling Rubric section below)
4. Track confidence (high/medium/low)

### Data Source 3: Optional Bootstrap

**What:** MathDial dataset (annotated math tutoring)  
**Size:** ~3,000 turns (optional)  
**Usage:** Additional training signal if available  
**Source:** Public (GitHub)

### Labeling Rubric

```
EXTRACTION-ORIENTED (Student prioritizes getting answer):
- Asks directly for answer/code/solution
- Provides minimal context or reasoning
- Shows low effort ("just give me", "tell me")
- Responds to scaffolding by asking for more direct help
Examples: "What's the code?", "Just tell me"
Label: 1

LEARNING-ORIENTED (Student seeks understanding):
- Explains what they've tried
- Asks clarification questions about concepts
- Responds to scaffolding by engaging with hints
- Shows reasoning, even if incomplete
Examples: "I tried X but...", "Why does...?"
Label: 0

When unsure:
- Look at previous turns (pattern?)
- Check if they provide any reasoning
- Default to dominant mode in that exchange
```

---

## IMPLEMENTATION TIMELINE

### Week 1-3: Data & Labeling

**Tasks:**
1. Download StudyChat dataset
2. Clean/format data into standardized CSV
3. Sample 500 turns randomly (stratified by dialogue)
4. Hand-label 500 turns using rubric
5. Measure inter-rater agreement (if co-labeling)
6. Split into 300/100/100 (train/val/test)

**Deliverable:** `labeled_500.csv` with splits ready

### Week 4-6: Classifier Training

**Tasks:**
1. Load training set (300 turns)
2. Tokenize with BERT
3. Fine-tune DeBERTa on extraction/learning classification
4. Validate on val set (100 turns)
5. Test on test set (100 turns)
6. Generate performance metrics
7. Save trained model

**Deliverable:** 
- `classifier_v1.pth` (trained weights)
- `classifier_report.txt` (accuracy, F1, confusion matrix)

### Week 6-9: Causal Analysis

**Tasks:**
1. Load full StudyChat dialogues
2. For each dialogue, reconstruct tutor→student sequences
3. Classify each student message (extraction or learning?)
4. Analyze: does tutor response type (leak vs. scaffold) predict next student mode?
5. Calculate effect sizes
6. Generate plots

**Deliverable:**
- `causal_analysis.csv` (tutor action → next student mode)
- `causal_plots.png` (bar charts)

### Week 9-12: RL Policy Training & Evaluation

**Tasks:**
1. Define RL environment (state, action, reward)
2. Implement PolicyNetwork class
3. Implement Q-learning training loop
4. Train on 400+ StudyChat dialogues
5. Monitor learning curves
6. Select final trained policy
7. Run evaluation on 20 test queries (baseline vs. adaptive)
8. Generate side-by-side comparisons

**Deliverable:**
- `policy_network_v1.pth` (trained RL policy)
- `evaluation_report.txt` (metrics + examples)
- `comparison_examples.md` (side-by-side responses)

### Week 12-14: Write-up & Presentation

**Tasks:**
1. Summarize methods
2. Report results (classifier accuracy, RL performance, evaluation)
3. Create presentation slides
4. Prepare demo script
5. Write limitations section

**Deliverable:**
- `REPORT.md` (full technical report)
- `PRESENTATION.pptx`
- `demo.py` (script to run live)

---

## CODE STRUCTURE

### Directory Layout

```
rl_tutoring_system/
├── data/
│   ├── studychat_full.csv           # Raw StudyChat data
│   ├── labeled_500.csv              # Hand-labeled extraction/learning
│   ├── train_300.csv                # Training split
│   ├── val_100.csv                  # Validation split
│   └── test_100.csv                 # Test split
│
├── models/
│   ├── classifier_v1.pth            # Trained classifier
│   ├── policy_network_v1.pth        # Trained RL policy
│   └── config.json                  # Model hyperparameters
│
├── src/
│   ├── __init__.py
│   ├── classifier.py                # DeBERTa classifier
│   ├── engagement_scorer.py         # Engagement scoring
│   ├── rl_policy_network.py         # Policy network + RL trainer
│   ├── adaptive_tutor.py            # Main inference loop
│   ├── utils.py                     # Helper functions
│   └── prompts.py                   # System prompts for 4 actions
│
├── scripts/
│   ├── 1_label_data.py              # Hand-label 500 turns (interactive)
│   ├── 2_train_classifier.py        # Train DeBERTa classifier
│   ├── 3_causal_analysis.py         # Analyze tutor→student patterns
│   ├── 4_train_rl_policy.py         # Train RL policy
│   ├── 5_evaluate.py                # Run evaluation (20 test queries)
│   └── demo.py                      # Interactive demo
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_classifier_analysis.ipynb
│   └── 03_rl_results.ipynb
│
├── results/
│   ├── classifier_metrics.json
│   ├── causal_analysis.csv
│   ├── training_curves.png
│   ├── evaluation_results.json
│   └── example_comparisons.md
│
├── requirements.txt
├── README.md
├── PROJECT_SPEC_COMPLETE.md         # This file
└── .gitignore
```

### File Descriptions

#### `src/classifier.py`

```python
"""
Fine-tuned DeBERTa classifier for extraction vs. learning mode detection

Classes:
- ExtractionClassifier: Wraps fine-tuned model
  ├── predict(message: str) → float (0-1 extraction score)
  ├── predict_batch(messages: List[str]) → List[float]
  └── train(train_dataset, val_dataset)

Usage:
  classifier = ExtractionClassifier()
  classifier.load("models/classifier_v1.pth")
  score = classifier.predict("Just give me the code")  # 0.94
"""
```

#### `src/engagement_scorer.py`

```python
"""
Calculate engagement level from student message (0-1)

Functions:
- score_reasoning(message) → float (0.0-0.4)
  Checks if student explained their attempt
  
- score_turn_length(message) → float (0.0-0.2)
  Longer messages = more effort
  
- score_questions(message) → float (0.0-0.2)
  Clarifying questions indicate thinking
  
- score_context(message) → float (0.0-0.2)
  Shows code, errors, expected output
  
- calculate_engagement(message) → float (0.0-1.0)
  Sum of all 4 signals

Usage:
  engagement = calculate_engagement("I tried X but got Y")  # 0.8
"""
```

#### `src/rl_policy_network.py`

```python
"""
RL Policy Network and Q-learning trainer

Classes:
- PolicyNetwork(nn.Module):
  Input: 771-dim state vector
  Hidden: Dense(256) → ReLU → Dense(128) → ReLU → Dense(64) → ReLU
  Output: 4 Q-values (one per action)
  
- RLTrainer:
  ├── select_action(state, epsilon) → action
  ├── train_step(state, action, reward, next_state, done) → loss
  └── train_episode(dialogue) → rewards

Usage:
  trainer = RLTrainer()
  for episode in range(400):
      episode_reward = trainer.train_episode(dialogues[episode])
      learning_curves.append(episode_reward)
  trainer.save("models/policy_network_v1.pth")
"""
```

#### `src/adaptive_tutor.py`

```python
"""
Main inference engine: student message → tutor response

Classes:
- AdaptiveTutor:
  ├── load_models(classifier_path, policy_path)
  ├── infer(student_message) → dict {
  │     "extraction_score": float,
  │     "engagement": float,
  │     "action_taken": str,
  │     "q_values": List[float],
  │     "tutor_response": str
  │   }
  └── log_interaction(state, action, response)

Usage:
  tutor = AdaptiveTutor()
  tutor.load_models("models/classifier_v1.pth", "models/policy_network_v1.pth")
  
  result = tutor.infer("I'm stuck, give me the code")
  print(result["tutor_response"])
  # "I see you want the answer directly. Let's think instead..."
"""
```

#### `src/utils.py`

```python
"""
Helper functions

Functions:
- load_csv(path) → DataFrame
- tokenize_message(message) → tokens
- encode_state(message, extraction_score, engagement) → tensor
- calculate_reward(score_before, score_after) → float
- create_system_prompt(action_id) → str
- call_llm(system_prompt, user_message) → str
- save_model(model, path)
- load_model(path) → model
"""
```

#### `src/prompts.py`

```python
"""
System prompts for the 4 actions

Constants:
- PROMPT_REDIRECT
- PROMPT_SCAFFOLD
- PROMPT_SOCRATIC
- PROMPT_DELAY

Function:
- get_system_prompt(action_id: int) → str
"""
```

---

## MODEL ARCHITECTURES

### Classifier Architecture

```
Input: Student message (text)
  ↓
[BERT Tokenizer]
  ↓
[BERT Embedding Layer] → 768-dim vector
  ↓
[Classification Head]
  ├─ Dense(768 → 256)
  ├─ ReLU
  ├─ Dropout(0.1)
  ├─ Dense(256 → 64)
  ├─ ReLU
  ├─ Dropout(0.1)
  └─ Dense(64 → 2)
  ↓
[Softmax] → [P(learning), P(extraction)]
  ↓
Output: extraction_score = P(extraction)
```

**Model Details:**
- Base: microsoft/deberta-base (pre-trained)
- Fine-tune: Last 2 layers
- Dropout: 0.1 for regularization
- Loss: CrossEntropyLoss
- Optimizer: AdamW (lr=2e-5)
- Batch size: 32
- Epochs: 3 (on 300 training examples)

### Policy Network Architecture

```
Input: State vector (771-dim)
  = [message_embedding (768) + extraction_score (1) + engagement (1) + turn_num (1)]
  ↓
[Linear(771 → 256)]
  ↓
[ReLU]
  ↓
[Linear(256 → 128)]
  ↓
[ReLU]
  ↓
[Linear(128 → 64)]
  ↓
[ReLU]
  ↓
[Linear(64 → 4)]  # One output per action
  ↓
Output: Q-values [Q_redirect, Q_scaffold, Q_socratic, Q_delay]
```

**Model Details:**
- Input: 771
- Hidden: [256, 128, 64]
- Output: 4 (one per action)
- Activation: ReLU
- No softmax (raw Q-values)
- Parameters: ~250k

---

## TRAINING PROCEDURES

### Classifier Training

```python
# 1. Load labeled data
train_df = pd.read_csv("data/train_300.csv")
val_df = pd.read_csv("data/val_100.csv")

# 2. Create datasets
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
train_dataset = ClassificationDataset(train_df, tokenizer)
val_dataset = ClassificationDataset(val_df, tokenizer)

# 3. Initialize model
model = ExtractionClassifier()

# 4. Train
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="models/",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        save_strategy="epoch",
        evaluation_strategy="epoch"
    ),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()

# 5. Evaluate on test set
test_df = pd.read_csv("data/test_100.csv")
test_dataset = ClassificationDataset(test_df, tokenizer)
metrics = trainer.evaluate(test_dataset)
print(f"Accuracy: {metrics['eval_accuracy']:.3f}")
print(f"F1: {metrics['eval_f1']:.3f}")

# 6. Save
model.save_pretrained("models/classifier_v1.pth")
```

### RL Policy Training

```python
# 1. Initialize
classifier = ExtractionClassifier()
classifier.load("models/classifier_v1.pth")

policy_net = PolicyNetwork()
trainer = RLTrainer(policy_net, classifier)

# 2. Load dialogue data
dialogues = load_studychat_dialogues("data/studychat_full.csv")
random.shuffle(dialogues)

# 3. Training loop
episode_rewards = []
for episode in range(400):
    dialogue = dialogues[episode % len(dialogues)]
    
    total_reward = 0
    state = None
    
    for turn_idx in range(0, len(dialogue)-2, 2):  # pairs of (student, tutor)
        student_msg = dialogue[turn_idx]
        
        # Encode state
        extraction_before = classifier.predict(student_msg)
        engagement = calculate_engagement(student_msg)
        state = encode_state(student_msg, extraction_before, engagement)
        
        # Select action (epsilon-greedy)
        action = trainer.select_action(state, epsilon=0.2)
        
        # Get next student message (from real dialogue)
        next_student_msg = dialogue[turn_idx + 2]  # After tutor response
        
        # Calculate reward
        extraction_after = classifier.predict(next_student_msg)
        reward = extraction_before - extraction_after
        
        # Encode next state
        next_state = encode_state(next_student_msg, extraction_after, ...)
        done = (turn_idx + 2 >= len(dialogue) - 1)
        
        # Update
        loss = trainer.train_step(state, action, reward, next_state, done)
        total_reward += reward
    
    episode_rewards.append(total_reward)
    
    if episode % 50 == 0:
        print(f"Episode {episode}: avg reward {np.mean(episode_rewards[-50:]):.3f}")

# 4. Save
trainer.save("models/policy_network_v1.pth")

# 5. Plot learning curve
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Episode Reward")
plt.savefig("results/learning_curve.png")
```

---

## EVALUATION METHODOLOGY

### Evaluation 1: Classifier Metrics

**What:** Measure classifier accuracy on 100 held-out test examples

**Metrics:**
```
Accuracy:   % correct classifications
Precision:  When we say "extraction," are we right?
Recall:     Do we catch actual extraction-seeking?
F1:         Harmonic mean of precision & recall
Confusion Matrix: TP, TN, FP, FN breakdown
```

**Report:**
```
CLASSIFIER EVALUATION
═════════════════════════════════════════════════════════════

Test Set: 100 student turns (held-out)

Results:
├─ Accuracy:  84%
├─ Precision: 0.82 (when we say extraction, we're right 82% of the time)
├─ Recall:    0.79 (we catch 79% of actual extraction-seeking)
├─ F1:        0.80
│
├─ Confusion Matrix:
│    Predicted
│    Extr  Learn
│ A  79     21
│ c  12     88
│ t
│
└─ Best features for classification:
   1. Direct answer request keywords (0.42 importance)
   2. Turn length < 30 words (0.31 importance)
   3. Zero reasoning shown (0.22 importance)
```

### Evaluation 2: Policy Evaluation

**What:** Compare baseline tutor vs. adaptive tutor on 20 extraction-mode test queries

**Process:**
1. Select 20 queries from test set where classifier predicts extraction-mode (score > 0.7)
2. For each query:
   - Generate baseline response (no policy, standard prompt)
   - Generate adaptive response (with RL policy)
   - Compare side-by-side
3. Score both responses using rubric
4. Calculate win/tie/loss statistics

**Rubric (1-5 scale):**
```
5 = Excellent
  - Detects extraction-mode correctly
  - Responds with appropriate intervention
  - Pedagogically sound
  
4 = Good
  - Addresses help-seeking
  - Minor pedagogical issue
  
3 = Okay
  - Attempts to help
  - Unclear if addresses offloading
  
2 = Weak
  - Gives answer anyway
  - Dismissive
  
1 = Bad
  - Enables offloading
  - Counterproductive
```

**Report:**
```
POLICY EVALUATION
═════════════════════════════════════════════════════════════

Test Set: 20 extraction-mode queries

Results by category:
├─ Policy superior:    14 (70%)
├─ Tie:                4 (20%)
└─ Baseline superior:  2 (10%)

Average scores:
├─ Baseline:  2.3/5 (often leaks answer)
└─ Adaptive:  3.8/5 (redirects appropriately)

Statistical test:
├─ t-test: t=3.24, p=0.004 **
└─ Conclusion: Adaptive is significantly better

Examples: [Show 5 side-by-side comparisons with scores]
```

### Example Comparison

```
EXAMPLE 1:
═════════════════════════════════════════════════════════════

Query: "I'm stuck on this algorithm. Just tell me how to solve it."
Extraction score: 0.94 (HIGH)
Engagement: 0.2 (LOW)

BASELINE RESPONSE:
"Here's a solution approach:
1. Sort the array
2. Use two pointers
3. Return the result

You can implement this with a for loop..."

Score: 1/5 (leaks entire solution)

ADAPTIVE RESPONSE (with RL policy):
"I see you're asking for the solution directly. I understand 
it's frustrating, but getting the answer won't help you learn.

Let's break it down instead:
- What does 'solving it' mean to you?
- What have you tried so far?
- Can you describe the first step?"

Score: 4/5 (redirects effectively)

Decision: Policy SUPERIOR ✓
```

---

## DELIVERABLES

### By End of Week 14

**Code:**
- [ ] `src/classifier.py` (trained, functional)
- [ ] `src/engagement_scorer.py` (working)
- [ ] `src/rl_policy_network.py` (trained)
- [ ] `src/adaptive_tutor.py` (inference working)
- [ ] `scripts/1_label_data.py` (for labeling)
- [ ] `scripts/2_train_classifier.py` (runnable)
- [ ] `scripts/4_train_rl_policy.py` (runnable)
- [ ] `scripts/5_evaluate.py` (generates evaluation report)
- [ ] `scripts/demo.py` (interactive demo)

**Data:**
- [ ] `data/labeled_500.csv` (hand-labeled)
- [ ] `data/train_300.csv`, `val_100.csv`, `test_100.csv` (splits)

**Models:**
- [ ] `models/classifier_v1.pth` (trained weights)
- [ ] `models/policy_network_v1.pth` (trained weights)

**Results:**
- [ ] `results/classifier_metrics.json` (accuracy, F1, etc.)
- [ ] `results/evaluation_results.json` (policy win/tie/loss)
- [ ] `results/example_comparisons.md` (5-10 side-by-sides)
- [ ] `results/learning_curve.png` (RL training progress)

**Documentation:**
- [ ] `README.md` (how to run everything)
- [ ] `REPORT.md` (full technical report)
- [ ] `PRESENTATION.pptx` (slides for professor)

**Demo:**
- [ ] Interactive script that lets you:
  1. Type a student message
  2. See classifier output
  3. See policy network Q-values
  4. See tutor response
  5. Repeat

---

## DEPENDENCIES

```
# Core ML/NLP
torch==2.1.0
transformers==4.35.0
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.0.3

# Utilities
matplotlib==3.8.0
seaborn==0.13.0
tqdm==4.66.1

# For LLM integration (pick one)
openai==1.3.0  # For GPT-4
anthropic==0.7.0  # For Claude

# Dev
jupyter==1.0.0
ipython==8.16.0
black==23.10.1
pytest==7.4.3
```

**Installation:**
```bash
pip install -r requirements.txt
```

---

## KEY METRICS

### Classifier Metrics

```
MUST ACHIEVE:
├─ Accuracy ≥ 80% (can distinguish modes)
├─ Precision ≥ 0.75 (minimize false positives)
├─ Recall ≥ 0.75 (catch actual extraction-seeking)
└─ F1 ≥ 0.75

NICE TO HAVE:
├─ Accuracy ≥ 85%
├─ Precision ≥ 0.82
└─ F1 ≥ 0.80
```

### RL Training Metrics

```
Learning curve should show:
├─ Episode 1-100: avg reward oscillates around 0.0
├─ Episode 100-200: avg reward increases to +0.15
├─ Episode 200-300: avg reward reaches +0.35
└─ Episode 300-400: avg reward plateaus at +0.38

If learning curve doesn't improve after 200 episodes:
└─ Debug: Check reward calculation, state encoding, network architecture
```

### Policy Evaluation Metrics

```
MUST ACHIEVE:
├─ Policy superior in ≥ 50% of cases (at least 10/20)
├─ Average policy score > baseline score
└─ Baseline leaks answer in ≥ 50% of cases (shows problem is real)

NICE TO HAVE:
├─ Policy superior in ≥ 70% of cases (14/20)
├─ Policy score ≥ 3.5/5
└─ Baseline score ≤ 2.5/5
```

---

## RUNNING INSTRUCTIONS

### Step 1: Prepare Data (Week 1-3)

```bash
# Download StudyChat
git clone https://github.com/[studychat-repo]
# Or manually download from Hugging Face

# Convert to standard format
python scripts/0_prepare_data.py --input raw_studychat.csv --output data/studychat_full.csv

# Run labeling interface (interactive)
python scripts/1_label_data.py --input data/studychat_full.csv --output data/labeled_500.csv

# Create splits
python scripts/1b_create_splits.py --input data/labeled_500.csv
# Generates: data/train_300.csv, data/val_100.csv, data/test_100.csv
```

### Step 2: Train Classifier (Week 4-6)

```bash
python scripts/2_train_classifier.py \
  --train_data data/train_300.csv \
  --val_data data/val_100.csv \
  --test_data data/test_100.csv \
  --output_dir models/ \
  --epochs 3

# Output:
# - models/classifier_v1.pth
# - results/classifier_metrics.json
# - results/classifier_report.txt
```

### Step 3: Causal Analysis (Week 6-9)

```bash
python scripts/3_causal_analysis.py \
  --studychat_data data/studychat_full.csv \
  --classifier models/classifier_v1.pth \
  --output results/causal_analysis.csv

# Output:
# - results/causal_analysis.csv
# - results/causal_plot.png
```

### Step 4: Train RL Policy (Week 9-12)

```bash
python scripts/4_train_rl_policy.py \
  --studychat_data data/studychat_full.csv \
  --classifier models/classifier_v1.pth \
  --output_dir models/ \
  --episodes 400 \
  --learning_rate 1e-4

# Output:
# - models/policy_network_v1.pth
# - results/learning_curve.png
# - results/training_log.csv
```

### Step 5: Evaluate (Week 12-14)

```bash
python scripts/5_evaluate.py \
  --classifier models/classifier_v1.pth \
  --policy models/policy_network_v1.pth \
  --test_queries data/test_100.csv \
  --output results/evaluation_results.json

# Output:
# - results/evaluation_results.json
# - results/example_comparisons.md
# - results/evaluation_report.txt
```

### Step 6: Demo (Interactive)

```bash
python scripts/demo.py \
  --classifier models/classifier_v1.pth \
  --policy models/policy_network_v1.pth

# Prompts:
# > Enter student message: "I'm stuck, give me the code"
# 
# Extraction score: 0.94
# Engagement: 0.2
# Action: REDIRECT (Q-value: 0.88)
# 
# Tutor Response:
# "I see you're asking for the solution directly..."
#
# (repeat)
```

### Full Pipeline (One Command)

```bash
# Run everything in sequence
bash scripts/run_all.sh
```

---

## DEBUGGING CHECKLIST

### Classifier Not Training Well

```
If accuracy < 75% after 3 epochs:
☐ Check class balance in labeled data
  - Are there ~250 extraction, ~250 learning?
☐ Check label quality
  - Do inter-rater labels agree?
  - Are ambiguous cases labeled correctly?
☐ Increase epochs to 5
☐ Lower learning rate to 1e-5
☐ Add more training data (label 100 more turns)
```

### RL Learning Curve Not Improving

```
If reward stays near 0.0 after 200 episodes:
☐ Check reward calculation
  - Is extraction score actually changing?
  - Are you calculating before/after correctly?
☐ Check state encoding
  - Is 771-dim vector being passed correctly?
  - Are embeddings normalized?
☐ Increase exploration (epsilon=0.3 instead of 0.2)
☐ Lower learning rate (1e-5 instead of 1e-4)
☐ Check LLM integration
  - Are system prompts actually different?
  - Is LLM respecting system prompt?
```

### Policy Not Better Than Baseline

```
If policy wins < 50% of cases:
☐ Check reward signal is meaningful
  - Sample some student→next_student pairs
  - Does extraction score actually drop?
☐ Check that actions are diverse
  - Is policy stuck in one action?
  - Print Q-values to inspect
☐ Increase training episodes (500 instead of 400)
☐ Verify baseline is actually standard (not already good)
```

---

## COMMON PITFALLS TO AVOID

1. **Using same split for training and evaluation**
   - Always use held-out test set
   - Never train on test data

2. **Not normalizing message embeddings**
   - BERT output should be normalized
   - Affects policy network input

3. **Reward signal too noisy**
   - Single turn might not be enough
   - Consider smoothing or multi-step rewards

4. **Forgetting to disable gradients during inference**
   - Use `torch.no_grad()` when not training
   - Speeds up prediction 10x

5. **LLM not respecting system prompt**
   - Test system prompts manually first
   - Make prompts very explicit
   - Temperature=0 (deterministic)

6. **Over-fitting on test set**
   - Don't tune hyperparameters on test set
   - Use validation set only

---

## OPTIONAL ENHANCEMENTS (If Time Permits)

### Enhancement 1: Multi-Step Rewards

Instead of single-turn reward, consider next 2-3 turns:

```python
reward = (extraction_before - extraction_after_1) * 0.5 + \
         (extraction_before - extraction_after_2) * 0.3 + \
         (extraction_before - extraction_after_3) * 0.2
```

### Enhancement 2: Context Attention

Include conversation history in state:

```python
state = {
    "current_message_embedding": 768-dim,
    "previous_messages": [msg1_embedding, msg2_embedding, ...],
    "extraction_score": float,
    "engagement": float
}
```

### Enhancement 3: Imitation Learning

Pre-train policy network on "expert" annotated interventions:

```
- Collect 50 dialogues with expert-labeled best actions
- Pre-train policy network with supervised learning
- Then fine-tune with RL
```

### Enhancement 4: User Study

If budget allows:
- Deploy system with 10-20 real students
- Compare to baseline (no policy)
- Measure learning outcomes

---

## CONTACT & TROUBLESHOOTING

**Common Questions:**

Q: How long does classifier training take?
A: ~10 minutes on GPU, ~1 hour on CPU

Q: How long does RL training take?
A: ~2-4 hours on GPU (400 episodes)

Q: Can I use different LLM (Claude vs. GPT-4)?
A: Yes, just change LLM API call in `src/adaptive_tutor.py`

Q: What if I don't have access to GPT-4/Claude?
A: You can still run classifier + RL. For demo, use smaller open model (Llama, Mistral)

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Oct 2026 | Initial specification |

---

**END OF SPECIFICATION**

All code should be:
- ✅ Well-commented
- ✅ Runnable end-to-end
- ✅ Saving outputs at each step
- ✅ Printing progress/metrics
- ✅ Ready for evaluation

Good luck building! 🚀
