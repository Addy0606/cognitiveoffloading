"""
Utility functions for the RL-Based Adaptive Tutoring System.

Includes: data loading, state encoding, reward calculation, LLM integration,
model saving/loading, synthetic data generation, and logging setup.
"""

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
STATE_DIM = 771          # 768 (embedding) + 1 (extraction) + 1 (engagement) + 1 (turn_num)
EMBEDDING_DIM = 768      # DeBERTa-base hidden size
NUM_ACTIONS = 4
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

ACTION_NAMES = {0: "REDIRECT", 1: "SCAFFOLD", 2: "SOCRATIC", 3: "DELAY"}


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure logging for the project."""
    handlers: list = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


# -------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------
REQUIRED_COLUMNS = {"dialogue_id", "turn_num", "speaker", "message"}
LABELED_COLUMNS = {"dialogue_id", "turn_num", "message", "label"}


def load_csv(path: str, required_columns: Optional[set] = None) -> pd.DataFrame:
    """
    Load a CSV file and validate required columns are present.

    Args:
        path: Path to the CSV file.
        required_columns: Set of column names that must be present.

    Returns:
        DataFrame with the CSV contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if required_columns:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {path.name}: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    return df


def load_dialogues(path: str) -> List[List[Dict[str, Any]]]:
    """
    Load dialogue data grouped by dialogue_id.

    Args:
        path: Path to the CSV file with columns:
              dialogue_id, turn_num, speaker, message.

    Returns:
        List of dialogues, where each dialogue is a list of turn dicts.
    """
    df = load_csv(path, required_columns=REQUIRED_COLUMNS)
    df = df.sort_values(["dialogue_id", "turn_num"])

    dialogues = []
    for _did, group in df.groupby("dialogue_id"):
        turns = group.to_dict("records")
        dialogues.append(turns)

    return dialogues


# -------------------------------------------------------------------
# State Encoding
# -------------------------------------------------------------------
def encode_state(
    message_embedding: torch.Tensor,
    extraction_score: float,
    engagement_score: float,
    turn_number: int,
    max_turns: int = 50,
) -> torch.Tensor:
    """
    Construct the 771-dimensional state vector.

    Args:
        message_embedding: 768-dim embedding tensor (from DeBERTa CLS token).
        extraction_score: Classifier output in [0, 1].
        engagement_score: Engagement scorer output in [0, 1].
        turn_number: Current turn number in the dialogue.
        max_turns: Maximum turn number for normalization (default 50).

    Returns:
        Tensor of shape (771,).

    The embedding is L2-normalized. Scalar features are NOT normalized
    by the same transform — they are appended as-is since they already
    lie in [0, 1] or are explicitly normalized here (turn_number).
    """
    # Ensure embedding is the right shape
    if message_embedding.dim() > 1:
        message_embedding = message_embedding.squeeze(0)
    assert message_embedding.shape == (EMBEDDING_DIM,), (
        f"Expected embedding of shape ({EMBEDDING_DIM},), "
        f"got {message_embedding.shape}"
    )

    # L2-normalize the embedding (only the embedding, not the scalars)
    norm = torch.norm(message_embedding, p=2)
    if norm > 0:
        message_embedding = message_embedding / norm

    # Normalize turn number to [0, 1]
    normalized_turn = min(turn_number / max_turns, 1.0)

    # Concatenate: [768 embedding] + [extraction] + [engagement] + [turn_num]
    scalar_features = torch.tensor(
        [extraction_score, engagement_score, normalized_turn],
        dtype=torch.float32,
    )
    state = torch.cat([message_embedding.float(), scalar_features])

    assert state.shape == (STATE_DIM,), (
        f"State vector has wrong dimension: {state.shape}, expected ({STATE_DIM},)"
    )
    return state


# -------------------------------------------------------------------
# Reward Calculation
# -------------------------------------------------------------------
def calculate_reward(
    extraction_score_before: float,
    extraction_score_after: float,
) -> float:
    """
    Calculate the RL reward from extraction score change.

    reward = extraction_before - extraction_after

    Positive reward: student shifted from extraction → learning mode.
    Negative reward: student shifted toward extraction mode.

    Args:
        extraction_score_before: Classifier score for current student message.
        extraction_score_after: Classifier score for next student message.

    Returns:
        Reward in approximately [-1.0, +1.0].
    """
    return float(extraction_score_before - extraction_score_after)


# -------------------------------------------------------------------
# LLM Integration
# -------------------------------------------------------------------
def call_llm(
    system_prompt: str,
    user_message: str,
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.0,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Call an LLM provider to generate a tutoring response.

    Args:
        system_prompt: The system prompt specifying intervention behavior.
        user_message: The student's message.
        provider: LLM provider ("openai" or "fallback").
        model: Model name override. Defaults to OPENAI_MODEL env var or "gpt-4o".
        temperature: Sampling temperature (0.0 = deterministic).
        conversation_history: Prior conversation turns as a list of
            {"role": "student"|"tutor", "message": "..."}  dicts.

    Returns:
        Generated response string.

    Raises:
        ValueError: If API key is missing (for openai provider).
        RuntimeError: If API call fails.
    """
    if provider == "fallback":
        return _fallback_response(system_prompt, user_message)

    if provider == "openai":
        return _call_openai(system_prompt, user_message, model, temperature, conversation_history)

    if provider == "ollama":
        return _call_ollama(system_prompt, user_message, model, temperature, conversation_history)

    raise ValueError(f"Unknown LLM provider: {provider}")


def _call_openai(
    system_prompt: str,
    user_message: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Call OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-key-here":
        raise ValueError(
            "OPENAI_API_KEY not set. Either:\n"
            "  1. Set OPENAI_API_KEY in your .env file\n"
            "  2. Export OPENAI_API_KEY as an environment variable\n"
            "  3. Use provider='fallback' for template-based responses"
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o")

        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(_build_history_messages(conversation_history))
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")


def _fallback_response(system_prompt: str, user_message: str) -> str:
    """
    Deterministic template-based fallback when no LLM API is available.

    WARNING: These are template responses, NOT LLM-generated.
    They are intended for development and testing only.
    """
    prompt_lower = system_prompt.lower()

    if "redirect" in prompt_lower:
        return (
            "[FALLBACK RESPONSE - Template, not LLM-generated]\n\n"
            "I notice you're looking for a direct answer. Let's take a step back — "
            "what part of this problem have you already thought about? "
            "Try describing your understanding so far, and I'll help guide you from there."
        )
    elif "scaffold" in prompt_lower:
        return (
            "[FALLBACK RESPONSE - Template, not LLM-generated]\n\n"
            "Good effort so far! Here's a hint to help you move forward: "
            "think about breaking the problem into smaller parts. "
            "What would the first step look like? Try that, and let me know what happens."
        )
    elif "socratic" in prompt_lower:
        return (
            "[FALLBACK RESPONSE - Template, not LLM-generated]\n\n"
            "Let me ask you a few questions to help you think this through:\n"
            "1. What is the input to this problem, and what output do you expect?\n"
            "2. Can you think of a simpler version of this problem?\n"
            "3. What approach would you try first, even if it's not optimal?"
        )
    elif "delay" in prompt_lower or "pause" in prompt_lower or "break" in prompt_lower:
        return (
            "[FALLBACK RESPONSE - Template, not LLM-generated]\n\n"
            "It sounds like you've been working hard on this. That's great! "
            "Sometimes taking a short break helps you see the problem with fresh eyes. "
            "When you come back, try starting with just the first step of the problem."
        )
    else:
        return (
            "[FALLBACK RESPONSE - Template, not LLM-generated]\n\n"
            f"I'd be happy to help you with your question. "
            f"Let's work through this together step by step."
        )


def _build_history_messages(
    conversation_history: Optional[List[Dict[str, str]]] = None,
    max_turns: int = 10,
) -> List[Dict[str, str]]:
    """
    Convert internal conversation history into LLM chat messages.

    Keeps the most recent `max_turns` entries to avoid blowing up the
    context window.  Maps 'student' → 'user' and 'tutor' → 'assistant'.
    """
    if not conversation_history:
        return []

    role_map = {"student": "user", "tutor": "assistant"}
    # Take only the last N turns
    recent = conversation_history[-max_turns:]
    return [
        {"role": role_map.get(turn["role"], "user"), "content": turn["message"]}
        for turn in recent
    ]


def _call_ollama(
    system_prompt: str,
    user_message: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Call a local Ollama instance via its REST API.

    Args:
        system_prompt: System prompt with pedagogical instructions.
        user_message: Student query message.
        model: Model name (defaults to OLLAMA_MODEL or 'llama3.2').
        temperature: Generation temperature.
        conversation_history: Prior turns for multi-turn context.

    Returns:
        Generated text response.
    """
    import json
    import urllib.error
    import urllib.request

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model_name = model or os.getenv("OLLAMA_MODEL", "llama3.2")

    # Build messages: system → history → current user message
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_build_history_messages(conversation_history))
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("message", {}).get("content", "").strip()
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to connect to Ollama at {base_url}. Ensure Ollama is running "
            f"and model '{model_name}' is available. Error: {e}"
        )


def get_llm_provider() -> str:
    """
    Determine which LLM provider to use based on available configuration.

    Order of precedence:
    1. Explicit LLM_PROVIDER in environment ("ollama", "openai", "fallback")
    2. OPENAI_API_KEY if present
    3. Ollama if running on localhost
    4. Fallback templates

    Returns:
        'ollama', 'openai', or 'fallback'.
    """
    provider_env = os.getenv("LLM_PROVIDER", "").lower().strip()
    if provider_env in ("ollama", "openai", "fallback"):
        return provider_env

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "your-key-here":
        return "openai"

    # Check if Ollama is accessible
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        import urllib.request
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=1) as resp:
            if resp.status == 200:
                return "ollama"
    except Exception:
        pass

    logger.warning(
        "No active LLM provider found (OpenAI key or Ollama). "
        "Using template-based fallback responses."
    )
    return "fallback"


# -------------------------------------------------------------------
# Synthetic Data Generation
# -------------------------------------------------------------------
# Extraction-oriented student messages (label=1)
SYNTHETIC_EXTRACTION_MESSAGES = [
    "Just give me the code for this problem",
    "What's the answer to question 3?",
    "Can you solve this for me?",
    "Tell me the solution",
    "I need the code that does X, just paste it",
    "What's the output of this function?",
    "Give me the implementation",
    "Just tell me how to do it",
    "What do I write for this assignment?",
    "Send me the complete code",
    "I don't want to figure it out, just tell me",
    "Give me the answer",
    "Copy paste the solution please",
    "What's the correct code?",
    "Just show me what to write",
    "I need the answer now",
    "Stop explaining, just give the code",
    "Can you write this function for me?",
    "What goes in the blank?",
    "Give me a working solution",
    "Please just solve it",
    "I'm stuck, give me the code",
    "Tell me the steps without me thinking",
    "What's the shortcut to solve this?",
    "Just do it for me",
    "Hand me the code for sorting",
    "I just want the right answer, no explanations",
    "Paste the solution here",
    "Do my homework please",
    "What's the formula?",
]

# Learning-oriented student messages (label=0)
SYNTHETIC_LEARNING_MESSAGES = [
    "I tried using a for loop but I'm getting an index out of range error. Here's my code: for i in range(len(arr)+1). Why does this happen?",
    "I understand that recursion involves a base case, but I'm confused about how the stack unwinds. Can you explain?",
    "I attempted to implement binary search but my mid calculation seems off. I used mid = (left + right) / 2 — is there a problem with integer division?",
    "Why does quicksort have O(n²) worst case? I thought the pivot should always split evenly",
    "I wrote this function to reverse a linked list but it's returning None. I think the issue is with my pointer reassignment. Here's what I have...",
    "Can you explain the difference between a stack and a queue? I know one is FIFO and one is LIFO but I get confused about which is which",
    "I tried to implement merge sort and the splitting works but my merge function isn't combining correctly. The sorted halves are right but the final array has duplicates",
    "I think I understand how hash maps work — you hash the key to get an index — but what happens when two keys hash to the same index?",
    "My BFS implementation visits nodes but doesn't find the shortest path. I think I need to track the parent of each node? Here's my attempt...",
    "I was thinking about this problem and I believe we need dynamic programming because there are overlapping subproblems. Is that the right approach?",
    "I expected my function to return 5 but it returns None. I think I forgot a return statement somewhere. Here's the code...",
    "Why does my recursive Fibonacci function take so long for n=40? I calculated that it should make about 2^40 calls — is there a better way?",
    "I noticed that adding to the beginning of a Python list is slow. Is this because lists are arrays internally? What should I use instead?",
    "I'm trying to understand Big O notation. When we say O(n log n), does that mean the algorithm always takes exactly n*log(n) steps?",
    "I wrote a class for a binary tree but I'm not sure if my insert method handles duplicates correctly. Should duplicates go left or right?",
    "After reading about graph algorithms, I'm confused about when to use Dijkstra's vs Bellman-Ford. Is it about negative edges?",
    "My sorting function works for small arrays but fails on edge cases like empty arrays and single elements. How should I handle those?",
    "I think the bug is in my loop condition. I have while i <= n but should it be while i < n? The off-by-one errors are confusing me",
    "I tried implementing a trie for autocomplete. The insertion works but my search function doesn't handle the case where the word is a prefix of another word",
    "I believe the time complexity is O(n) because I iterate through the list once, but I'm not sure about the space complexity since I create a dictionary",
    "Can you help me understand why memoization turns exponential time into polynomial? I added a cache but I want to understand why it works",
    "I was working on this problem for an hour and I think I need a different approach. I tried brute force but it times out. Would a sliding window work here?",
    "The error says 'maximum recursion depth exceeded'. I think my base case is wrong — I have if n == 0 but maybe it should be if n <= 1?",
    "I'm comparing my solution to the expected output and they differ for negative numbers. I think my absolute value calculation is wrong",
    "Why does Python pass lists by reference but integers by value? Or is it more nuanced than that? I'm seeing unexpected behavior when I modify a list inside a function",
    "I implemented the algorithm but I want to understand the intuition. Why does the greedy approach work here? It seems like it could miss the optimal solution",
    "My code works correctly but it's not efficient enough. The time limit is 1 second and my O(n²) solution takes 3 seconds. How can I optimize it?",
    "I debugged my code and found that the issue was with how I was updating the visited set. I was adding nodes before checking their neighbors. Is that correct for DFS?",
    "I read that Python dictionaries maintain insertion order since 3.7. Does that mean they use a different data structure than a regular hash table?",
    "I'm confused about the difference between deep copy and shallow copy. When I copy my 2D list, modifying the copy also changes the original",
]

# Tutor responses of various types
SYNTHETIC_TUTOR_RESPONSES = {
    "redirect": [
        "What have you tried so far?",
        "Before I help, can you share your attempt?",
        "Let's think about this step by step. What's the first thing you'd try?",
        "I'd like to help you learn, not just give answers. What part confuses you?",
    ],
    "scaffold": [
        "Good start! You're on the right track with the loop. Now think about the boundary condition — what happens when i equals the length?",
        "Here's a hint: consider what happens when you reach the base case. Your recursive call is correct, but the return value needs adjustment.",
        "You've identified the right approach. Let me show you the first step, and you try the rest: start by sorting the array.",
    ],
    "socratic": [
        "What do you think would happen if you tried a smaller input first?",
        "Can you explain why you chose that data structure? What properties does it need?",
        "If the input were [1, 2, 3], what would your function return at each step?",
    ],
    "direct": [
        "Here's the solution: use a two-pointer approach. Initialize left=0, right=len(arr)-1...",
        "The answer is to use dynamic programming. Here's the complete implementation...",
        "You need to use a hash map. Here's the code: def solve(arr): d = {}; ...",
    ],
}


def generate_synthetic_dataset(
    num_dialogues: int = 50,
    min_turns: int = 4,
    max_turns: int = 12,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic dialogue dataset for pipeline testing.

    WARNING: This data is SYNTHETIC and should NEVER be presented as
    real StudyChat data or used as evidence of real-world performance.

    Args:
        num_dialogues: Number of dialogues to generate.
        min_turns: Minimum turns per dialogue.
        max_turns: Maximum turns per dialogue.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: dialogue_id, turn_num, speaker, message, context.
    """
    rng = random.Random(seed)
    rows = []

    for d_idx in range(num_dialogues):
        dialogue_id = f"DIALOGUE_{d_idx + 1:04d}"
        num_turns = rng.randint(min_turns, max_turns)
        # Make num_turns even so we get student-tutor pairs
        if num_turns % 2 != 0:
            num_turns += 1

        for t in range(num_turns):
            if t % 2 == 0:
                # Student turn
                speaker = "student"
                # Mix of extraction and learning messages
                if rng.random() < 0.4:
                    message = rng.choice(SYNTHETIC_EXTRACTION_MESSAGES)
                else:
                    message = rng.choice(SYNTHETIC_LEARNING_MESSAGES)
            else:
                # Tutor turn
                speaker = "tutor"
                resp_type = rng.choice(list(SYNTHETIC_TUTOR_RESPONSES.keys()))
                message = rng.choice(SYNTHETIC_TUTOR_RESPONSES[resp_type])

            rows.append({
                "dialogue_id": dialogue_id,
                "turn_num": t + 1,
                "speaker": speaker,
                "message": message,
                "context": f"Week {rng.randint(1, 15)}",
            })

    df = pd.DataFrame(rows)
    logger.info(
        f"Generated synthetic dataset: {num_dialogues} dialogues, "
        f"{len(df)} total turns (SYNTHETIC DATA)"
    )
    return df


def generate_synthetic_labeled(
    num_samples: int = 500,
    extraction_ratio: float = 0.4,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic labeled data for classifier pipeline testing.

    WARNING: Labels are assigned by construction (based on message source list),
    NOT by human annotation. These should NEVER be presented as hand-labeled data.

    Args:
        num_samples: Number of labeled examples to generate.
        extraction_ratio: Fraction of samples that are extraction-oriented.
        seed: Random seed.

    Returns:
        DataFrame with columns: dialogue_id, turn_num, message, label, confidence, notes.
    """
    rng = random.Random(seed)
    rows = []
    n_extraction = int(num_samples * extraction_ratio)
    n_learning = num_samples - n_extraction

    # Extraction samples (label=1)
    for i in range(n_extraction):
        msg = rng.choice(SYNTHETIC_EXTRACTION_MESSAGES)
        rows.append({
            "dialogue_id": f"DIALOGUE_{rng.randint(1, 200):04d}",
            "turn_num": rng.randint(1, 20),
            "message": msg,
            "label": 1,
            "confidence": rng.choice(["high", "medium"]),
            "notes": "extraction",
        })

    # Learning samples (label=0)
    for i in range(n_learning):
        msg = rng.choice(SYNTHETIC_LEARNING_MESSAGES)
        rows.append({
            "dialogue_id": f"DIALOGUE_{rng.randint(1, 200):04d}",
            "turn_num": rng.randint(1, 20),
            "message": msg,
            "label": 0,
            "confidence": rng.choice(["high", "medium"]),
            "notes": "learning",
        })

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(
        f"Generated {num_samples} synthetic labeled examples "
        f"(extraction={n_extraction}, learning={n_learning}) — SYNTHETIC DATA"
    )
    return df


# -------------------------------------------------------------------
# Tutor Response Categorization (for offline RL)
# -------------------------------------------------------------------
def categorize_tutor_response(message: str) -> int:
    """
    Heuristically categorize an observed tutor response into one of the
    4 action types based on text content.

    This is an approximation for offline RL. The actual observed tutor may not
    have been following any of these strategies explicitly.

    Args:
        message: The tutor's response text.

    Returns:
        Estimated action ID (0-3).
    """
    if not message or not message.strip():
        return 1  # Default to scaffold

    msg_lower = message.lower()

    # Check for direct answer / solution leaking → might indicate what NOT to do
    direct_patterns = [
        r"here'?s the (solution|answer|code)",
        r"the answer is",
        r"here'?s (a |the )?complete",
        r"def \w+\(.+\):",       # Giving actual function code
    ]
    is_direct = any(re.search(p, msg_lower) for p in direct_patterns)

    # Check for Socratic patterns
    socratic_patterns = [
        r"what (do|would|did) you think",
        r"what happens (if|when)",
        r"can you (explain|think|describe)",
        r"why do you think",
        r"what would happen",
        r"have you considered",
    ]
    is_socratic = any(re.search(p, msg_lower) for p in socratic_patterns)

    # Check for redirect patterns
    redirect_patterns = [
        r"what have you tried",
        r"before i (help|give)",
        r"try (it |this )?first",
        r"show me (your|what you)",
        r"attempt.*first",
    ]
    is_redirect = any(re.search(p, msg_lower) for p in redirect_patterns)

    # Check for delay / break patterns
    delay_patterns = [
        r"take a break",
        r"step away",
        r"come back (later|tomorrow|fresh)",
        r"rest",
        r"it'?s okay to (feel|be) (frustrated|stuck)",
    ]
    is_delay = any(re.search(p, msg_lower) for p in delay_patterns)

    # Priority: delay > redirect > socratic > scaffold (> direct as scaffold)
    if is_delay:
        return 3  # DELAY
    elif is_redirect:
        return 0  # REDIRECT
    elif is_socratic:
        return 2  # SOCRATIC
    else:
        return 1  # SCAFFOLD (default, including direct responses)


# -------------------------------------------------------------------
# Model I/O helpers
# -------------------------------------------------------------------
def get_device() -> torch.device:
    """Get the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data: Any, path: str) -> None:
    """Save data as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON: {path}")


def load_json(path: str) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
