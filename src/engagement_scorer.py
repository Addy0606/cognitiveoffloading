"""
Engagement scorer: estimates student engagement level from message text.

Combines four heuristic signals:
  1. Evidence of reasoning (0.0 - 0.4)
  2. Message length / effort (0.0 - 0.2)
  3. Clarifying questions (0.0 - 0.2)
  4. Context provided (code, errors, expected output) (0.0 - 0.2)

Total engagement score: 0.0 - 1.0

Note: These heuristics are transparent approximations. Long messages do NOT
automatically get high scores — length is only one of four signals and is
capped at 0.2. Similarly, short messages can still score well if they
show reasoning and ask good questions.
"""

import re
from typing import Dict


# --- Reasoning indicators ---
REASONING_PHRASES = [
    r"\bi tried\b", r"\bi attempted\b", r"\bi thought\b",
    r"\bi think\b", r"\bmy approach\b", r"\bi understand that\b",
    r"\bbecause\b", r"\bso (that|i|it)\b", r"\bwhich means\b",
    r"\bthe reason\b", r"\bmy logic\b", r"\bi expected\b",
    r"\bi figured\b", r"\bi realized\b", r"\bmy understanding\b",
    r"\bwhen i\b", r"\bafter i\b", r"\bi was thinking\b",
    r"\bi noticed\b", r"\bit seems like\b", r"\bi believe\b",
    r"\bbut (then|it|i)\b", r"\bhowever\b", r"\balthough\b",
]

# --- Context indicators ---
CONTEXT_PATTERNS = [
    r"```",                              # Code blocks
    r"\berror\b",                        # Error mentions
    r"\btraceback\b",                    # Python traceback
    r"\boutput\b",                       # Output description
    r"\bexpected\b",                     # Expected behavior
    r"\bresult\b",                       # Result description
    r"\blog\b",                          # Log output
    r"\bexception\b",                    # Exception
    r"def \w+",                          # Function definition
    r"class \w+",                        # Class definition
    r"\bprint\(",                        # Print statement
    r"\breturn\b",                       # Return statement
    r"\bfor .+ in\b",                    # For loop
    r"\bwhile\b",                        # While loop
    r"\bif .+:",                         # If statement
    r"\bimport\b",                       # Import statement
]

# --- Question indicators ---
QUESTION_PATTERNS = [
    r"\bwhy does\b", r"\bhow does\b", r"\bwhat does\b",
    r"\bwhat happens\b", r"\bcan you explain\b",
    r"\bwhat is the difference\b", r"\bwhy is\b",
    r"\bhow can i\b", r"\bwhat if\b", r"\bcould you clarify\b",
    r"\bis it (because|true|correct|possible)\b",
    r"\bdoes (this|it|that) mean\b", r"\bwhy (would|should|do)\b",
]


def score_reasoning(message: str) -> float:
    """
    Score evidence of reasoning in the student's message.

    Checks for phrases indicating the student has thought about the problem,
    tried something, or is building on prior understanding.

    Args:
        message: The student's message text.

    Returns:
        Score between 0.0 and 0.4.
    """
    if not message or not message.strip():
        return 0.0

    msg_lower = message.lower()
    matches = sum(
        1 for pattern in REASONING_PHRASES
        if re.search(pattern, msg_lower)
    )

    # Scale: 0 matches → 0.0, 1 match → 0.15, 2 → 0.25, 3+ → 0.4
    if matches == 0:
        return 0.0
    elif matches == 1:
        return 0.15
    elif matches == 2:
        return 0.25
    else:
        return 0.4


def score_turn_length(message: str) -> float:
    """
    Score message length as a proxy for effort.

    Very short messages get low scores. Medium-length messages get moderate
    scores. Very long messages plateau — length alone does not indicate
    high engagement.

    Args:
        message: The student's message text.

    Returns:
        Score between 0.0 and 0.2.
    """
    if not message or not message.strip():
        return 0.0

    word_count = len(message.split())

    if word_count < 5:
        return 0.0
    elif word_count < 15:
        return 0.05
    elif word_count < 30:
        return 0.10
    elif word_count < 60:
        return 0.15
    else:
        return 0.20


def score_questions(message: str) -> float:
    """
    Score presence of clarifying questions indicating active thinking.

    Higher scores for conceptual questions (\"why\", \"how\") vs. direct
    answer requests.

    Args:
        message: The student's message text.

    Returns:
        Score between 0.0 and 0.2.
    """
    if not message or not message.strip():
        return 0.0

    msg_lower = message.lower()
    matches = sum(
        1 for pattern in QUESTION_PATTERNS
        if re.search(pattern, msg_lower)
    )

    if matches == 0:
        return 0.0
    elif matches == 1:
        return 0.10
    else:
        return 0.20


def score_context(message: str) -> float:
    """
    Score whether the student provides relevant context like code,
    error messages, expected output, or attempted solutions.

    Args:
        message: The student's message text.

    Returns:
        Score between 0.0 and 0.2.
    """
    if not message or not message.strip():
        return 0.0

    msg_lower = message.lower()
    matches = sum(
        1 for pattern in CONTEXT_PATTERNS
        if re.search(pattern, msg_lower)
    )

    if matches == 0:
        return 0.0
    elif matches == 1:
        return 0.07
    elif matches == 2:
        return 0.12
    else:
        return 0.20


def calculate_engagement(message: str) -> float:
    """
    Calculate overall engagement score by combining four signals.

    Args:
        message: The student's message text.

    Returns:
        Engagement score between 0.0 and 1.0.

    Component breakdown:
        - Reasoning evidence: up to 0.4 (largest weight)
        - Message length/effort: up to 0.2
        - Clarifying questions: up to 0.2
        - Context provided: up to 0.2
    """
    if not message or not message.strip():
        return 0.0

    reasoning = score_reasoning(message)
    length = score_turn_length(message)
    questions = score_questions(message)
    context = score_context(message)

    total = reasoning + length + questions + context
    # Clamp to [0, 1] — should not exceed 1.0 given component caps
    return min(max(total, 0.0), 1.0)


def calculate_engagement_detailed(message: str) -> Dict[str, float]:
    """
    Calculate engagement score with per-component breakdown.

    Args:
        message: The student's message text.

    Returns:
        Dictionary with keys:
            'total': overall engagement score (0-1)
            'reasoning': reasoning evidence score (0-0.4)
            'length': message length score (0-0.2)
            'questions': question quality score (0-0.2)
            'context': context provision score (0-0.2)
    """
    if not message or not message.strip():
        return {
            "total": 0.0,
            "reasoning": 0.0,
            "length": 0.0,
            "questions": 0.0,
            "context": 0.0,
        }

    reasoning = score_reasoning(message)
    length = score_turn_length(message)
    questions = score_questions(message)
    context = score_context(message)
    total = min(max(reasoning + length + questions + context, 0.0), 1.0)

    return {
        "total": total,
        "reasoning": reasoning,
        "length": length,
        "questions": questions,
        "context": context,
    }
