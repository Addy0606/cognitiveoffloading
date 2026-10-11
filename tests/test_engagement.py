"""
Tests for the engagement scorer module.

Tests verify:
- Each sub-scorer returns values within documented bounds
- Empty/None input handling
- Known-good examples produce expected score ranges
- Component scores sum correctly
- Detailed breakdown matches total
"""

import pytest
from src.engagement_scorer import (
    calculate_engagement,
    calculate_engagement_detailed,
    score_context,
    score_questions,
    score_reasoning,
    score_turn_length,
)


# ---------------------------------------------------------------
# score_reasoning
# ---------------------------------------------------------------
class TestScoreReasoning:
    def test_empty_message(self):
        assert score_reasoning("") == 0.0

    def test_none_message(self):
        assert score_reasoning(None) == 0.0

    def test_whitespace_only(self):
        assert score_reasoning("   ") == 0.0

    def test_no_reasoning(self):
        """Message with no reasoning phrases should score 0."""
        assert score_reasoning("Hello world") == 0.0

    def test_single_reasoning_phrase(self):
        """One reasoning phrase → 0.15."""
        score = score_reasoning("I tried using a for loop")
        assert score == 0.15

    def test_two_reasoning_phrases(self):
        """Two reasoning phrases → 0.25."""
        score = score_reasoning("I tried using a for loop because it seemed right")
        assert score == 0.25

    def test_three_or_more_reasoning_phrases(self):
        """Three or more phrases → 0.4."""
        msg = "I tried this because I thought it would work, however it didn't"
        score = score_reasoning(msg)
        assert score == 0.4

    def test_score_within_bounds(self):
        messages = [
            "Just give me the code",
            "I think this is right because I tested it however it breaks",
            "I realized after I tried this that my understanding was wrong",
        ]
        for msg in messages:
            score = score_reasoning(msg)
            assert 0.0 <= score <= 0.4, f"Score {score} out of bounds for: {msg}"


# ---------------------------------------------------------------
# score_turn_length
# ---------------------------------------------------------------
class TestScoreTurnLength:
    def test_empty(self):
        assert score_turn_length("") == 0.0

    def test_none(self):
        assert score_turn_length(None) == 0.0

    def test_very_short(self):
        """< 5 words → 0.0."""
        assert score_turn_length("help me") == 0.0

    def test_short(self):
        """5-14 words → 0.05."""
        msg = "I need help with this specific problem please"
        score = score_turn_length(msg)
        assert score == 0.05

    def test_medium(self):
        """15-29 words → 0.10."""
        msg = " ".join(["word"] * 20)
        score = score_turn_length(msg)
        assert score == 0.10

    def test_long(self):
        """30-59 words → 0.15."""
        msg = " ".join(["word"] * 40)
        score = score_turn_length(msg)
        assert score == 0.15

    def test_very_long(self):
        """60+ words → 0.20 (capped)."""
        msg = " ".join(["word"] * 100)
        score = score_turn_length(msg)
        assert score == 0.20

    def test_score_within_bounds(self):
        for n_words in [0, 3, 10, 25, 50, 100]:
            msg = " ".join(["word"] * n_words) if n_words > 0 else ""
            score = score_turn_length(msg)
            assert 0.0 <= score <= 0.2


# ---------------------------------------------------------------
# score_questions
# ---------------------------------------------------------------
class TestScoreQuestions:
    def test_empty(self):
        assert score_questions("") == 0.0

    def test_no_questions(self):
        assert score_questions("This is a statement.") == 0.0

    def test_single_question(self):
        score = score_questions("Why does this function return None?")
        assert score == 0.10

    def test_multiple_questions(self):
        msg = "Why does this happen? How can I fix it? What if I try recursion?"
        score = score_questions(msg)
        assert score == 0.20

    def test_score_within_bounds(self):
        msgs = [
            "Hello",
            "Why does this work?",
            "How can I fix this and what if it fails?",
        ]
        for msg in msgs:
            assert 0.0 <= score_questions(msg) <= 0.2


# ---------------------------------------------------------------
# score_context
# ---------------------------------------------------------------
class TestScoreContext:
    def test_empty(self):
        assert score_context("") == 0.0

    def test_no_context(self):
        assert score_context("Help me with my homework") == 0.0

    def test_code_block(self):
        msg = "Here's my code:\n```\ndef foo(): pass\n```"
        score = score_context(msg)
        assert score > 0.0

    def test_error_mention(self):
        msg = "I'm getting an error when I run the function"
        score = score_context(msg)
        assert score > 0.0

    def test_multiple_context_indicators(self):
        msg = "I have this error in my output when I run def foo(): return result"
        score = score_context(msg)
        assert score > 0.0

    def test_score_within_bounds(self):
        msgs = [
            "Hello",
            "error traceback exception def foo import os for x in range while True if x:",
        ]
        for msg in msgs:
            assert 0.0 <= score_context(msg) <= 0.2


# ---------------------------------------------------------------
# calculate_engagement
# ---------------------------------------------------------------
class TestCalculateEngagement:
    def test_empty(self):
        assert calculate_engagement("") == 0.0

    def test_none(self):
        assert calculate_engagement(None) == 0.0

    def test_extraction_message_low_engagement(self):
        """Short, no-reasoning extraction messages should score low."""
        score = calculate_engagement("Just give me the code")
        assert score < 0.3

    def test_learning_message_higher_engagement(self):
        """Thoughtful learning messages should score higher."""
        msg = (
            "I tried using a for loop because I thought iterating through "
            "the array would work, however I'm getting an error. "
            "Why does this happen? Can you explain the difference between "
            "indexing with range(len(arr)) vs range(len(arr)+1)?"
        )
        score = calculate_engagement(msg)
        assert score >= 0.3

    def test_always_between_0_and_1(self):
        messages = [
            "",
            "hi",
            "Just give me the answer",
            "I tried this because I thought X however it failed and I realized Y "
            "but then I noticed Z. Why does this happen? How can I fix it? "
            "What if I use a different approach? " * 5,
        ]
        for msg in messages:
            score = calculate_engagement(msg)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1]"

    def test_maximum_possible_score(self):
        """A message hitting all signals should approach 1.0 but not exceed it."""
        msg = (
            "I tried implementing this because I thought the approach would work, "
            "however I realized my understanding was wrong. I noticed that the error "
            "traceback shows an exception in my function def solve(arr): return output "
            "for x in arr. Why does this happen? How can I fix this issue? "
            "What if I tried a different algorithm? " * 3
        )
        score = calculate_engagement(msg)
        assert 0.5 <= score <= 1.0


# ---------------------------------------------------------------
# calculate_engagement_detailed
# ---------------------------------------------------------------
class TestCalculateEngagementDetailed:
    def test_empty(self):
        result = calculate_engagement_detailed("")
        assert result["total"] == 0.0
        assert all(v == 0.0 for v in result.values())

    def test_has_all_keys(self):
        result = calculate_engagement_detailed("I tried this because it seemed right")
        expected_keys = {"total", "reasoning", "length", "questions", "context"}
        assert set(result.keys()) == expected_keys

    def test_total_equals_sum_of_components(self):
        """Total should equal the sum of individual components (clamped to 1.0)."""
        msg = (
            "I tried this because I thought it would work. "
            "Why does this return an error in the output? How can I fix it?"
        )
        result = calculate_engagement_detailed(msg)
        component_sum = (
            result["reasoning"]
            + result["length"]
            + result["questions"]
            + result["context"]
        )
        expected_total = min(max(component_sum, 0.0), 1.0)
        assert abs(result["total"] - expected_total) < 1e-9

    def test_component_bounds(self):
        """Each component should be within its documented bounds."""
        msg = "I tried this because I realized the error was in my code"
        result = calculate_engagement_detailed(msg)
        assert 0.0 <= result["reasoning"] <= 0.4
        assert 0.0 <= result["length"] <= 0.2
        assert 0.0 <= result["questions"] <= 0.2
        assert 0.0 <= result["context"] <= 0.2
        assert 0.0 <= result["total"] <= 1.0
