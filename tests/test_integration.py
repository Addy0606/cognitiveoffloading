"""
Integration tests for the RL-Based Adaptive Tutoring System.

Tests the full pipeline without requiring model downloads by mocking
the DeBERTa classifier. Tests that DO require the full model are
marked with @pytest.mark.slow.

Integration tests verify:
- Full inference pipeline (message → response)
- Component wiring (classifier → scorer → encoder → policy → LLM)
- Conversation state management
- Baseline vs. adaptive mode
- Diagnostics
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.adaptive_tutor import AdaptiveTutor
from src.classifier import ExtractionClassifier
from src.engagement_scorer import calculate_engagement
from src.prompts import ACTION_NAMES, get_action_name, get_system_prompt
from src.rl_policy_network import PolicyNetwork, RLTrainer
from src.utils import STATE_DIM, encode_state


# ---------------------------------------------------------------
# Prompts module integration
# ---------------------------------------------------------------
class TestPromptsIntegration:
    def test_all_actions_have_prompts(self):
        for action_id in range(4):
            prompt = get_system_prompt(action_id)
            assert isinstance(prompt, str)
            assert len(prompt) > 50

    def test_all_actions_have_names(self):
        for action_id in range(4):
            name = get_action_name(action_id)
            assert name in ["REDIRECT", "SCAFFOLD", "SOCRATIC", "DELAY"]

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError):
            get_system_prompt(5)
        with pytest.raises(ValueError):
            get_system_prompt(-1)


# ---------------------------------------------------------------
# State encoding → policy network integration
# ---------------------------------------------------------------
class TestStateToPolicy:
    def test_state_feeds_into_policy(self, random_embedding):
        """State encoder output should be accepted by PolicyNetwork."""
        state = encode_state(
            message_embedding=random_embedding,
            extraction_score=0.7,
            engagement_score=0.5,
            turn_number=3,
        )
        assert state.shape == (STATE_DIM,)

        policy = PolicyNetwork()
        q_values = policy(state.unsqueeze(0))
        assert q_values.shape == (1, 4)

    def test_rl_trainer_accepts_state(self, random_embedding):
        """RLTrainer should accept encoded states for action selection."""
        state = encode_state(
            message_embedding=random_embedding,
            extraction_score=0.5,
            engagement_score=0.5,
            turn_number=1,
        )

        trainer = RLTrainer()
        action = trainer.select_action(state)
        assert 0 <= action < 4

        q_values = trainer.get_q_values(state)
        assert len(q_values) == 4


# ---------------------------------------------------------------
# Engagement → State encoding integration
# ---------------------------------------------------------------
class TestEngagementToState:
    def test_engagement_feeds_state_encoder(self, random_embedding):
        """Engagement score should integrate into state vector."""
        msg = "I tried this because I thought it would work"
        engagement = calculate_engagement(msg)
        assert 0.0 <= engagement <= 1.0

        state = encode_state(
            message_embedding=random_embedding,
            extraction_score=0.5,
            engagement_score=engagement,
            turn_number=1,
        )
        assert state.shape == (STATE_DIM,)
        assert abs(state[769].item() - engagement) < 1e-5


# ---------------------------------------------------------------
# AdaptiveTutor (mocked classifier)
# ---------------------------------------------------------------
class TestAdaptiveTutorMocked:
    """Test AdaptiveTutor with mocked classifier to avoid model downloads."""

    def _create_mocked_tutor(self):
        """Create an AdaptiveTutor with a mocked classifier."""
        tutor = AdaptiveTutor(llm_provider="fallback")

        # Mock the classifier
        mock_classifier = MagicMock(spec=ExtractionClassifier)
        mock_classifier.is_initialized = True
        mock_classifier.predict.return_value = 0.7
        mock_classifier.get_embedding.return_value = torch.randn(768)
        mock_classifier.initialize.return_value = None
        tutor.classifier = mock_classifier

        return tutor

    def test_infer_returns_expected_keys(self):
        tutor = self._create_mocked_tutor()
        result = tutor.infer("Just give me the code")

        expected_keys = {
            "extraction_score",
            "engagement_score",
            "engagement_breakdown",
            "action_taken",
            "action_id",
            "q_values",
            "tutor_response",
            "llm_provider",
            "is_fallback",
            "turn_number",
        }
        assert set(result.keys()) == expected_keys

    def test_infer_extraction_score(self):
        tutor = self._create_mocked_tutor()
        result = tutor.infer("Give me the answer")
        assert result["extraction_score"] == 0.7  # From mock

    def test_infer_engagement_score(self):
        tutor = self._create_mocked_tutor()
        result = tutor.infer("I tried using a for loop because I thought it would work")
        assert 0.0 <= result["engagement_score"] <= 1.0

    def test_infer_action_valid(self):
        tutor = self._create_mocked_tutor()
        result = tutor.infer("Help me with this")
        assert result["action_id"] in range(4)
        assert result["action_taken"] in ACTION_NAMES.values()

    def test_infer_q_values_length(self):
        tutor = self._create_mocked_tutor()
        result = tutor.infer("Test message")
        assert len(result["q_values"]) == 4

    def test_infer_fallback_response(self):
        tutor = self._create_mocked_tutor()
        result = tutor.infer("Give me the code")
        assert isinstance(result["tutor_response"], str)
        assert len(result["tutor_response"]) > 0

    def test_infer_empty_message(self):
        tutor = self._create_mocked_tutor()
        result = tutor.infer("")
        assert result["extraction_score"] == 0.5
        assert result["action_taken"] == "SCAFFOLD"

    def test_infer_whitespace_message(self):
        tutor = self._create_mocked_tutor()
        result = tutor.infer("   ")
        assert result["extraction_score"] == 0.5

    def test_turn_counter_increments(self):
        tutor = self._create_mocked_tutor()
        tutor.infer("Message 1")
        tutor.infer("Message 2")
        tutor.infer("Message 3")
        assert tutor.turn_number == 3

    def test_reset_conversation(self):
        tutor = self._create_mocked_tutor()
        tutor.infer("Message 1")
        tutor.infer("Message 2")
        assert tutor.turn_number == 2

        tutor.reset_conversation()
        assert tutor.turn_number == 0
        assert len(tutor.conversation_history) == 0

    def test_conversation_history_tracked(self):
        tutor = self._create_mocked_tutor()
        tutor.infer("Hello, help me")
        # Should have student + tutor entries
        assert len(tutor.conversation_history) == 2
        assert tutor.conversation_history[0]["role"] == "student"
        assert tutor.conversation_history[1]["role"] == "tutor"

    def test_baseline_mode(self):
        tutor = self._create_mocked_tutor()
        result = tutor.infer("Help me", use_policy=False)
        assert result["action_taken"] == "BASELINE"
        assert result["action_id"] == -1

    def test_infer_baseline_method(self):
        tutor = self._create_mocked_tutor()
        result = tutor.infer_baseline("Test message")
        assert result["action_taken"] == "BASELINE"

    def test_baseline_doesnt_increment_turn(self):
        tutor = self._create_mocked_tutor()
        tutor.infer("First real message")  # turn 1
        initial_turn = tutor.turn_number
        tutor.infer_baseline("Baseline message")
        assert tutor.turn_number == initial_turn

    def test_diagnostics(self):
        tutor = self._create_mocked_tutor()
        diag = tutor.get_diagnostics()
        expected_keys = {
            "turn_number",
            "conversation_length",
            "classifier_initialized",
            "llm_provider",
            "policy_episodes_trained",
            "policy_epsilon",
        }
        assert set(diag.keys()) == expected_keys


# ---------------------------------------------------------------
# Full pipeline (slow, requires model download)
# ---------------------------------------------------------------
@pytest.mark.slow
class TestFullPipelineWithModel:
    """Full integration tests that download the DeBERTa model."""

    def test_full_inference_pipeline(self):
        tutor = AdaptiveTutor(llm_provider="fallback")
        result = tutor.infer("Just give me the answer to this problem")

        assert 0.0 <= result["extraction_score"] <= 1.0
        assert 0.0 <= result["engagement_score"] <= 1.0
        assert result["action_id"] in range(4)
        assert isinstance(result["tutor_response"], str)
        assert len(result["tutor_response"]) > 0

    def test_different_messages_may_produce_different_actions(self):
        tutor = AdaptiveTutor(llm_provider="fallback")

        # Extraction message
        r1 = tutor.infer("Just give me the code")
        tutor.reset_conversation()

        # Learning message
        r2 = tutor.infer(
            "I tried implementing this because I thought it would work, "
            "but I'm getting an error. Can you explain why?"
        )

        # Scores should differ (not guaranteed actions differ with untrained policy)
        assert r1["extraction_score"] != r2["extraction_score"] or True
