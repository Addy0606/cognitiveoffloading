"""
Adaptive Tutor: main inference engine combining all components.

Pipeline:
    Student Message → Classifier → Engagement Scorer → State Encoder →
    Policy Network → Action Selection → Prompt Generation → LLM → Response

This module integrates:
    - ExtractionClassifier (extraction/learning detection)
    - Engagement scorer (heuristic engagement scoring)
    - State encoder (771-dim state vector)
    - PolicyNetwork (Q-learning intervention selection)
    - LLM integration (response generation)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.classifier import ExtractionClassifier
from src.engagement_scorer import calculate_engagement, calculate_engagement_detailed
from src.prompts import (
    ACTION_NAMES,
    get_action_name,
    get_baseline_prompt,
    get_system_prompt,
)
from src.rl_policy_network import PolicyNetwork, RLTrainer
from src.utils import (
    STATE_DIM,
    call_llm,
    encode_state,
    get_llm_provider,
)

logger = logging.getLogger(__name__)


class AdaptiveTutor:
    """
    Main inference engine: student message → tutor response.

    Combines the extraction classifier, engagement scorer, RL policy network,
    and LLM integration into a single end-to-end inference pipeline.
    """

    def __init__(
        self,
        classifier: Optional[ExtractionClassifier] = None,
        rl_trainer: Optional[RLTrainer] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Args:
            classifier: Pre-initialized ExtractionClassifier.
            rl_trainer: Pre-initialized RLTrainer with loaded policy.
            llm_provider: LLM provider ("openai" or "fallback").
            llm_model: LLM model name override.
        """
        self.classifier = classifier or ExtractionClassifier()
        self.rl_trainer = rl_trainer or RLTrainer()
        self.llm_provider = llm_provider or get_llm_provider()
        self.llm_model = llm_model
        self.turn_number = 0
        self.conversation_history: List[Dict[str, str]] = []

    def load_models(
        self,
        classifier_path: str,
        policy_path: str,
    ) -> None:
        """
        Load trained classifier and policy models from disk.

        Args:
            classifier_path: Path to classifier directory (with backbone/ and head.pth).
            policy_path: Path to policy checkpoint (.pth file).
        """
        # Load classifier
        classifier_dir = Path(classifier_path)
        if classifier_dir.exists():
            self.classifier.load(str(classifier_dir))
            logger.info(f"Classifier loaded from {classifier_dir}")
        else:
            logger.warning(
                f"Classifier path not found: {classifier_dir}. "
                f"Initializing with pretrained model (untrained head)."
            )
            self.classifier.initialize()

        # Load policy
        policy_path_obj = Path(policy_path)
        if policy_path_obj.exists():
            self.rl_trainer.load(str(policy_path_obj))
            logger.info(f"Policy loaded from {policy_path_obj}")
        else:
            logger.warning(
                f"Policy path not found: {policy_path_obj}. "
                f"Using randomly initialized policy."
            )

    def infer(
        self,
        student_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_policy: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full inference pipeline on a student message.

        Args:
            student_message: The student's message text.
            conversation_history: Optional prior conversation turns.
            use_policy: If True, use the RL policy. If False, use baseline.

        Returns:
            Dictionary containing:
                - extraction_score: float (0-1)
                - engagement_score: float (0-1)
                - engagement_breakdown: dict of component scores
                - action_taken: str (e.g., "REDIRECT")
                - action_id: int (0-3)
                - q_values: list of 4 floats
                - tutor_response: str
                - llm_provider: str
                - is_fallback: bool
        """
        if not student_message or not student_message.strip():
            return {
                "extraction_score": 0.5,
                "engagement_score": 0.0,
                "engagement_breakdown": {},
                "action_taken": "SCAFFOLD",
                "action_id": 1,
                "q_values": [0.0, 0.0, 0.0, 0.0],
                "tutor_response": "I didn't catch that. Could you rephrase your question?",
                "llm_provider": "none",
                "is_fallback": True,
            }

        self.turn_number += 1
        if conversation_history:
            self.conversation_history = conversation_history

        # Step 1: Compute extraction probability
        self.classifier.initialize()
        extraction_score = self.classifier.predict(student_message)

        # Step 2: Compute engagement score
        engagement_details = calculate_engagement_detailed(student_message)
        engagement_score = engagement_details["total"]

        # Step 3: Get text embedding and construct state vector
        embedding = self.classifier.get_embedding(student_message)
        state = encode_state(
            message_embedding=embedding,
            extraction_score=extraction_score,
            engagement_score=engagement_score,
            turn_number=self.turn_number,
        )

        # Step 4: Get Q-values and select action
        if use_policy:
            q_values = self.rl_trainer.get_q_values(state)
            action_id = self.rl_trainer.select_action(state, greedy=True)
        else:
            q_values = [0.0, 0.0, 0.0, 0.0]
            action_id = -1  # baseline mode

        action_name = get_action_name(action_id) if action_id >= 0 else "BASELINE"

        # Step 5: Generate system prompt
        if use_policy:
            system_prompt = get_system_prompt(action_id)
        else:
            system_prompt = get_baseline_prompt()

        # Step 6: Call LLM
        is_fallback = self.llm_provider == "fallback"
        try:
            tutor_response = call_llm(
                system_prompt=system_prompt,
                user_message=student_message,
                provider=self.llm_provider,
                model=self.llm_model,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}. Using fallback.")
            tutor_response = call_llm(
                system_prompt=system_prompt,
                user_message=student_message,
                provider="fallback",
            )
            is_fallback = True

        # Step 7: Log interaction (without sensitive content)
        self._log_interaction(
            extraction_score=extraction_score,
            engagement_score=engagement_score,
            action_id=action_id,
            action_name=action_name,
            q_values=q_values,
        )

        # Track conversation
        self.conversation_history.append({
            "role": "student",
            "message": student_message,
        })
        self.conversation_history.append({
            "role": "tutor",
            "message": tutor_response,
        })

        return {
            "extraction_score": extraction_score,
            "engagement_score": engagement_score,
            "engagement_breakdown": engagement_details,
            "action_taken": action_name,
            "action_id": action_id,
            "q_values": q_values,
            "tutor_response": tutor_response,
            "llm_provider": self.llm_provider,
            "is_fallback": is_fallback,
            "turn_number": self.turn_number,
        }

    def infer_baseline(self, student_message: str) -> Dict[str, Any]:
        """Generate a baseline (non-adaptive) response for comparison."""
        # Save and restore turn number so baseline doesn't increment it
        saved_turn = self.turn_number
        result = self.infer(student_message, use_policy=False)
        self.turn_number = saved_turn
        return result

    def reset_conversation(self) -> None:
        """Reset the conversation state for a new dialogue."""
        self.turn_number = 0
        self.conversation_history = []

    def _log_interaction(
        self,
        extraction_score: float,
        engagement_score: float,
        action_id: int,
        action_name: str,
        q_values: List[float],
    ) -> None:
        """Log interaction metadata (no sensitive content)."""
        logger.info(
            f"Interaction: extraction={extraction_score:.3f}, "
            f"engagement={engagement_score:.3f}, "
            f"action={action_name} (id={action_id}), "
            f"Q-values={[f'{q:.3f}' for q in q_values]}"
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the current tutor state."""
        return {
            "turn_number": self.turn_number,
            "conversation_length": len(self.conversation_history),
            "classifier_initialized": self.classifier.is_initialized,
            "llm_provider": self.llm_provider,
            "policy_episodes_trained": self.rl_trainer.episode_count,
            "policy_epsilon": self.rl_trainer.epsilon,
        }
