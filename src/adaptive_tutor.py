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
        self.struggle_turns = 0
        self.is_post_break = False
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
        topic_context: Optional[str] = None,
        student_context: Optional[str] = None,
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

        msg_lower = student_message.lower().strip()

        # Step 0: Check if student is returning from break or switching topics (Idea 1)
        break_return_cues = [
            "i'm back", "im back", "i am back", "took a break", "after break",
            "from break", "rested", "feeling better", "ready now", "let's try again",
            "lets try again", "new topic", "next problem", "next question",
            "another problem", "different problem", "start fresh", "reset"
        ]
        is_returning = any(cue in msg_lower for cue in break_return_cues)
        if is_returning:
            logger.info("Student returned from break or switched topic — resetting struggle counter")
            self.struggle_turns = 0
            self.is_post_break = True

        # Step 1: Compute extraction probability
        self.classifier.initialize()
        extraction_score = self.classifier.predict(student_message)

        # Step 2: Compute engagement score
        engagement_details = calculate_engagement_detailed(student_message)
        engagement_score = engagement_details["total"]

        # Step 2b: Update dynamic struggle counter (Idea 2)
        # If student shows good engagement or asks thoughtful questions, decrease struggle
        # If student persistently demands code or stays stuck, increase struggle
        if is_returning:
            self.struggle_turns = 0
        elif engagement_score > 0.20 or extraction_score < 0.25:
            # Active learning / productive engagement — decay struggle count
            self.struggle_turns = max(0, self.struggle_turns - 1)
        elif extraction_score > 0.6 or engagement_score < 0.08:
            # High extraction or zero-effort short message — accumulate struggle
            self.struggle_turns += 1

        effective_turns = self.struggle_turns if not self.is_post_break else 0

        # Step 3: Get text embedding and construct state vector using effective turns
        embedding = self.classifier.get_embedding(student_message)
        state = encode_state(
            message_embedding=embedding,
            extraction_score=extraction_score,
            engagement_score=engagement_score,
            turn_number=effective_turns,
        )

        # Step 4: Get Q-values and select action
        if use_policy:
            q_values = self.rl_trainer.get_q_values(state)
            action_id = self.rl_trainer.select_action(state, greedy=True)
        else:
            q_values = [0.0, 0.0, 0.0, 0.0]
            action_id = -1  # baseline mode

        # Step 4b: Apply rule-based guardrails to override undertrained policy
        if use_policy:
            action_id = self._apply_guardrails(
                action_id,
                student_message,
                extraction_score,
                engagement_score,
                is_returning=is_returning,
            )

        action_name = get_action_name(action_id) if action_id >= 0 else "BASELINE"

        # Step 5: Generate system prompt
        if use_policy:
            system_prompt = get_system_prompt(action_id)
        else:
            system_prompt = get_baseline_prompt()

        if topic_context and topic_context.strip():
            system_prompt += f"\n\nCURRENT DSA TOPIC CONTEXT:\n{topic_context.strip()}"
        if student_context and student_context.strip():
            system_prompt += f"\n\nSTUDENT LEARNING HISTORY & RETENTION CONTEXT:\n{student_context.strip()}"

        # Step 6: Call LLM (with conversation history for multi-turn context)
        is_fallback = self.llm_provider == "fallback"
        try:
            tutor_response = call_llm(
                system_prompt=system_prompt,
                user_message=student_message,
                provider=self.llm_provider,
                model=self.llm_model,
                conversation_history=self.conversation_history,
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
        self.struggle_turns = 0
        self.is_post_break = False
        self.conversation_history = []

    def _apply_guardrails(
        self,
        policy_action: int,
        student_message: str,
        extraction_score: float,
        engagement_score: float,
        is_returning: bool = False,
    ) -> int:
        """
        Rule-based safety net that overrides the RL policy for obvious cases.

        The RL policy is trained with offline Q-learning. These guardrails catch
        clear-cut pedagogical situations to ensure reliable responses.

        Override rules (in priority order):
            0. Return from break / fresh start → REDIRECT or SCAFFOLD (never duplicate DELAY)
            1. High extraction (>0.7) → REDIRECT (action 0)
            2. Medium extraction (>0.4) + zero engagement → REDIRECT (action 0)
            3. Dismissive/lazy short messages → REDIRECT (action 0)
            4. Student explicitly says stuck/confused/lost → SCAFFOLD (action 1)
            5. Student shows partial understanding → SOCRATIC (action 2)
            6. Rushed, impatient messages → DELAY (action 3)

        Args:
            policy_action: The action chosen by the RL policy.
            student_message: The student's raw message text.
            extraction_score: Classifier extraction probability.
            engagement_score: Engagement heuristic score.
            is_returning: Whether student is returning from break or switching topic.

        Returns:
            Final action ID (possibly overridden).
        """
        msg_lower = student_message.lower().strip()
        word_count = len(msg_lower.split())

        # --- Rule 0: Return from break or fresh topic start ---
        if is_returning:
            if extraction_score > 0.4:
                logger.info("Guardrail override: Post-break extraction demand → REDIRECT")
                return 0  # REDIRECT
            if policy_action == 3:
                logger.info("Guardrail override: Student returned from break, preventing duplicate DELAY → SCAFFOLD")
                return 1  # SCAFFOLD

        if self.is_post_break:
            self.is_post_break = False
            if extraction_score > 0.4:
                logger.info("Guardrail override: Post-break extraction attempt → REDIRECT")
                return 0  # REDIRECT

        # --- Rule 1: High extraction (>0.7) with struggle escalation ---
        if extraction_score > 0.7:
            # Pedagogical escalation: if the student has repeatedly asked/struggled:
            # 1-2 turns: REDIRECT (withhold code, prompt independent thinking)
            # 3-4 turns: SCAFFOLD (provide structured steps/hints, never full code)
            # 5+ turns:  DELAY (suggest a break to prevent burnout)
            if self.struggle_turns >= 5:
                logger.info(
                    f"Guardrail escalation: High struggle ({self.struggle_turns} turns) "
                    f"with extraction → DELAY (Action 3)"
                )
                return 3  # DELAY
            elif self.struggle_turns >= 3:
                logger.info(
                    f"Guardrail escalation: Persistent struggle ({self.struggle_turns} turns) "
                    f"with extraction → SCAFFOLD (Action 1)"
                )
                return 1  # SCAFFOLD
            else:
                if policy_action != 0:
                    logger.info(
                        f"Guardrail override: {policy_action}→REDIRECT "
                        f"(extraction={extraction_score:.2f} > 0.7, struggle={self.struggle_turns})"
                    )
                return 0  # REDIRECT

        # --- Rule 2: Medium extraction + zero engagement → REDIRECT ---
        if extraction_score > 0.4 and engagement_score < 0.05:
            # Dismissive short replies like "no idea", "idk", "whatever"
            dismissive_cues = [
                "no idea", "idk", "i don't know", "i dont know", "whatever",
                "just tell me", "no clue", "dunno", "nope", "can't", "cant",
                "i give up", "too hard", "impossible",
            ]
            if any(cue in msg_lower for cue in dismissive_cues) or word_count <= 3:
                if policy_action != 0:
                    logger.info(
                        f"Guardrail override: {policy_action}→REDIRECT "
                        f"(extraction={extraction_score:.2f}, engagement={engagement_score:.2f}, "
                        f"dismissive short reply)"
                    )
                return 0  # REDIRECT

        # --- Rule 3: Student explicitly stuck → SCAFFOLD ---
        stuck_cues = [
            "i'm stuck", "im stuck", "i am stuck", "don't understand",
            "dont understand", "don't get it", "dont get it", "confused",
            "i'm lost", "im lost", "help me understand", "can you explain",
            "break it down", "where do i start", "how do i begin",
            "not sure how to", "struggling with", "having trouble",
            "can you help me", "i need help", "please help",
        ]
        if any(cue in msg_lower for cue in stuck_cues) and extraction_score < 0.7:
            if policy_action != 1:
                logger.info(
                    f"Guardrail override: {policy_action}→SCAFFOLD "
                    f"(student explicitly stuck)"
                )
            return 1  # SCAFFOLD

        # --- Rule 4: Student shows conceptual inquiry or partial understanding → SOCRATIC ---
        understanding_cues = [
            "i think", "i believe", "my understanding", "i tried",
            "i wrote", "i attempted", "here's my code", "here is my code",
            "i got", "my approach", "i noticed", "i expected",
            "i'm not sure why", "im not sure why", "but why",
            "how come", "what if", "would it work if",
            "how do you", "how does", "how to", "how would", "how can we",
            "what is", "why does", "what does", "difference between",
            "lets start", "let's start", "what about", "tell me about",
            "lets learn", "let's learn", "teach me about",
        ]
        if any(cue in msg_lower for cue in understanding_cues) and extraction_score < 0.6:
            if policy_action != 2:
                logger.info(
                    f"Guardrail override: {policy_action}→SOCRATIC "
                    f"(student shows conceptual inquiry or active reasoning)"
                )
            return 2  # SOCRATIC

        # --- Rule 5: Rushed, impatient messages or cognitive fatigue → DELAY ---
        rush_and_fatigue_cues = [
            "quick question", "real quick", "fast", "asap",
            "right now", "hurry", "urgent", "one liner",
            "in one line", "quickly", "briefly",
            "exhausted", "brain is fried", "brain is dead", "head hurts",
            "need a break", "so tired", "burnout", "burnt out",
            "staring at this for hours", "spent hours", "overwhelmed",
        ]
        if any(cue in msg_lower for cue in rush_and_fatigue_cues) and extraction_score < 0.7:
            if policy_action != 3:
                logger.info(
                    f"Guardrail override: {policy_action}→DELAY "
                    f"(rushed or cognitively exhausted student)"
                )
            return 3  # DELAY

        # No override — trust the policy
        return policy_action

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
