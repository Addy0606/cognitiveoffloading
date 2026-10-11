"""
Script 4: Train RL Policy Network

Trains the Q-learning policy on dialogue data using offline RL.

OFFLINE RL LIMITATION:
    The policy learns from OBSERVED dialogue transitions. The reward
    (extraction score change) is associated with whatever action the
    original tutor ACTUALLY took, not a hypothetical action. We use a
    heuristic to categorize the observed tutor response into one of
    4 action types, but this is an approximation.

Usage:
    python scripts/4_train_rl_policy.py --studychat_data data/studychat_full.csv --classifier models/classifier_v1 --output_dir models/ --episodes 400
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import ExtractionClassifier
from src.engagement_scorer import calculate_engagement
from src.rl_policy_network import RLTrainer, PolicyNetwork
from src.utils import (
    categorize_tutor_response,
    encode_state,
    load_dialogues,
    save_json,
    set_seed,
    setup_logging,
    calculate_reward,
    ACTION_NAMES,
)

logger = logging.getLogger(__name__)


def build_transitions(
    dialogues: List[List[Dict]],
    classifier: ExtractionClassifier,
    max_episodes: int = 400,
    max_steps: int = 20,
) -> List[List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]]]:
    """
    Build (state, action, reward, next_state, done) transitions from dialogues.

    Each dialogue becomes an episode. For each student→tutor→student triplet:
      - state: encode(student_msg, extraction_before, engagement, turn_num)
      - action: categorize(tutor_response) — the OBSERVED action, not policy's choice
      - reward: extraction_before - extraction_after
      - next_state: encode(next_student_msg, extraction_after, engagement_after, turn_num+1)
      - done: True if last transition in dialogue

    Args:
        dialogues: List of dialogue turn sequences.
        classifier: Trained ExtractionClassifier.
        max_episodes: Maximum number of dialogues to process.
        max_steps: Maximum transitions per dialogue.

    Returns:
        List of episodes, each a list of transitions.
    """
    episodes = []

    for d_idx, dialogue in enumerate(dialogues[:max_episodes]):
        sorted_turns = sorted(dialogue, key=lambda t: t.get("turn_num", 0))
        transitions = []
        step_count = 0

        i = 0
        while i < len(sorted_turns) - 2 and step_count < max_steps:
            t1 = sorted_turns[i]
            t2 = sorted_turns[i + 1]
            t3 = sorted_turns[i + 2]

            if (t1.get("speaker") == "student" and
                t2.get("speaker") == "tutor" and
                t3.get("speaker") == "student"):

                msg_before = str(t1.get("message", ""))
                tutor_msg = str(t2.get("message", ""))
                msg_after = str(t3.get("message", ""))

                if msg_before.strip() and tutor_msg.strip() and msg_after.strip():
                    # Compute features for current state
                    extraction_before = classifier.predict(msg_before)
                    engagement_before = calculate_engagement(msg_before)
                    embedding_before = classifier.get_embedding(msg_before)
                    state = encode_state(
                        embedding_before, extraction_before,
                        engagement_before, step_count
                    )

                    # Observed action from tutor response
                    observed_action = categorize_tutor_response(tutor_msg)

                    # Compute features for next state
                    extraction_after = classifier.predict(msg_after)
                    engagement_after = calculate_engagement(msg_after)
                    embedding_after = classifier.get_embedding(msg_after)
                    next_state = encode_state(
                        embedding_after, extraction_after,
                        engagement_after, step_count + 1
                    )

                    # Reward
                    reward = calculate_reward(extraction_before, extraction_after)

                    # Check if this is the last valid transition
                    is_last = (i + 4 >= len(sorted_turns))
                    done = is_last

                    transitions.append((state, observed_action, reward, next_state, done))
                    step_count += 1

                i += 2
            else:
                i += 1

        if transitions:
            episodes.append(transitions)

        if (d_idx + 1) % 50 == 0:
            logger.info(f"Processed {d_idx + 1}/{min(len(dialogues), max_episodes)} dialogues")

    logger.info(
        f"Built {len(episodes)} episodes with "
        f"{sum(len(e) for e in episodes)} total transitions"
    )
    return episodes


def plot_learning_curves(
    rewards: List[float],
    losses: List[float],
    output_dir: str,
) -> None:
    """Plot and save RL training learning curves."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Reward curve with smoothing
    if rewards:
        ax1.plot(rewards, alpha=0.3, color="blue", label="Raw")
        # Moving average
        window = min(50, len(rewards) // 4 + 1)
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax1.plot(range(window - 1, len(rewards)), smoothed,
                     color="blue", linewidth=2, label=f"MA({window})")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Episode Reward")
        ax1.set_title("RL Training: Episode Rewards")
        ax1.legend()

    # Loss curve
    if losses:
        ax2.plot(losses, alpha=0.3, color="red", label="Raw")
        window = min(50, len(losses) // 4 + 1)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            ax2.plot(range(window - 1, len(losses)), smoothed,
                     color="red", linewidth=2, label=f"MA({window})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Average Loss")
        ax2.set_title("RL Training: Episode Loss")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(out / "learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Learning curves saved to {out / 'learning_curve.png'}")


def main():
    parser = argparse.ArgumentParser(description="Train RL policy network")
    parser.add_argument("--studychat_data", type=str, default="data/studychat_full.csv")
    parser.add_argument("--classifier", type=str, default="models/classifier_v1")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    setup_logging(log_file="logs/train_rl_policy.log")
    set_seed(args.seed)

    # Load data
    data_path = Path(args.studychat_data)
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        logger.info("Run: python scripts/0_prepare_data.py --synthetic")
        sys.exit(1)

    dialogues = load_dialogues(str(data_path))
    logger.info(f"Loaded {len(dialogues)} dialogues")

    # Load classifier
    classifier = ExtractionClassifier()
    classifier_path = Path(args.classifier)
    if classifier_path.exists():
        classifier.load(str(classifier_path))
        logger.info("Trained classifier loaded")
    else:
        logger.warning(
            f"Classifier not found at {classifier_path}. "
            f"Using untrained model — RL training will learn from random extraction scores."
        )
        classifier.initialize()

    # Build transitions
    logger.info("Building transitions from dialogues...")
    episodes = build_transitions(
        dialogues, classifier,
        max_episodes=args.episodes,
        max_steps=args.max_steps,
    )

    if not episodes:
        logger.error("No valid episodes built. Check data format.")
        sys.exit(1)

    # Initialize trainer
    trainer = RLTrainer(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
    )

    # Training loop
    logger.info(f"Starting RL training: {len(episodes)} episodes")
    print("\n" + "=" * 60)
    print("RL POLICY TRAINING")
    print("=" * 60)

    for ep_idx, transitions in enumerate(episodes):
        metrics = trainer.train_episode(transitions)

        if (ep_idx + 1) % 50 == 0 or ep_idx == 0:
            avg_recent = np.mean(trainer.episode_rewards[-50:]) if trainer.episode_rewards else 0
            print(
                f"Episode {ep_idx + 1}/{len(episodes)}: "
                f"reward={metrics['total_reward']:.3f}, "
                f"avg_50={avg_recent:.3f}, "
                f"loss={metrics['avg_loss']:.4f}, "
                f"epsilon={metrics['epsilon']:.3f}"
            )

    # Save policy
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "policy_network_v1.pth"
    trainer.save(str(policy_path))

    # Save training metrics
    training_metrics = trainer.get_training_metrics()
    training_metrics["data_source"] = str(data_path)
    training_metrics["classifier_source"] = str(classifier_path)
    training_metrics["note"] = (
        "Trained with offline Q-learning on observed dialogue transitions. "
        "Rewards reflect observed extraction score changes, not causal effects "
        "of policy actions."
    )
    save_json(training_metrics, str(Path("results") / "rl_training_metrics.json"))

    # Save training log
    training_log = {
        "episode_rewards": trainer.episode_rewards,
        "episode_losses": trainer.episode_losses,
    }
    save_json(training_log, str(Path("results") / "training_log.json"))

    # Plot learning curves
    plot_learning_curves(
        trainer.episode_rewards,
        trainer.episode_losses,
        "results",
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Episodes: {training_metrics['total_episodes']}")
    print(f"Avg reward: {training_metrics['avg_reward']:.4f}")
    print(f"Avg reward (last 50): {training_metrics['avg_reward_last_50']:.4f}")
    print(f"Policy saved: {policy_path}")


if __name__ == "__main__":
    main()
