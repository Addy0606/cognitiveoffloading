"""
Interactive Demo: Adaptive Tutoring System

Lets you type student messages and see the full pipeline in action:
  1. Extraction score (classifier)
  2. Engagement score
  3. RL policy Q-values and selected action
  4. Tutoring response

Usage:
    python scripts/demo.py
    python scripts/demo.py --classifier models/classifier_v1 --policy models/policy_network_v1.pth
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adaptive_tutor import AdaptiveTutor
from src.classifier import ExtractionClassifier
from src.rl_policy_network import RLTrainer
from src.utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


def print_banner():
    print("\n" + "=" * 65)
    print("  🎓  ADAPTIVE TUTORING SYSTEM — Interactive Demo")
    print("=" * 65)
    print()
    print("  This system detects if a student is seeking direct answers")
    print("  (extraction-mode) and uses RL to select an intervention.")
    print()
    print("  Actions: REDIRECT | SCAFFOLD | SOCRATIC | DELAY")
    print()
    print("  Commands:")
    print("    type a message  → get adaptive response")
    print("    /baseline       → compare with baseline response")
    print("    /diagnostics    → show tutor diagnostics")
    print("    /reset          → reset conversation")
    print("    /quit           → exit")
    print("=" * 65)


def format_q_values(q_values, action_names):
    parts = []
    best = max(range(len(q_values)), key=lambda i: q_values[i])
    for i, (name, q) in enumerate(zip(action_names, q_values)):
        marker = " ◄" if i == best else ""
        parts.append(f"    {name:12s}: {q:+.4f}{marker}")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Interactive tutoring demo")
    parser.add_argument("--classifier", type=str, default="models/classifier_v1")
    parser.add_argument("--policy", type=str, default="models/policy_network_v1.pth")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    setup_logging(log_level="WARNING")
    set_seed(args.seed)

    # Initialize tutor
    tutor = AdaptiveTutor()

    # Load models if available
    classifier_path = Path(args.classifier)
    policy_path = Path(args.policy)

    if classifier_path.exists():
        tutor.classifier.load(str(classifier_path))
        print(f"✓ Classifier loaded from {classifier_path}")
    else:
        print(f"⚠ Classifier not found at {classifier_path}. Using untrained model.")
        tutor.classifier.initialize()

    if policy_path.exists():
        tutor.rl_trainer.load(str(policy_path))
        print(f"✓ Policy loaded from {policy_path}")
    else:
        print(f"⚠ Policy not found at {policy_path}. Using random policy.")

    print(f"✓ LLM provider: {tutor.llm_provider}")

    print_banner()

    action_names = ["REDIRECT", "SCAFFOLD", "SOCRATIC", "DELAY"]

    while True:
        try:
            user_input = input("\n📝 Student > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            tutor.reset_conversation()
            print("🔄 Conversation reset.")
            continue

        if user_input.lower() == "/diagnostics":
            diag = tutor.get_diagnostics()
            print("\n📊 Diagnostics:")
            for k, v in diag.items():
                print(f"    {k}: {v}")
            continue

        if user_input.lower() == "/baseline":
            print("Enter a message to compare baseline vs adaptive:")
            msg = input("📝 Student > ").strip()
            if not msg:
                continue

            baseline = tutor.infer_baseline(msg)
            tutor.reset_conversation()
            adaptive = tutor.infer(msg, use_policy=True)

            print("\n" + "-" * 60)
            print("BASELINE RESPONSE:")
            print("-" * 60)
            print(baseline["tutor_response"])
            print("\n" + "-" * 60)
            print(f"ADAPTIVE RESPONSE (Action: {adaptive['action_taken']}):")
            print("-" * 60)
            print(adaptive["tutor_response"])
            continue

        # Main inference
        result = tutor.infer(user_input)

        print()
        print(f"  📊 Extraction Score: {result['extraction_score']:.3f}", end="")
        if result['extraction_score'] > 0.7:
            print(" (HIGH — likely seeking direct answer)")
        elif result['extraction_score'] > 0.4:
            print(" (MEDIUM)")
        else:
            print(" (LOW — likely trying to learn)")

        print(f"  📈 Engagement Score: {result['engagement_score']:.3f}")

        print(f"\n  🎯 Q-Values:")
        print(format_q_values(result["q_values"], action_names))

        print(f"\n  ▶  Selected Action: {result['action_taken']}")

        print(f"\n  🤖 Tutor Response:")
        print(f"  {'─' * 55}")
        for line in result["tutor_response"].split("\n"):
            print(f"  {line}")
        print(f"  {'─' * 55}")

        if result["is_fallback"]:
            print(f"  ⚠ (Fallback template — set OPENAI_API_KEY for real LLM)")


if __name__ == "__main__":
    main()
