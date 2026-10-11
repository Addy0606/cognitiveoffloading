"""
Script: Run All — Full pipeline orchestration

Runs the complete pipeline in sequence:
  0. Data preparation (synthetic)
  1. Labeling (synthetic)
  1b. Create splits
  2. Train classifier
  3. Causal analysis
  4. Train RL policy
  5. Evaluate

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --skip-training   # Skip long training steps
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logging

logger = logging.getLogger(__name__)

PYTHON = sys.executable


def run_step(name: str, cmd: list, skip: bool = False) -> bool:
    """Run a pipeline step and return success status."""
    print(f"\n{'=' * 60}")
    print(f"  STEP: {name}")
    print(f"{'=' * 60}")

    if skip:
        print(f"  SKIPPED (--skip-training)")
        return True

    print(f"  Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))

    if result.returncode != 0:
        print(f"\n  ❌ FAILED: {name} (exit code {result.returncode})")
        return False
    else:
        print(f"\n  ✓ COMPLETED: {name}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip classifier and RL training (use existing models)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rl-episodes", type=int, default=400)

    args = parser.parse_args()
    setup_logging()

    steps = [
        ("0. Prepare Data (Synthetic)",
         [PYTHON, "scripts/0_prepare_data.py", "--synthetic"],
         False),

        ("1. Generate Synthetic Labels",
         [PYTHON, "scripts/1_label_data.py", "--synthetic", "--output", "data/labeled_500.csv"],
         False),

        ("1b. Create Train/Val/Test Splits",
         [PYTHON, "scripts/1b_create_splits.py", "--input", "data/labeled_500.csv"],
         False),

        ("2. Train Classifier",
         [PYTHON, "scripts/2_train_classifier.py",
          "--train_data", "data/train_300.csv",
          "--val_data", "data/val_100.csv",
          "--test_data", "data/test_100.csv",
          "--output_dir", "models/classifier_v1",
          "--epochs", str(args.epochs),
          "--batch_size", "16"],
         args.skip_training),

        ("3. Causal Analysis",
         [PYTHON, "scripts/3_causal_analysis.py",
          "--studychat_data", "data/studychat_full.csv",
          "--classifier", "models/classifier_v1"],
         args.skip_training),

        ("4. Train RL Policy",
         [PYTHON, "scripts/4_train_rl_policy.py",
          "--studychat_data", "data/studychat_full.csv",
          "--classifier", "models/classifier_v1",
          "--episodes", str(args.rl_episodes)],
         args.skip_training),

        ("5. Evaluate",
         [PYTHON, "scripts/5_evaluate.py",
          "--classifier", "models/classifier_v1",
          "--policy", "models/policy_network_v1.pth",
          "--test_queries", "data/test_100.csv"],
         args.skip_training),
    ]

    print("\n" + "=" * 60)
    print("  COGNITIVE OFFLOADING — Full Pipeline")
    print("=" * 60)

    results = []
    for name, cmd, skip in steps:
        success = run_step(name, cmd, skip=skip)
        results.append((name, success))
        if not success:
            print(f"\n⚠ Pipeline stopped at: {name}")
            break

    print("\n\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "✓" if success else "❌"
        print(f"  {status}  {name}")
    print("=" * 60)

    all_passed = all(s for _, s in results)
    if all_passed:
        print("\n  🎉 All steps completed successfully!")
        print("\n  Next steps:")
        print("    python scripts/demo.py  # Interactive demo")
    else:
        print("\n  Some steps failed. Check logs above for details.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
