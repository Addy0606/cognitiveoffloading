"""
Script 3: Causal Analysis

Analyzes tutor→student patterns in dialogue data to identify
associations between tutor response types and subsequent student behavior.

IMPORTANT: This analysis identifies descriptive ASSOCIATIONS, not causal effects.
Historical observational data alone cannot establish that a tutor response type
CAUSED a change in student behavior. Confounders (topic difficulty, student
ability, time of day, etc.) are not controlled.

Usage:
    python scripts/3_causal_analysis.py --studychat_data data/studychat_full.csv --classifier models/classifier_v1 --output results/causal_analysis.csv
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
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import ExtractionClassifier
from src.utils import (
    categorize_tutor_response,
    load_dialogues,
    save_json,
    set_seed,
    setup_logging,
    ACTION_NAMES,
)

logger = logging.getLogger(__name__)


def reconstruct_sequences(
    dialogues: List[List[Dict]],
    classifier: ExtractionClassifier,
) -> pd.DataFrame:
    """
    Reconstruct student→tutor→student sequences and classify each message.

    For each valid triplet (student_msg, tutor_response, next_student_msg):
      - Classify student_msg → extraction_before
      - Categorize tutor_response → observed_action
      - Classify next_student_msg → extraction_after
      - Compute reward = extraction_before - extraction_after

    Args:
        dialogues: List of dialogues, each a list of turn dicts.
        classifier: Trained ExtractionClassifier.

    Returns:
        DataFrame with one row per valid triplet.
    """
    rows = []

    for dialogue in dialogues:
        student_turns = [t for t in dialogue if t.get("speaker") == "student"]
        tutor_turns = [t for t in dialogue if t.get("speaker") == "tutor"]

        # Build sequential pairs: student[i] → tutor[i] → student[i+1]
        sorted_turns = sorted(dialogue, key=lambda t: t.get("turn_num", 0))

        i = 0
        while i < len(sorted_turns) - 2:
            t1 = sorted_turns[i]
            t2 = sorted_turns[i + 1]
            t3 = sorted_turns[i + 2]

            # Look for student → tutor → student pattern
            if (t1.get("speaker") == "student" and
                t2.get("speaker") == "tutor" and
                t3.get("speaker") == "student"):

                msg_before = str(t1.get("message", ""))
                tutor_msg = str(t2.get("message", ""))
                msg_after = str(t3.get("message", ""))

                if msg_before.strip() and tutor_msg.strip() and msg_after.strip():
                    extraction_before = classifier.predict(msg_before)
                    extraction_after = classifier.predict(msg_after)
                    observed_action = categorize_tutor_response(tutor_msg)
                    reward = extraction_before - extraction_after

                    rows.append({
                        "dialogue_id": t1.get("dialogue_id", "unknown"),
                        "turn_before": t1.get("turn_num", 0),
                        "student_msg_before": msg_before[:200],
                        "tutor_response": tutor_msg[:200],
                        "student_msg_after": msg_after[:200],
                        "extraction_before": round(extraction_before, 4),
                        "extraction_after": round(extraction_after, 4),
                        "observed_action": observed_action,
                        "observed_action_name": ACTION_NAMES.get(observed_action, "UNKNOWN"),
                        "reward": round(reward, 4),
                    })
                i += 2  # Skip to next potential triplet
            else:
                i += 1

    df = pd.DataFrame(rows)
    logger.info(f"Reconstructed {len(df)} student→tutor→student sequences")
    return df


def analyze_action_outcomes(df: pd.DataFrame) -> Dict:
    """
    Analyze associations between tutor action types and student outcomes.

    Returns descriptive statistics, NOT causal claims.
    """
    results = {}

    for action_id, action_name in ACTION_NAMES.items():
        action_df = df[df["observed_action"] == action_id]
        n = len(action_df)

        if n == 0:
            results[action_name] = {
                "count": 0,
                "note": "No observed instances of this action type",
            }
            continue

        rewards = action_df["reward"].values
        results[action_name] = {
            "count": int(n),
            "mean_reward": round(float(np.mean(rewards)), 4),
            "std_reward": round(float(np.std(rewards)), 4),
            "median_reward": round(float(np.median(rewards)), 4),
            "positive_reward_pct": round(float((rewards > 0.05).mean() * 100), 1),
            "negative_reward_pct": round(float((rewards < -0.05).mean() * 100), 1),
            "mean_extraction_before": round(float(action_df["extraction_before"].mean()), 4),
            "mean_extraction_after": round(float(action_df["extraction_after"].mean()), 4),
        }

    return results


def generate_plots(df: pd.DataFrame, output_dir: str) -> None:
    """Generate causal analysis visualizations."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.warning("No data for plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tutor Response Type Analysis\n(Descriptive Associations — NOT Causal)",
                 fontsize=14, fontweight="bold")

    # 1. Mean reward by action type
    ax = axes[0, 0]
    action_stats = df.groupby("observed_action_name")["reward"].agg(["mean", "std", "count"])
    if not action_stats.empty:
        bars = ax.bar(action_stats.index, action_stats["mean"],
                      yerr=action_stats["std"] / np.sqrt(action_stats["count"]),
                      capsize=5, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"])
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Mean Reward (± SE)")
        ax.set_title("Mean Reward by Response Type")
        ax.tick_params(axis='x', rotation=15)

    # 2. Distribution of extraction scores before/after
    ax = axes[0, 1]
    if len(df) > 0:
        ax.hist(df["extraction_before"], bins=20, alpha=0.6, label="Before", color="#e74c3c")
        ax.hist(df["extraction_after"], bins=20, alpha=0.6, label="After", color="#2ecc71")
        ax.set_xlabel("Extraction Score")
        ax.set_ylabel("Count")
        ax.set_title("Extraction Score Distribution")
        ax.legend()

    # 3. Sample counts per action type
    ax = axes[1, 0]
    counts = df["observed_action_name"].value_counts()
    if not counts.empty:
        ax.bar(counts.index, counts.values, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][:len(counts)])
        ax.set_ylabel("Count")
        ax.set_title("Sample Counts by Response Type")
        ax.tick_params(axis='x', rotation=15)

    # 4. Reward distribution box plot
    ax = axes[1, 1]
    if len(df) > 0:
        action_groups = [group["reward"].values for _, group in df.groupby("observed_action_name")]
        action_labels = [name for name, _ in df.groupby("observed_action_name")]
        if action_groups:
            bp = ax.boxplot(action_groups, labels=action_labels, patch_artist=True)
            colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
            for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Reward")
            ax.set_title("Reward Distribution by Type")
            ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plot_path = out / "causal_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plots saved to {plot_path}")


def write_summary(analysis: Dict, df: pd.DataFrame, output_dir: str) -> None:
    """Write a human-readable analysis summary."""
    out = Path(output_dir)
    summary_path = out / "causal_analysis_summary.txt"

    with open(summary_path, "w") as f:
        f.write("TUTOR RESPONSE TYPE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("IMPORTANT DISCLAIMER:\n")
        f.write("This analysis describes ASSOCIATIONS between tutor response types\n")
        f.write("and subsequent student behavior. It does NOT establish causation.\n")
        f.write("Confounders (topic difficulty, student ability, etc.) are uncontrolled.\n\n")
        f.write(f"Total sequences analyzed: {len(df)}\n")
        f.write(f"Number of dialogues: {df['dialogue_id'].nunique() if len(df) > 0 else 0}\n\n")

        for action_name, stats in analysis.items():
            f.write(f"\n{action_name}:\n")
            f.write("-" * 40 + "\n")
            for key, val in stats.items():
                f.write(f"  {key}: {val}\n")

        f.write("\n\nInterpretation Notes:\n")
        f.write("- Positive reward = extraction score decreased (student shifted toward learning)\n")
        f.write("- Negative reward = extraction score increased (student shifted toward extraction)\n")
        f.write("- These patterns may reflect confounders, not causal effects\n")

    logger.info(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Causal analysis of tutor→student patterns")
    parser.add_argument("--studychat_data", type=str, default="data/studychat_full.csv")
    parser.add_argument("--classifier", type=str, default="models/classifier_v1")
    parser.add_argument("--output", type=str, default="results/causal_analysis.csv")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    setup_logging(log_file="logs/causal_analysis.log")
    set_seed(args.seed)

    # Load data
    data_path = Path(args.studychat_data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
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
            f"Using untrained model — results will not be meaningful."
        )
        classifier.initialize()

    # Reconstruct sequences
    logger.info("Reconstructing student→tutor→student sequences...")
    df = reconstruct_sequences(dialogues, classifier)

    if df.empty:
        logger.warning("No valid sequences found. Check data format.")
        return

    # Save raw analysis data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Analysis data saved to {output_path}")

    # Analyze
    analysis = analyze_action_outcomes(df)
    save_json(analysis, str(output_path.parent / "causal_analysis_results.json"))

    # Generate plots
    generate_plots(df, str(output_path.parent))

    # Write summary
    write_summary(analysis, df, str(output_path.parent))

    print("\n" + "=" * 60)
    print("CAUSAL ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nSequences analyzed: {len(df)}")
    print(f"\nAction type breakdown:")
    for action_name, stats in analysis.items():
        count = stats.get("count", 0)
        mean_r = stats.get("mean_reward", "N/A")
        print(f"  {action_name}: n={count}, mean_reward={mean_r}")
    print(f"\nOutputs: {output_path.parent}/")


if __name__ == "__main__":
    main()
