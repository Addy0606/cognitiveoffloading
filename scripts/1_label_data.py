"""
Script 1: Interactive Data Labeling

Interactive terminal-based labeling interface for annotating student messages
as extraction-oriented (1) or learning-oriented (0).

Features:
- Shows student message with context (surrounding turns)
- Supports saving progress and resuming interrupted annotation
- Records annotator confidence and optional notes
- Generates labeled output CSV

Usage:
    python scripts/1_label_data.py --input data/studychat_full.csv --output data/labeled_500.csv
    python scripts/1_label_data.py --input data/studychat_full.csv --output data/labeled_500.csv --num-samples 500
    python scripts/1_label_data.py --synthetic --output data/labeled_500.csv
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    generate_synthetic_labeled,
    load_csv,
    setup_logging,
    RANDOM_SEED,
    REQUIRED_COLUMNS,
)

logger = logging.getLogger(__name__)


def sample_student_turns(
    df: pd.DataFrame,
    num_samples: int = 500,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Stratified sampling of student turns for annotation.

    Samples proportionally from each dialogue to ensure coverage.

    Args:
        df: Full dialogue DataFrame.
        num_samples: Number of student turns to sample.
        seed: Random seed.

    Returns:
        DataFrame of sampled student turns.
    """
    student_turns = df[df["speaker"] == "student"].copy()

    if len(student_turns) < num_samples:
        logger.warning(
            f"Only {len(student_turns)} student turns available, "
            f"requested {num_samples}. Using all."
        )
        return student_turns

    # Stratified sampling by dialogue
    dialogue_ids = student_turns["dialogue_id"].unique()
    samples_per_dialogue = max(1, num_samples // len(dialogue_ids))

    sampled = []
    rng = random.Random(seed)

    for did in dialogue_ids:
        dialogue_turns = student_turns[student_turns["dialogue_id"] == did]
        n = min(samples_per_dialogue, len(dialogue_turns))
        indices = rng.sample(list(dialogue_turns.index), n)
        sampled.extend(indices)

    # If we don't have enough, sample more randomly
    remaining = num_samples - len(sampled)
    if remaining > 0:
        available = [
            i for i in student_turns.index if i not in sampled
        ]
        if available:
            extra = rng.sample(available, min(remaining, len(available)))
            sampled.extend(extra)

    # Shuffle and truncate
    rng.shuffle(sampled)
    sampled = sampled[:num_samples]

    return student_turns.loc[sampled].reset_index(drop=True)


def interactive_labeling(
    samples: pd.DataFrame,
    full_df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """
    Interactive terminal-based labeling of student messages.

    Labels:
        0 = Learning-oriented (trying to understand)
        1 = Extraction-oriented (trying to get answer)

    Supports:
        - Saving progress with 'save' command
        - Skipping with 's' command
        - Showing context with 'c' command
        - Quitting with 'q' command (auto-saves)

    Args:
        samples: DataFrame of student turns to label.
        full_df: Full dialogue DataFrame (for context).
        output_path: Path to save labeled data.

    Returns:
        DataFrame with labels added.
    """
    # Check for existing progress
    progress_path = Path(output_path).with_suffix(".progress.json")
    labeled_data = []
    start_idx = 0

    if progress_path.exists():
        try:
            with open(progress_path) as f:
                progress = json.load(f)
            labeled_data = progress.get("labeled", [])
            start_idx = len(labeled_data)
            logger.info(f"Resuming from {start_idx}/{len(samples)} labels")
        except Exception as e:
            logger.warning(f"Could not load progress: {e}")

    print("\n" + "=" * 60)
    print("LABELING INTERFACE")
    print("=" * 60)
    print("\nLabeling Rubric:")
    print("  0 = LEARNING-ORIENTED")
    print("      - Explains what they tried")
    print("      - Asks conceptual questions")
    print("      - Shows reasoning")
    print("")
    print("  1 = EXTRACTION-ORIENTED")
    print("      - Asks directly for answer/code")
    print("      - Minimal context or effort")
    print("      - 'Just give me...' / 'Tell me...'")
    print("")
    print("Commands: 0/1 (label), c (context), s (skip), save, q (quit)")
    print("=" * 60 + "\n")

    for idx in range(start_idx, len(samples)):
        row = samples.iloc[idx]
        dialogue_id = row.get("dialogue_id", "unknown")
        turn_num = row.get("turn_num", 0)
        message = str(row.get("message", ""))

        print(f"\n--- Sample {idx + 1}/{len(samples)} ---")
        print(f"Dialogue: {dialogue_id} | Turn: {turn_num}")
        print(f"Message: {message}")
        print()

        while True:
            user_input = input("Label (0=learning, 1=extraction, c=context, s=skip, q=quit): ").strip().lower()

            if user_input == "q":
                _save_progress(progress_path, labeled_data)
                print(f"\nProgress saved. {len(labeled_data)} labels recorded.")
                return _to_dataframe(labeled_data)

            elif user_input == "save":
                _save_progress(progress_path, labeled_data)
                print(f"Progress saved ({len(labeled_data)} labels).")
                continue

            elif user_input == "s":
                print("Skipped.")
                break

            elif user_input == "c":
                _show_context(full_df, dialogue_id, turn_num)
                continue

            elif user_input in ("0", "1"):
                label = int(user_input)

                # Ask for confidence
                conf = input("Confidence (h=high, m=medium, l=low) [h]: ").strip().lower()
                confidence = {"h": "high", "m": "medium", "l": "low"}.get(conf, "high")

                # Optional notes
                notes = input("Notes (optional, press Enter to skip): ").strip()

                labeled_data.append({
                    "dialogue_id": dialogue_id,
                    "turn_num": int(turn_num),
                    "message": message,
                    "label": label,
                    "confidence": confidence,
                    "notes": notes,
                })

                label_name = "LEARNING" if label == 0 else "EXTRACTION"
                print(f"  → Labeled as {label_name} ({confidence})")
                break

            else:
                print("Invalid input. Enter 0, 1, c, s, save, or q.")

        # Auto-save every 10 labels
        if len(labeled_data) % 10 == 0 and len(labeled_data) > 0:
            _save_progress(progress_path, labeled_data)

    # Save final
    _save_progress(progress_path, labeled_data)
    df_labeled = _to_dataframe(labeled_data)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(output_path, index=False)
    print(f"\nLabeling complete! {len(labeled_data)} labels saved to {output_path}")

    # Report distribution
    if len(df_labeled) > 0:
        dist = df_labeled["label"].value_counts()
        print(f"\nClass distribution:")
        print(f"  Learning (0): {dist.get(0, 0)}")
        print(f"  Extraction (1): {dist.get(1, 0)}")

    return df_labeled


def _show_context(df: pd.DataFrame, dialogue_id: str, turn_num: int) -> None:
    """Show surrounding turns for context."""
    dialogue = df[df["dialogue_id"] == dialogue_id].sort_values("turn_num")
    if dialogue.empty:
        print("  (No context available)")
        return

    print("\n  --- Context ---")
    for _, turn in dialogue.iterrows():
        marker = " >>>" if turn["turn_num"] == turn_num else "    "
        speaker = turn.get("speaker", "?")
        msg = str(turn.get("message", ""))[:100]
        print(f"  {marker} [{speaker}] {msg}")
    print("  --- End context ---\n")


def _save_progress(path: Path, labeled_data: list) -> None:
    """Save labeling progress to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"labeled": labeled_data}, f, indent=2)


def _to_dataframe(labeled_data: list) -> pd.DataFrame:
    """Convert labeled data list to DataFrame."""
    if not labeled_data:
        return pd.DataFrame(columns=["dialogue_id", "turn_num", "message", "label", "confidence", "notes"])
    return pd.DataFrame(labeled_data)


def main():
    parser = argparse.ArgumentParser(description="Interactive data labeling")
    parser.add_argument(
        "--input", type=str, default="data/studychat_full.csv",
        help="Path to full dialogue CSV",
    )
    parser.add_argument(
        "--output", type=str, default="data/labeled_500.csv",
        help="Path for labeled output CSV",
    )
    parser.add_argument(
        "--num-samples", type=int, default=500,
        help="Number of samples to label",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic labeled data instead of interactive labeling",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed",
    )

    args = parser.parse_args()
    setup_logging()

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        logger.warning(
            "Generating SYNTHETIC labeled data. These labels are assigned by "
            "construction, NOT by human annotation."
        )
        df = generate_synthetic_labeled(
            num_samples=args.num_samples,
            seed=args.seed,
        )
        df.to_csv(args.output, index=False)
        print(f"\nSynthetic labeled data saved to {args.output}")
        print(f"  Total: {len(df)} samples")
        print(f"  Learning (0): {(df['label'] == 0).sum()}")
        print(f"  Extraction (1): {(df['label'] == 1).sum()}")
        return

    # Interactive labeling
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run `python scripts/0_prepare_data.py` first, or use --synthetic")
        sys.exit(1)

    full_df = load_csv(str(input_path), required_columns=REQUIRED_COLUMNS)
    samples = sample_student_turns(full_df, num_samples=args.num_samples, seed=args.seed)
    interactive_labeling(samples, full_df, args.output)


if __name__ == "__main__":
    main()
