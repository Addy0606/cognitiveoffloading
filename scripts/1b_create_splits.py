"""
Script 1b: Create Train/Validation/Test Splits

Splits labeled data into 300/100/100 (train/val/test) with:
- Dialogue-aware splitting: turns from the same dialogue stay together
- Stratified class balance
- Leakage prevention
- Statistics reporting

Usage:
    python scripts/1b_create_splits.py --input data/labeled_500.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import save_json, setup_logging, RANDOM_SEED

logger = logging.getLogger(__name__)


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = RANDOM_SEED,
    dialogue_aware: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split labeled data into train/val/test sets.

    If dialogue_aware=True, all turns from the same dialogue_id go into
    the same split to prevent data leakage.

    Args:
        df: Labeled DataFrame with 'label' and 'dialogue_id' columns.
        train_ratio: Fraction for training (default 0.6 → 300 of 500).
        val_ratio: Fraction for validation (default 0.2 → 100 of 500).
        test_ratio: Fraction for test (default 0.2 → 100 of 500).
        seed: Random seed.
        dialogue_aware: If True, keep dialogue turns together.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    rng = np.random.RandomState(seed)

    if dialogue_aware and "dialogue_id" in df.columns:
        return _dialogue_aware_split(df, train_ratio, val_ratio, rng)
    else:
        return _stratified_split(df, train_ratio, val_ratio, rng)


def _dialogue_aware_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.RandomState,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by dialogue_id to prevent leakage.
    Turns from the same dialogue never appear in different splits.
    """
    dialogue_ids = df["dialogue_id"].unique()
    rng.shuffle(dialogue_ids)

    n_total = len(dialogue_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_dids = set(dialogue_ids[:n_train])
    val_dids = set(dialogue_ids[n_train : n_train + n_val])
    test_dids = set(dialogue_ids[n_train + n_val :])

    train_df = df[df["dialogue_id"].isin(train_dids)].reset_index(drop=True)
    val_df = df[df["dialogue_id"].isin(val_dids)].reset_index(drop=True)
    test_df = df[df["dialogue_id"].isin(test_dids)].reset_index(drop=True)

    logger.info(
        f"Dialogue-aware split: {len(train_dids)} train dialogues, "
        f"{len(val_dids)} val dialogues, {len(test_dids)} test dialogues"
    )

    return train_df, val_df, test_df


def _stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.RandomState,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified random split maintaining class balance.
    Falls back when dialogue_id is unavailable.
    """
    from sklearn.model_selection import train_test_split

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df["label"],
        random_state=rng.randint(0, 2**31),
    )

    # Second split: val vs test (equal proportion of remaining)
    val_fraction = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_fraction,
        stratify=temp_df["label"],
        random_state=rng.randint(0, 2**31),
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def report_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Report statistics for each split."""
    stats = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["label"].value_counts().to_dict()
        n_dialogues = split_df["dialogue_id"].nunique() if "dialogue_id" in split_df.columns else "N/A"
        stats[name] = {
            "num_samples": len(split_df),
            "num_dialogues": n_dialogues,
            "learning_count": int(dist.get(0, 0)),
            "extraction_count": int(dist.get(1, 0)),
            "extraction_ratio": round(dist.get(1, 0) / max(len(split_df), 1), 3),
        }
        logger.info(
            f"{name}: {len(split_df)} samples "
            f"(learning={dist.get(0, 0)}, extraction={dist.get(1, 0)}, "
            f"dialogues={n_dialogues})"
        )

    # Check for leakage
    if "dialogue_id" in train_df.columns:
        train_ids = set(train_df["dialogue_id"])
        val_ids = set(val_df["dialogue_id"])
        test_ids = set(test_df["dialogue_id"])

        leakage_tv = train_ids & val_ids
        leakage_tt = train_ids & test_ids
        leakage_vt = val_ids & test_ids

        if leakage_tv or leakage_tt or leakage_vt:
            logger.warning(
                f"LEAKAGE DETECTED! "
                f"train∩val: {len(leakage_tv)}, "
                f"train∩test: {len(leakage_tt)}, "
                f"val∩test: {len(leakage_vt)}"
            )
            stats["leakage_detected"] = True
        else:
            logger.info("No dialogue-level leakage detected.")
            stats["leakage_detected"] = False

    return stats


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument(
        "--input", type=str, default="data/labeled_500.csv",
        help="Path to labeled data CSV",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory for split files",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.6,
        help="Training split ratio (default: 0.6)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--no-dialogue-aware", action="store_true",
        help="Disable dialogue-aware splitting",
    )

    args = parser.parse_args()
    setup_logging()

    # Load labeled data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Labeled data not found: {input_path}")
        logger.info("Run `python scripts/1_label_data.py` first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    if "label" not in df.columns:
        logger.error(f"'label' column not found in {input_path}")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} labeled samples from {input_path}")

    # Create splits
    train_df, val_df, test_df = create_splits(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        dialogue_aware=not args.no_dialogue_aware,
    )

    # Save splits
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_300.csv"
    val_path = out_dir / "val_100.csv"
    test_path = out_dir / "test_100.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Splits saved to {out_dir}/")

    # Report stats
    stats = report_split_stats(train_df, val_df, test_df)
    save_json(stats, str(out_dir / "split_statistics.json"))

    print(f"\nSplits created:")
    print(f"  Train: {train_path} ({len(train_df)} samples)")
    print(f"  Val:   {val_path} ({len(val_df)} samples)")
    print(f"  Test:  {test_path} ({len(test_df)} samples)")


if __name__ == "__main__":
    main()
