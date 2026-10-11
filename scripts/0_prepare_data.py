"""
Script 0: Data Preparation

Prepares the dialogue dataset for the RL tutoring system.

Since the StudyChat dataset does not have a verified public repository URL,
this script:
1. Checks for manually provided raw data files
2. If unavailable, generates a SYNTHETIC dataset for pipeline testing
3. Normalizes the data to the standard format
4. Reports statistics

Usage:
    python scripts/0_prepare_data.py --output data/studychat_full.csv
    python scripts/0_prepare_data.py --input raw_data.csv --output data/studychat_full.csv
    python scripts/0_prepare_data.py --synthetic --output data/studychat_full.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    generate_synthetic_dataset,
    load_csv,
    save_json,
    setup_logging,
    REQUIRED_COLUMNS,
)

logger = logging.getLogger(__name__)


def normalize_raw_data(input_path: str, output_path: str) -> None:
    """
    Normalize a raw dialogue CSV to the standard format.

    Expected columns: dialogue_id, turn_num, speaker, message
    Speaker values normalized to 'student' or 'tutor'.
    """
    df = load_csv(input_path)

    # Try to map column names
    column_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ("dialogue_id", "dialog_id", "conversation_id", "session_id"):
            column_map[col] = "dialogue_id"
        elif col_lower in ("turn_num", "turn_number", "turn", "turn_id"):
            column_map[col] = "turn_num"
        elif col_lower in ("speaker", "role", "sender", "user_type"):
            column_map[col] = "speaker"
        elif col_lower in ("message", "text", "content", "utterance"):
            column_map[col] = "message"
        elif col_lower in ("context", "topic", "week"):
            column_map[col] = "context"
        elif col_lower in ("timestamp", "time", "date"):
            column_map[col] = "timestamp"

    if column_map:
        df = df.rename(columns=column_map)

    # Check required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot normalize: missing columns {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Normalize speaker roles
    speaker_map = {
        "student": "student",
        "learner": "student",
        "user": "student",
        "human": "student",
        "tutor": "tutor",
        "assistant": "tutor",
        "ai": "tutor",
        "bot": "tutor",
        "teacher": "tutor",
        "system": "tutor",
    }
    df["speaker"] = df["speaker"].str.lower().str.strip().map(speaker_map)
    unmapped = df["speaker"].isna().sum()
    if unmapped > 0:
        logger.warning(f"{unmapped} rows have unmapped speaker roles. Dropping them.")
        df = df.dropna(subset=["speaker"])

    # Sort by dialogue and turn
    df = df.sort_values(["dialogue_id", "turn_num"]).reset_index(drop=True)

    # Drop rows with empty messages
    df = df[df["message"].notna() & (df["message"].str.strip() != "")]
    df = df.reset_index(drop=True)

    # Add context column if missing
    if "context" not in df.columns:
        df["context"] = ""

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Report stats
    num_dialogues = df["dialogue_id"].nunique()
    num_turns = len(df)
    num_student = (df["speaker"] == "student").sum()
    num_tutor = (df["speaker"] == "tutor").sum()

    stats = {
        "total_turns": int(num_turns),
        "total_dialogues": int(num_dialogues),
        "student_turns": int(num_student),
        "tutor_turns": int(num_tutor),
        "avg_turns_per_dialogue": round(num_turns / max(num_dialogues, 1), 1),
    }

    logger.info(f"Normalized data saved to {output_path}")
    logger.info(f"Statistics: {stats}")

    stats_path = str(Path(output_path).parent / "data_statistics.json")
    save_json(stats, stats_path)


def prepare_synthetic(output_path: str, num_dialogues: int = 50) -> None:
    """
    Generate a synthetic dataset for pipeline testing.

    WARNING: This data is SYNTHETIC and is explicitly marked as such.
    It should NEVER be presented as real StudyChat data.
    """
    logger.warning(
        "=" * 60 + "\n"
        "GENERATING SYNTHETIC DATA FOR PIPELINE TESTING\n"
        "This is NOT real StudyChat data. Do not use for research claims.\n"
        + "=" * 60
    )

    df = generate_synthetic_dataset(num_dialogues=num_dialogues)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    stats = {
        "data_type": "SYNTHETIC",
        "total_turns": len(df),
        "total_dialogues": df["dialogue_id"].nunique(),
        "student_turns": int((df["speaker"] == "student").sum()),
        "tutor_turns": int((df["speaker"] == "tutor").sum()),
        "warning": "This is synthetic data for pipeline testing only.",
    }

    stats_path = str(Path(output_path).parent / "data_statistics.json")
    save_json(stats, stats_path)

    logger.info(f"Synthetic data saved to {output_path}")
    logger.info(f"Statistics: {stats}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dialogue data")
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to raw input CSV (if available)",
    )
    parser.add_argument(
        "--output", type=str, default="data/studychat_full.csv",
        help="Path for output normalized CSV",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic data instead of using real data",
    )
    parser.add_argument(
        "--num-dialogues", type=int, default=50,
        help="Number of synthetic dialogues to generate (default: 50)",
    )

    args = parser.parse_args()
    setup_logging()

    if args.input:
        # Normalize provided raw data
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        normalize_raw_data(str(input_path), args.output)
    elif args.synthetic:
        # Generate synthetic data
        prepare_synthetic(args.output, args.num_dialogues)
    else:
        # Check for existing data, otherwise generate synthetic
        output_path = Path(args.output)
        if output_path.exists():
            logger.info(f"Data already exists at {output_path}. Validating...")
            df = load_csv(str(output_path), required_columns=REQUIRED_COLUMNS)
            logger.info(
                f"Valid: {len(df)} turns, {df['dialogue_id'].nunique()} dialogues"
            )
        else:
            logger.info(
                "No input data provided and no existing data found.\n"
                "The StudyChat dataset does not have a verified public download URL.\n"
                "\n"
                "Options:\n"
                "  1. Provide your own data: python scripts/0_prepare_data.py --input <path>\n"
                "  2. Generate synthetic data: python scripts/0_prepare_data.py --synthetic\n"
                "\n"
                "Generating synthetic data for pipeline testing..."
            )
            prepare_synthetic(args.output)


if __name__ == "__main__":
    main()
