"""
Script 2: Train DeBERTa Classifier

Fine-tunes DeBERTa-base for extraction/learning classification.

Usage:
    python scripts/2_train_classifier.py --train_data data/train_300.csv --val_data data/val_100.csv --test_data data/test_100.csv
    python scripts/2_train_classifier.py --epochs 3 --batch_size 16
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import ExtractionClassifier
from src.utils import save_json, set_seed, setup_logging, RANDOM_SEED

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train extraction/learning classifier")
    parser.add_argument("--train_data", type=str, default="data/train_300.csv")
    parser.add_argument("--val_data", type=str, default="data/val_100.csv")
    parser.add_argument("--test_data", type=str, default="data/test_100.csv")
    parser.add_argument("--output_dir", type=str, default="models/classifier_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    args = parser.parse_args()
    setup_logging(log_file="logs/train_classifier.log")
    set_seed(args.seed)

    # Validate inputs
    for path_name, path_val in [("train_data", args.train_data), ("val_data", args.val_data)]:
        if not Path(path_val).exists():
            logger.error(f"{path_name} not found: {path_val}")
            logger.info("Run data preparation scripts first:")
            logger.info("  python scripts/0_prepare_data.py --synthetic")
            logger.info("  python scripts/1_label_data.py --synthetic")
            logger.info("  python scripts/1b_create_splits.py")
            sys.exit(1)

    # Load data
    train_df = pd.read_csv(args.train_data)
    val_df = pd.read_csv(args.val_data)

    train_messages = train_df["message"].astype(str).tolist()
    train_labels = train_df["label"].astype(int).tolist()
    val_messages = val_df["message"].astype(str).tolist()
    val_labels = val_df["label"].astype(int).tolist()

    logger.info(f"Training data: {len(train_messages)} samples")
    logger.info(f"Validation data: {len(val_messages)} samples")
    logger.info(f"Train label distribution: {pd.Series(train_labels).value_counts().to_dict()}")
    logger.info(f"Val label distribution: {pd.Series(val_labels).value_counts().to_dict()}")

    # Initialize and train classifier
    classifier = ExtractionClassifier(config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    })

    logger.info("Starting classifier training...")
    history = classifier.train_model(
        train_messages=train_messages,
        train_labels=train_labels,
        val_messages=val_messages,
        val_labels=val_labels,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    # Save model
    classifier.save(args.output_dir)
    logger.info(f"Classifier saved to {args.output_dir}")

    # Save training history
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(history, str(results_dir / "training_history.json"))

    # Evaluate on test set if available
    if Path(args.test_data).exists():
        test_df = pd.read_csv(args.test_data)
        test_messages = test_df["message"].astype(str).tolist()
        test_labels = test_df["label"].astype(int).tolist()

        logger.info(f"\nEvaluating on test set ({len(test_messages)} samples)...")
        metrics = classifier.evaluate(test_messages, test_labels)

        # Save metrics
        save_json(metrics, str(results_dir / "classifier_metrics.json"))

        # Print report
        print("\n" + "=" * 60)
        print("CLASSIFIER EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nTest Set: {len(test_messages)} samples")
        print(f"  Learning (0): {metrics['class_distribution']['learning']}")
        print(f"  Extraction (1): {metrics['class_distribution']['extraction']}")
        print(f"\nResults:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        cm = metrics["confusion_matrix"]
        print(f"              Pred Learning  Pred Extraction")
        print(f"  True Learning      {cm[0][0]:4d}          {cm[0][1]:4d}")
        print(f"  True Extraction    {cm[1][0]:4d}          {cm[1][1]:4d}")
        print(f"\n{metrics['classification_report']}")

        # Write readable report
        report_path = results_dir / "classifier_report.txt"
        with open(report_path, "w") as f:
            f.write("CLASSIFIER EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {classifier.config['model_name']}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Learning Rate: {args.learning_rate}\n")
            f.write(f"Random Seed: {args.seed}\n\n")
            f.write(f"Training samples: {len(train_messages)}\n")
            f.write(f"Validation samples: {len(val_messages)}\n")
            f.write(f"Test samples: {len(test_messages)}\n\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1:        {metrics['f1']:.4f}\n\n")
            f.write(f"Classification Report:\n{metrics['classification_report']}\n")

            # Add limitations note
            f.write("\n" + "-" * 60 + "\n")
            f.write("LIMITATIONS:\n")
            f.write(f"- Trained on only {len(train_messages)} examples\n")
            f.write("- Small dataset risks overfitting\n")
            f.write("- Generalization to unseen domains is unverified\n")
            if any("synthetic" in str(n).lower() for n in train_df.get("notes", [])):
                f.write("- WARNING: Trained on SYNTHETIC data, not real student messages\n")

        logger.info(f"Report saved to {report_path}")
    else:
        logger.warning(f"Test data not found: {args.test_data}. Skipping evaluation.")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
