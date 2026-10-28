"""
Multi-Transformer Comparative Benchmark Study for Cognitive Offloading Detection.

Evaluates 5 Transformer Backbones:
1. DeBERTa-base (microsoft/deberta-base) - Current primary architecture
2. RoBERTa-base (roberta-base) - Optimized BERT baseline
3. BERT-base (bert-base-uncased) - Standard literature baseline
4. DistilBERT (distilbert-base-uncased) - Lightweight edge deployment
5. ELECTRA-base (google/electra-base-discriminator) - Sample-efficient pretraining

Metrics Evaluated:
- Classification Accuracy (%)
- Macro F1 (%)
- Extraction (Offloading) F1 (%)
- Learning (Reasoning) F1 (%)
- Inference Latency per message (ms)
- Total Parameters (M)
- Model Size on Disk (MB)
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.classifier import ExtractionClassifier
from src.utils import set_seed, setup_logging, RANDOM_SEED, save_json

logger = logging.getLogger(__name__)

MODELS_TO_BENCHMARK = [
    {
        "name": "DistilBERT",
        "checkpoint": "distilbert-base-uncased",
        "description": "Lightweight, 40% smaller and 60% faster edge transformer"
    },
    {
        "name": "BERT-base",
        "checkpoint": "bert-base-uncased",
        "description": "Foundational bidirectional transformer baseline"
    },
    {
        "name": "RoBERTa-base",
        "checkpoint": "roberta-base",
        "description": "Robustly optimized BERT with dynamic byte-level BPE"
    },
    {
        "name": "ELECTRA-base",
        "checkpoint": "google/electra-base-discriminator",
        "description": "Replaced token detection discriminator architecture"
    },
    {
        "name": "DeBERTa-base (Ours)",
        "checkpoint": "microsoft/deberta-base",
        "description": "Disentangled attention mechanism with enhanced mask decoding"
    }
]


def benchmark_model(
    model_info: Dict[str, str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42
) -> Dict[str, Any]:
    """Train and benchmark a single transformer architecture."""
    set_seed(seed)
    name = model_info["name"]
    checkpoint = model_info["checkpoint"]

    print("\n" + "=" * 70)
    print(f"BENCHMARKING: {name} ({checkpoint})")
    print("=" * 70)

    # Initialize model
    config = {
        "model_name": checkpoint,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": 256
    }
    classifier = ExtractionClassifier(config=config)
    classifier.initialize()

    # Calculate parameter count
    total_params = sum(p.numel() for p in classifier.model.parameters()) + sum(p.numel() for p in classifier.head.parameters())
    params_m = round(total_params / 1e6, 1)

    train_msgs = train_df["message"].astype(str).tolist()
    train_labels = train_df["label"].astype(int).tolist()
    val_msgs = val_df["message"].astype(str).tolist()
    val_labels = val_df["label"].astype(int).tolist()
    test_msgs = test_df["message"].astype(str).tolist()
    test_labels = test_df["label"].astype(int).tolist()

    # Train
    t0 = time.time()
    history = classifier.train_model(
        train_messages=train_msgs,
        train_labels=train_labels,
        val_messages=val_msgs,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    training_duration = round(time.time() - t0, 2)

    # Inference Latency Benchmark (100 test samples)
    latencies = []
    # Warmup
    for _ in range(5):
        _ = classifier.predict("Can you write the complete code for an AVL tree rotation?")

    for msg in test_msgs[:50]:
        t_start = time.perf_counter()
        _ = classifier.predict(msg)
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    avg_latency_ms = round(float(np.mean(latencies)), 2)
    p95_latency_ms = round(float(np.percentile(latencies, 95)), 2)

    # Evaluate on Test Set
    preds_prob = classifier.predict_batch(test_msgs)
    preds = [1 if p >= 0.5 else 0 for p in preds_prob]

    acc = float(accuracy_score(test_labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="macro", zero_division=0
    )

    result = {
        "model_name": name,
        "checkpoint": checkpoint,
        "description": model_info["description"],
        "parameters_m": params_m,
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(float(macro_f1) * 100, 2),
        "extraction_f1": round(float(f1[1]) * 100, 2),
        "extraction_precision": round(float(precision[1]) * 100, 2),
        "extraction_recall": round(float(recall[1]) * 100, 2),
        "learning_f1": round(float(f1[0]) * 100, 2),
        "inference_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "training_time_seconds": training_duration
    }

    print(f"Results for {name}:")
    print(f"  Accuracy:       {result['accuracy']}%")
    print(f"  Macro F1:       {result['macro_f1']}%")
    print(f"  Extraction F1:  {result['extraction_f1']}%")
    print(f"  Inference Time: {result['inference_latency_ms']} ms/msg")
    print(f"  Parameters:     {result['parameters_m']} M")

    # Clean GPU memory between models
    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-Transformer Comparison Benchmark")
    parser.add_argument("--train_data", type=str, default="data/train_300.csv")
    parser.add_argument("--val_data", type=str, default="data/val_100.csv")
    parser.add_argument("--test_data", type=str, default="data/test_100.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_file", type=str, default="results/transformer_comparison_results.json")
    args = parser.parse_args()

    setup_logging(log_file="logs/benchmark_transformers.log")

    train_df = pd.read_csv(args.train_data)
    val_df = pd.read_csv(args.val_data)
    test_df = pd.read_csv(args.test_data)

    results = []
    for model_info in MODELS_TO_BENCHMARK:
        try:
            res = benchmark_model(
                model_info=model_info,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            results.append(res)
        except Exception as e:
            logger.error(f"Failed benchmarking {model_info['name']}: {e}", exc_info=True)
            print(f"Error benchmarking {model_info['name']}: {e}")

    # Save results
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json({"models": results}, str(out_path))

    # Print Summary Markdown Table
    print("\n" + "=" * 80)
    print("TRANSFORMER COMPARISON BENCHMARK TABLE")
    print("=" * 80)
    print("| Model Architecture | Params (M) | Accuracy (%) | Macro F1 (%) | Extraction F1 (%) | Latency (ms) |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| **{r['model_name']}** | {r['parameters_m']}M | {r['accuracy']}% | {r['macro_f1']}% | {r['extraction_f1']}% | ~{r['inference_latency_ms']} ms |")
    print("=" * 80)


if __name__ == "__main__":
    main()
