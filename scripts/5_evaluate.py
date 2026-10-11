"""
Script 5: Evaluation

Evaluates classifier and adaptive tutor performance:
1. Classifier metrics (accuracy, precision, recall, F1, confusion matrix)
2. Baseline vs. Adaptive tutor comparison on test queries

Usage:
    python scripts/5_evaluate.py --classifier models/classifier_v1 --policy models/policy_network_v1.pth --test_queries data/test_100.csv --output results/evaluation_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adaptive_tutor import AdaptiveTutor
from src.classifier import ExtractionClassifier
from src.utils import save_json, set_seed, setup_logging

logger = logging.getLogger(__name__)


def evaluate_classifier(
    classifier: ExtractionClassifier,
    test_path: str,
) -> Dict:
    """Evaluate classifier on held-out test set."""
    df = pd.read_csv(test_path)
    messages = df["message"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    logger.info(f"Evaluating classifier on {len(messages)} test samples...")
    metrics = classifier.evaluate(messages, labels)
    return metrics


def select_evaluation_queries(
    test_path: str,
    classifier: ExtractionClassifier,
    num_queries: int = 20,
    extraction_threshold: float = 0.7,
) -> List[Dict]:
    """
    Select test queries for baseline vs. adaptive comparison.

    Prioritizes extraction-mode queries (score > threshold) since
    those are where the adaptive tutor should differ most from baseline.
    """
    df = pd.read_csv(test_path)
    messages = df["message"].astype(str).tolist()

    queries = []
    for idx, msg in enumerate(messages):
        score = classifier.predict(msg)
        queries.append({
            "index": idx,
            "message": msg,
            "extraction_score": score,
            "label": int(df.iloc[idx].get("label", -1)),
        })

    # Sort by extraction score (highest first)
    queries.sort(key=lambda q: q["extraction_score"], reverse=True)

    # Select top queries above threshold, pad with others if needed
    selected = [q for q in queries if q["extraction_score"] >= extraction_threshold]
    if len(selected) < num_queries:
        remaining = [q for q in queries if q["extraction_score"] < extraction_threshold]
        selected.extend(remaining[:num_queries - len(selected)])

    return selected[:num_queries]


def compare_responses(
    tutor: AdaptiveTutor,
    queries: List[Dict],
) -> List[Dict]:
    """
    Generate baseline and adaptive responses for each query.

    For each query, generates both a baseline (no policy) and adaptive
    (RL policy) response and records all diagnostic metadata.
    """
    comparisons = []

    for i, query in enumerate(queries):
        message = query["message"]
        tutor.reset_conversation()

        logger.info(f"Evaluating query {i + 1}/{len(queries)}: {message[:60]}...")

        # Generate baseline response
        baseline = tutor.infer_baseline(message)

        # Generate adaptive response
        tutor.reset_conversation()
        adaptive = tutor.infer(message, use_policy=True)

        comparisons.append({
            "query_index": i + 1,
            "message": message,
            "true_label": query.get("label", -1),
            "extraction_score": adaptive["extraction_score"],
            "engagement_score": adaptive["engagement_score"],
            "action_taken": adaptive["action_taken"],
            "action_id": adaptive["action_id"],
            "q_values": adaptive["q_values"],
            "baseline_response": baseline["tutor_response"],
            "adaptive_response": adaptive["tutor_response"],
            "is_fallback": adaptive["is_fallback"],
            "llm_provider": adaptive["llm_provider"],
        })

    return comparisons


def score_comparison(comparison: Dict) -> Dict:
    """
    Auto-score a comparison using heuristic rubric.

    NOTE: This is AUTOMATED scoring, not human evaluation.
    Automated scoring is a rough proxy and should not be presented
    as equivalent to independent human assessment.

    Rubric (1-5):
        5 = Redirects effectively, pedagogically sound
        4 = Addresses help-seeking, minor issues
        3 = Attempts to help, unclear on offloading
        2 = Gives answer anyway or dismissive
        1 = Enables offloading or counterproductive
    """
    extraction = comparison["extraction_score"]
    baseline = comparison["baseline_response"].lower()
    adaptive = comparison["adaptive_response"].lower()

    # Score baseline
    baseline_score = _auto_score(baseline, extraction)
    # Score adaptive
    adaptive_score = _auto_score(adaptive, extraction)

    return {
        "baseline_score": baseline_score,
        "adaptive_score": adaptive_score,
        "winner": (
            "adaptive" if adaptive_score > baseline_score
            else "baseline" if baseline_score > adaptive_score
            else "tie"
        ),
    }


def _auto_score(response: str, extraction_score: float) -> int:
    """
    Heuristic auto-scoring of a tutor response.

    This is a rough approximation. It checks for:
    - Answer leaking (giving solution directly)
    - Redirection (asking what they tried)
    - Scaffolding (providing hints)
    - Dismissiveness
    """
    # Check for direct answer leaking
    leak_patterns = [
        "here's the solution", "here's the code", "the answer is",
        "here's a complete", "def ", "return ", "for i in range",
    ]
    has_leak = any(p in response for p in leak_patterns)

    # Check for pedagogical redirection
    redirect_patterns = [
        "what have you tried", "what part", "can you explain",
        "think about", "let's break", "try", "consider",
        "what would happen", "step by step",
    ]
    has_redirect = any(p in response for p in redirect_patterns)

    # Check for question asking (Socratic)
    has_questions = response.count("?") >= 2

    if extraction_score > 0.6:
        # For extraction-mode queries, redirection is better
        if has_leak and not has_redirect:
            return 1  # Leaks answer
        elif has_leak and has_redirect:
            return 2  # Mixed
        elif has_redirect and has_questions:
            return 5  # Great pedagogical response
        elif has_redirect:
            return 4  # Good redirection
        elif has_questions:
            return 3  # Some effort
        else:
            return 2
    else:
        # For learning-mode queries, direct help is acceptable
        if has_redirect or has_questions:
            return 4
        else:
            return 3


def generate_comparison_report(
    comparisons: List[Dict],
    scores: List[Dict],
    output_dir: str,
) -> None:
    """Generate a detailed comparison report with side-by-side examples."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    wins = sum(1 for s in scores if s["winner"] == "adaptive")
    ties = sum(1 for s in scores if s["winner"] == "tie")
    losses = sum(1 for s in scores if s["winner"] == "baseline")
    total = len(scores)

    baseline_scores = [s["baseline_score"] for s in scores]
    adaptive_scores = [s["adaptive_score"] for s in scores]

    # Statistical test
    from scipy import stats as scipy_stats
    if len(baseline_scores) > 1 and np.std(baseline_scores) > 0 and np.std(adaptive_scores) > 0:
        t_stat, p_value = scipy_stats.ttest_rel(adaptive_scores, baseline_scores)
        effect_size = (np.mean(adaptive_scores) - np.mean(baseline_scores)) / np.std(
            np.array(adaptive_scores) - np.array(baseline_scores)
        )
    else:
        t_stat, p_value, effect_size = 0.0, 1.0, 0.0

    # Write markdown report
    report_path = out / "example_comparisons.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Baseline vs. Adaptive Tutor: Side-by-Side Comparison\n\n")
        f.write(f"**Evaluation Method:** Automated heuristic scoring (NOT human evaluation)\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total Queries | {total} |\n")
        f.write(f"| Adaptive Wins | {wins} ({wins/max(total,1)*100:.0f}%) |\n")
        f.write(f"| Ties | {ties} ({ties/max(total,1)*100:.0f}%) |\n")
        f.write(f"| Baseline Wins | {losses} ({losses/max(total,1)*100:.0f}%) |\n")
        f.write(f"| Avg Baseline Score | {np.mean(baseline_scores):.2f}/5 |\n")
        f.write(f"| Avg Adaptive Score | {np.mean(adaptive_scores):.2f}/5 |\n")
        f.write(f"| Paired t-test | t={t_stat:.3f}, p={p_value:.4f} |\n")
        f.write(f"| Effect Size (Cohen's d) | {effect_size:.3f} |\n\n")

        if p_value < 0.05:
            f.write(f"> **Note:** p < 0.05 with automated scoring. ")
            f.write(f"This does NOT confirm real-world effectiveness. ")
            f.write(f"Human evaluation is needed for valid conclusions.\n\n")
        else:
            f.write(f"> **Note:** p ≥ 0.05 — no significant difference detected ")
            f.write(f"with automated scoring.\n\n")

        # Show top 5 examples
        f.write(f"## Example Comparisons\n\n")
        num_examples = min(5, len(comparisons))
        for i in range(num_examples):
            comp = comparisons[i]
            sc = scores[i]
            f.write(f"### Example {i + 1}\n\n")
            f.write(f"**Query:** {comp['message']}\n\n")
            f.write(f"- Extraction Score: {comp['extraction_score']:.3f}\n")
            f.write(f"- Engagement Score: {comp['engagement_score']:.3f}\n")
            f.write(f"- Action: {comp['action_taken']}\n")
            f.write(f"- Q-values: {[f'{q:.3f}' for q in comp['q_values']]}\n\n")
            f.write(f"**Baseline Response** (Score: {sc['baseline_score']}/5):\n\n")
            f.write(f"> {comp['baseline_response'][:500]}\n\n")
            f.write(f"**Adaptive Response** (Score: {sc['adaptive_score']}/5):\n\n")
            f.write(f"> {comp['adaptive_response'][:500]}\n\n")
            f.write(f"**Winner:** {sc['winner'].upper()}\n\n")
            f.write(f"---\n\n")

    logger.info(f"Comparison report saved to {report_path}")

    # Save evaluation results JSON
    results = {
        "evaluation_type": "automated_heuristic",
        "warning": "Automated scoring — not equivalent to human evaluation",
        "num_queries": total,
        "adaptive_wins": wins,
        "ties": ties,
        "baseline_wins": losses,
        "avg_baseline_score": round(float(np.mean(baseline_scores)), 3),
        "avg_adaptive_score": round(float(np.mean(adaptive_scores)), 3),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "effect_size": round(float(effect_size), 4),
        "per_query_scores": [
            {
                "query": comp["message"][:100],
                "extraction_score": comp["extraction_score"],
                "action": comp["action_taken"],
                "baseline_score": sc["baseline_score"],
                "adaptive_score": sc["adaptive_score"],
                "winner": sc["winner"],
            }
            for comp, sc in zip(comparisons, scores)
        ],
    }
    save_json(results, str(out / "evaluation_results.json"))

    # Text report
    report_txt = out / "evaluation_report.txt"
    with open(report_txt, "w") as f:
        f.write("POLICY EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Set: {total} queries\n")
        f.write(f"Evaluation Method: Automated heuristic (NOT human)\n\n")
        f.write(f"Results:\n")
        f.write(f"  Adaptive superior: {wins} ({wins/max(total,1)*100:.0f}%)\n")
        f.write(f"  Tie:               {ties} ({ties/max(total,1)*100:.0f}%)\n")
        f.write(f"  Baseline superior: {losses} ({losses/max(total,1)*100:.0f}%)\n\n")
        f.write(f"Average scores:\n")
        f.write(f"  Baseline: {np.mean(baseline_scores):.2f}/5\n")
        f.write(f"  Adaptive: {np.mean(adaptive_scores):.2f}/5\n\n")
        f.write(f"Statistical test (paired t-test):\n")
        f.write(f"  t={t_stat:.3f}, p={p_value:.4f}\n")
        f.write(f"  Effect size (Cohen's d): {effect_size:.3f}\n")

    logger.info(f"Text report saved to {report_txt}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier and adaptive tutor")
    parser.add_argument("--classifier", type=str, default="models/classifier_v1")
    parser.add_argument("--policy", type=str, default="models/policy_network_v1.pth")
    parser.add_argument("--test_queries", type=str, default="data/test_100.csv")
    parser.add_argument("--output", type=str, default="results/evaluation_results.json")
    parser.add_argument("--num_queries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    setup_logging(log_file="logs/evaluate.log")
    set_seed(args.seed)

    # Load classifier
    classifier = ExtractionClassifier()
    classifier_path = Path(args.classifier)
    if classifier_path.exists():
        classifier.load(str(classifier_path))
    else:
        logger.warning(f"Classifier not found at {classifier_path}. Using untrained model.")
        classifier.initialize()

    # --- Classifier Evaluation ---
    test_path = Path(args.test_queries)
    if test_path.exists():
        print("\n" + "=" * 60)
        print("CLASSIFIER EVALUATION")
        print("=" * 60)

        clf_metrics = evaluate_classifier(classifier, str(test_path))

        print(f"\nAccuracy:  {clf_metrics['accuracy']:.4f}")
        print(f"Precision: {clf_metrics['precision']:.4f}")
        print(f"Recall:    {clf_metrics['recall']:.4f}")
        print(f"F1:        {clf_metrics['f1']:.4f}")

        cm = clf_metrics["confusion_matrix"]
        print(f"\nConfusion Matrix:")
        print(f"              Pred Learn  Pred Extract")
        print(f"  True Learn     {cm[0][0]:4d}        {cm[0][1]:4d}")
        print(f"  True Extract   {cm[1][0]:4d}        {cm[1][1]:4d}")

        save_json(clf_metrics, "results/classifier_metrics.json")
    else:
        logger.warning(f"Test data not found: {test_path}")

    # --- Adaptive Tutor Evaluation ---
    print("\n" + "=" * 60)
    print("BASELINE vs. ADAPTIVE TUTOR COMPARISON")
    print("=" * 60)

    # Select evaluation queries
    if test_path.exists():
        queries = select_evaluation_queries(
            str(test_path), classifier, num_queries=args.num_queries
        )
    else:
        logger.warning("No test data. Using synthetic queries for evaluation.")
        from src.utils import SYNTHETIC_EXTRACTION_MESSAGES
        queries = [
            {"index": i, "message": msg, "extraction_score": 0.8, "label": 1}
            for i, msg in enumerate(SYNTHETIC_EXTRACTION_MESSAGES[:args.num_queries])
        ]

    # Create adaptive tutor
    tutor = AdaptiveTutor(classifier=classifier)

    # Load policy if available
    policy_path = Path(args.policy)
    if policy_path.exists():
        tutor.rl_trainer.load(str(policy_path))
        logger.info("Policy loaded for evaluation")
    else:
        logger.warning(f"Policy not found at {policy_path}. Using untrained policy.")

    # Compare responses
    print(f"\nComparing {len(queries)} queries...")
    comparisons = compare_responses(tutor, queries)

    # Score comparisons
    scores = [score_comparison(c) for c in comparisons]

    # Generate reports
    output_dir = Path(args.output).parent
    generate_comparison_report(comparisons, scores, str(output_dir))

    wins = sum(1 for s in scores if s["winner"] == "adaptive")
    ties = sum(1 for s in scores if s["winner"] == "tie")
    losses = sum(1 for s in scores if s["winner"] == "baseline")

    print(f"\nResults:")
    print(f"  Adaptive wins: {wins}/{len(scores)}")
    print(f"  Ties: {ties}/{len(scores)}")
    print(f"  Baseline wins: {losses}/{len(scores)}")
    print(f"\nReports saved to {output_dir}/")


if __name__ == "__main__":
    main()
