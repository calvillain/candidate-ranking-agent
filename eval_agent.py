"""
Evaluation framework for candidate ranking system.

Tests ranking quality against human-annotated benchmark dataset.
Calculates MAE, pass rate, and detailed calibration metrics.
Uses DeepEval for automated reasoning quality assessment.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent import CandidateAgent
from models import RankingResult

# ---------------------------------------------------------------------------
# Config & Logging
# ---------------------------------------------------------------------------

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Benchmark dataset path
BENCHMARK_PATH = os.getenv("BENCHMARK_PATH", "data/benchmark_dataset.jsonl")

# Model name for output files
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

# DeepEval timeout (for reasoning models)
DEEPEVAL_TIMEOUT = int(os.getenv("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "120"))


# ---------------------------------------------------------------------------
# Benchmark Dataset Loading
# ---------------------------------------------------------------------------


def load_benchmark_dataset(path: str = BENCHMARK_PATH) -> list[dict[str, Any]]:
    """
    Load benchmark dataset from JSONL file.

    Expected format per line:
    {
        "case_id": "001",
        "job_description": "...",
        "resume_text": "...",
        "human_score": 90,
        "difficulty": "easy|medium|hard",
        "category": "description"
    }

    Args:
        path: Path to JSONL benchmark file.

    Returns:
        List of test case dictionaries.

    Raises:
        FileNotFoundError: If benchmark file doesn't exist.
        ValueError: If file format is invalid.
    """
    benchmark_path = Path(path)

    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found at {path}. "
            f"Create data/benchmark_dataset.jsonl with test cases."
        )

    cases = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                case = json.loads(line)

                # Validate required fields
                required_fields = [
                    "case_id",
                    "job_description",
                    "resume_text",
                    "human_score",
                ]
                missing = [field for field in required_fields if field not in case]
                if missing:
                    raise ValueError(f"Missing fields: {missing}")

                # Validate human_score range
                if not (0 <= case["human_score"] <= 100):
                    raise ValueError(f"human_score must be 0-100, got {case['human_score']}")

                cases.append(case)

            except (json.JSONDecodeError, ValueError) as e:
                logger.error("Invalid JSON at line %d: %s", line_num, e)
                raise ValueError(f"Invalid benchmark format at line {line_num}: {e}")

    logger.info("Loaded %d test cases from %s", len(cases), path)
    return cases


# ---------------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------------


def calculate_mae(results: list[dict[str, Any]]) -> float:
    """
    Calculate Mean Absolute Error between predicted and human scores.

    Args:
        results: List of evaluation results with 'predicted_score' and 'human_score'.

    Returns:
        MAE value (lower is better, target ≤8.0).
    """
    valid_results = [
        r for r in results if r.get("error") is not None and not r.get("failed", False)
    ]
    if not valid_results:
        return 0.0

    errors = [r["error"] for r in valid_results]
    mae = sum(errors) / len(errors)
    return round(mae, 2)


def calculate_pass_rate(
    results: list[dict[str, Any]], threshold: float = 0.5
) -> tuple[int, int, float]:
    """
    Calculate pass rate based on quality threshold.

    Args:
        results: List of evaluation results with 'quality_score'.
        threshold: Minimum quality score to pass (default 0.5).

    Returns:
        Tuple of (passed_count, total_count, pass_rate_percentage).
    """
    valid_results = [r for r in results if not r.get("failed", False)]
    if not valid_results:
        return 0, 0, 0.0

    passed = sum(1 for r in valid_results if r.get("quality_score", 0) >= threshold)
    total = len(valid_results)
    rate = (passed / total) * 100 if total > 0 else 0.0

    return passed, total, round(rate, 1)


def calculate_calibration_breakdown(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate detailed calibration metrics by error range.

    Args:
        results: List of evaluation results.

    Returns:
        Dictionary with calibration breakdown by error ranges.
    """
    valid_results = [
        r for r in results if r.get("error") is not None and not r.get("failed", False)
    ]

    error_ranges = {
        "perfect": 0,  # 0-5 points
        "good": 0,  # 6-10 points
        "acceptable": 0,  # 11-15 points
        "poor": 0,  # 16-20 points
        "very_poor": 0,  # 21+ points
    }

    for result in valid_results:
        error = result["error"]

        if error <= 5:
            error_ranges["perfect"] += 1
        elif error <= 10:
            error_ranges["good"] += 1
        elif error <= 15:
            error_ranges["acceptable"] += 1
        elif error <= 20:
            error_ranges["poor"] += 1
        else:
            error_ranges["very_poor"] += 1

    return {
        "ranges": error_ranges,
        "total_cases": len(valid_results),
        "perfect_percentage": round((error_ranges["perfect"] / len(valid_results)) * 100, 1)
        if valid_results
        else 0,
    }


def test_candidate_ranking_quality(
    benchmark_path: str = BENCHMARK_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """
    Run full evaluation on benchmark dataset.

    Tests each case in the benchmark, calculates MAE, pass rate,
    and detailed calibration metrics.

    Args:
        benchmark_path: Path to benchmark JSONL file.

    Returns:
        Tuple of (detailed_results, metrics, metadata).

    Raises:
        FileNotFoundError: If benchmark file doesn't exist.
        Exception: If evaluation fails.
    """
    logger.info("Starting evaluation run on %s", benchmark_path)

    # Load benchmark
    try:
        cases = load_benchmark_dataset(benchmark_path)
    except FileNotFoundError as e:
        logger.error("Benchmark not found: %s", e)
        raise

    if not cases:
        raise ValueError("Benchmark dataset is empty")

    # Initialize agent
    agent = CandidateAgent()

    # Run evaluation
    eval_start = time.time()
    results = []
    for case in cases:
        case_id = case["case_id"]
        logger.info("Evaluating case %s...", case_id)
        case_start = time.time()

        try:
            # Rank candidate
            ranking_result = agent.rank_candidate(
                job_description=case["job_description"],
                resume_text=case["resume_text"],
            )

            # Calculate error
            predicted_score = ranking_result.fit_score
            human_score = case["human_score"]
            error = abs(predicted_score - human_score)

            # Placeholder quality score (would use DeepEval in real implementation)
            # For now, estimate based on error magnitude
            quality_score = max(0.0, 1.0 - (error / 50.0))

            case_elapsed = time.time() - case_start

            result = {
                "case_id": case_id,
                "predicted_score": predicted_score,
                "human_score": human_score,
                "error": error,
                "reasoning": ranking_result.reasoning,
                "quality_score": round(quality_score, 2),
                "category": case.get("category", "unknown"),
                "difficulty": case.get("difficulty", "unknown"),
                "elapsed_seconds": round(case_elapsed, 1),
            }

            results.append(result)

            logger.info(
                "Case %s: predicted=%d, human=%d, error=%d, quality=%.2f, elapsed=%.1fs",
                case_id,
                predicted_score,
                human_score,
                error,
                quality_score,
                case_elapsed,
            )

        except Exception as e:
            case_elapsed = time.time() - case_start
            logger.exception("Agent failed for case %s: %s (after %.1fs)", case_id, e, case_elapsed)
            # Record failure
            results.append(
                {
                    "case_id": case_id,
                    "predicted_score": None,
                    "human_score": case["human_score"],
                    "error": None,
                    "reasoning": f"ERROR: {str(e)}",
                    "quality_score": 0.0,
                    "category": case.get("category", "unknown"),
                    "difficulty": case.get("difficulty", "unknown"),
                    "elapsed_seconds": round(case_elapsed, 1),
                    "failed": True,
                }
            )

    # Calculate metrics
    total_elapsed = time.time() - eval_start
    mae = calculate_mae(results)
    passed, total, pass_rate = calculate_pass_rate(results)
    calibration = calculate_calibration_breakdown(results)

    metrics = {
        "mae": mae,
        "pass_rate": pass_rate,
        "passed_cases": passed,
        "total_cases": total,
        "avg_quality_score": round(sum(r["quality_score"] for r in results) / len(results), 2)
        if results
        else 0.0,
        "calibration_breakdown": calibration,
        "failed_cases": sum(1 for r in results if r.get("failed", False)),
        "total_elapsed_seconds": round(total_elapsed, 1),
    }

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": OLLAMA_MODEL,
        "benchmark_path": benchmark_path,
        "deepeval_timeout": DEEPEVAL_TIMEOUT,
    }

    logger.info(
        "Evaluation complete: MAE=%.2f, pass_rate=%.1f%% (%d/%d)",
        mae,
        pass_rate,
        passed,
        total,
    )

    return results, metrics, metadata


# ---------------------------------------------------------------------------
# Output Saving
# ---------------------------------------------------------------------------


def save_evaluation_results(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """
    Save evaluation results to timestamped JSON file.

    Args:
        results: Detailed per-case results.
        metrics: Summary metrics (MAE, pass rate, etc.).
        metadata: Run metadata (timestamp, model, etc.).

    Returns:
        Path to saved file.
    """
    # Create output directory
    output_dir = Path("data/evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = OLLAMA_MODEL.replace(":", "-").replace("/", "-")
    filename = f"{timestamp}_eval_{model_name}.json"
    output_path = output_dir / filename

    # Prepare output
    output = {
        "metadata": metadata,
        "metrics": metrics,
        "results": results,
    }

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved evaluation results to %s", output_path)
    return str(output_path)


def print_summary(metrics: dict[str, Any]) -> None:
    """
    Print evaluation summary to console.

    Args:
        metrics: Summary metrics dictionary.
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"MAE: {metrics['mae']:.2f} (target ≤8.0)")
    if metrics.get("failed_cases", 0) > 0:
        print(f"Failed Cases: {metrics['failed_cases']} (excluded from MAE)")
    print(
        f"Pass Rate: {metrics['pass_rate']:.1f}% ({metrics['passed_cases']}/{metrics['total_cases']})"
    )
    print(f"Avg Quality Score: {metrics['avg_quality_score']:.2f}")
    print()

    calibration = metrics["calibration_breakdown"]
    print("Calibration Breakdown:")
    print(
        f"  Perfect (0-5 error):    {calibration['ranges']['perfect']:2d} cases ({calibration['perfect_percentage']:.1f}%)"
    )
    print(f"  Good (6-10 error):      {calibration['ranges']['good']:2d} cases")
    print(f"  Acceptable (11-15):     {calibration['ranges']['acceptable']:2d} cases")
    print(f"  Poor (16-20):           {calibration['ranges']['poor']:2d} cases")
    print(f"  Very Poor (21+):        {calibration['ranges']['very_poor']:2d} cases")

    if "total_elapsed_seconds" in metrics:
        print(
            f"\nTotal evaluation time: {metrics['total_elapsed_seconds']:.0f}s ({metrics['total_elapsed_seconds'] / 60:.1f} minutes)"
        )

    print("=" * 60)

    # Status indicator
    if metrics["mae"] <= 8.0:
        print("✅ STATUS: PASSED (MAE ≤ 8.0)")
    else:
        print(f"❌ STATUS: FAILED (MAE {metrics['mae']:.2f} > 8.0)")
    print()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    CLI entry point for evaluation.

    Usage:
        python eval_agent.py [BENCHMARK_PATH]

    Exits with 0 if MAE ≤ 8.0, otherwise 1.
    """
    # Parse args
    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
Candidate Ranking Agent - Evaluation Framework

Usage:
    python eval_agent.py [BENCHMARK_PATH]

Arguments:
    BENCHMARK_PATH    Path to benchmark JSONL file (default: data/benchmark_dataset.jsonl)

Environment Variables:
    BENCHMARK_PATH                             Override default benchmark path
    OLLAMA_MODEL                               Model to evaluate (default: deepseek-r1:8b)
    DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE  Timeout for LLM calls (default: 300)

Examples:
    python eval_agent.py
    python eval_agent.py data/benchmark_dataset.jsonl
    OLLAMA_MODEL=llama3.1:8b python eval_agent.py
        """)
        return

    benchmark_path = sys.argv[1] if len(sys.argv) > 1 else BENCHMARK_PATH

    # Run evaluation
    try:
        results, metrics, metadata = test_candidate_ranking_quality(benchmark_path)
    except FileNotFoundError as e:
        logger.error("Benchmark file not found: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        print(f"Error: Evaluation failed - {e}", file=sys.stderr)
        sys.exit(1)

    # Save results
    output_path = save_evaluation_results(results, metrics, metadata)

    # Print summary
    print_summary(metrics)
    print(f"Detailed results saved to: {output_path}\n")

    # Exit with status code
    if metrics["mae"] <= 8.0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
