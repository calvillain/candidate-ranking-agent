# Evaluation Results Directory

This directory stores timestamped evaluation results from benchmark runs.

## File Naming Convention

```
YYYY-MM-DD_HH-MM-SS_eval_MODEL-NAME.json
```

Example: `2026-02-08_18-39-33_eval_llama3.1-8b.json`

## Output Format

Each evaluation result file contains:

```json
{
  "metadata": {
    "timestamp": "2026-02-08T18:39:33",
    "model": "llama3.1:8b",
    "benchmark_path": "data/benchmark_dataset.jsonl",
    "deepeval_timeout": 300
  },
  "metrics": {
    "mae": 12.3,
    "pass_rate": 80.0,
    "passed_cases": 8,
    "total_cases": 10,
    "avg_quality_score": 0.66,
    "calibration_breakdown": {
      "ranges": {
        "perfect": 2,
        "good": 3,
        "acceptable": 1,
        "poor": 2,
        "very_poor": 2
      },
      "total_cases": 10,
      "perfect_percentage": 20.0
    }
  },
  "results": [
    {
      "case_id": "001",
      "predicted_score": 10,
      "human_score": 20,
      "error": 10,
      "reasoning": "...",
      "quality_score": 0.2,
      "category": "career_transition",
      "difficulty": "hard"
    }
    // ... more results
  ]
}
```

## Usage

Results are automatically generated when running:

```bash
uv run python eval_agent.py
```

View historical evaluations to track model performance over time and compare calibration improvements.

## Metrics Explained

### MAE (Mean Absolute Error)
- **Target**: ≤ 8.0
- Average difference between predicted and human scores
- Lower is better

### Pass Rate
- Percentage of cases with quality score ≥ 0.7
- Measures reasoning quality
- Target: 100%

### Calibration Breakdown
- **Perfect** (0-5 error): Excellent
- **Good** (6-10 error): Acceptable
- **Acceptable** (11-15 error): Minor issues
- **Poor** (16-20 error): Significant issues
- **Very Poor** (21+ error): Critical problems

## Historical Performance

| Date | Model | MAE | Pass Rate | Notes |
|------|-------|-----|-----------|-------|
| 2026-02-07 | deepseek-r1:8b | 6.4 | 90% | Best performance ✅ |
| 2026-02-08 | llama3.1:8b | 12.3 | 80% | After calibration changes |
| 2026-02-08 | llama3.1:8b | 14.3 | 100% | Earlier baseline |

The directory was recreated after data loss. Previous evaluation files were documented in:
- `EVALUATION_RESULTS_2026-02-08_18-39-33.md`
- `EVALUATION_RESULTS_FINAL_SUMMARY.md`
