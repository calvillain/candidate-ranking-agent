# Benchmark Dataset Documentation

## Overview

This directory contains the benchmark dataset used to evaluate the candidate ranking agent's performance. The dataset consists of 10 carefully crafted test cases that represent real-world hiring scenarios with varying levels of difficulty.

## Dataset Format

### File: `benchmark_dataset.jsonl`

The benchmark dataset is stored in JSON Lines format (JSONL), where each line is a valid JSON object representing one test case.

### Schema

Each test case contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `case_id` | string | Unique identifier (e.g., "001", "002") |
| `job_description` | string | The job posting text with requirements |
| `resume_text` | string | The candidate's resume/profile text |
| `human_score` | integer | Ground truth score (0-100) from human annotator |
| `difficulty` | string | Test case difficulty: "easy", "medium", or "hard" |
| `category` | string | Scenario type (see categories below) |

### Example Entry

```json
{
  "case_id": "001",
  "job_description": "We are seeking a DevOps Engineer with 4+ years...",
  "resume_text": "Data Scientist with 6 years of experience...",
  "human_score": 20,
  "difficulty": "hard",
  "category": "career_transition"
}
```

## Test Case Categories

### 1. Perfect Match (Cases 002, 003, 004)
**Human Score Range**: 90-92  
**Difficulty**: Easy

Candidates who meet or exceed all job requirements. These cases test the agent's ability to recognize strong fits and assign high scores appropriately.

- **Case 002**: DevOps/SRE → Site Reliability Engineer
- **Case 003**: Frontend Engineer → React/TypeScript Developer  
- **Case 004**: Data Engineer → Analytics Engineer

### 2. Career Transition (Cases 001, 005, 006, 010)
**Human Score Range**: 5-45  
**Difficulty**: Hard

Candidates attempting to move into a different domain with limited or no experience in the target role. Tests the agent's ability to distinguish between adjacent/transferable skills versus required expertise.

- **Case 001**: Data Scientist → DevOps Engineer (score: 20)
- **Case 005**: Node.js/Frontend → Java Backend Engineer (score: 15)
- **Case 006**: Marketing Manager → ML Engineer (score: 5)
- **Case 010**: Backend Developer → Senior Security Engineer (score: 45)

### 3. Partial Match (Cases 007, 008, 009)
**Human Score Range**: 55-70  
**Difficulty**: Medium

Candidates with some relevant experience but missing key requirements or seniority level. Tests the agent's calibration for moderate fits.

- **Case 007**: Senior Backend (Python/Node.js) → Backend (Go/gRPC) (score: 55)
- **Case 008**: Data Scientist → Data Engineer (Spark/Snowflake) (score: 70)
- **Case 009**: Mid-level Backend → Senior Backend (AWS/microservices) (score: 65)

## Key Testing Challenges

### Hard Scoring Floors
Cases test whether the agent properly applies score caps for fundamental mismatches:
- **Case 006**: Non-technical → Technical (should cap at 5-15)
- **Case 001**: No domain experience + missing all core skills (should cap at 15-25)
- **Case 005**: Wrong language ecosystem (Node.js vs Java) (should cap at 10-25)

### Transferable Skills Recognition
Cases test whether the agent distinguishes between similar-sounding but non-equivalent skills:
- **Case 001**: Python for data science ≠ Python for DevOps/automation
- **Case 005**: Node.js for BFFs ≠ Java/Spring for backend microservices  
- **Case 010**: OAuth2 implementation ≠ Security engineering expertise
- **Case 006**: Using AI tools ≠ Building ML systems

### Career Transition Penalties
Cases test whether the agent applies appropriate penalties for role changes:
- Minor transition (score: 85): Backend → Full-Stack (-5 to -10 points)
- Major transition (score: 15): Frontend → Backend JVM (-25 to -35 points)
- Cross-discipline (score: 20): Data Scientist → DevOps (-30 to -40 points)
- Non-technical → Technical (score: 5): Marketing → ML Engineer (cap at 5-15)

### Experience Level Mismatches
Cases test seniority gap handling:
- **Case 009**: 4 years vs 7+ years required + no leadership experience
- **Case 007**: Missing specific tech stack despite senior years

## Evaluation Metrics

### Mean Absolute Error (MAE)
**Target**: ≤ 8.0 points  
**Calculation**: Average of absolute differences between predicted and human scores

Best performance to date:
- DeepSeek-R1:8b: MAE 6.4 ✅
- llama3.1:8b: MAE 12.3 ❌

### Calibration Breakdown
Error ranges based on absolute difference:
- **Perfect** (0-5 points): Excellent calibration
- **Good** (6-10 points): Acceptable variance
- **Acceptable** (11-15 points): Minor miscalibration
- **Poor** (16-20 points): Significant error
- **Very Poor** (21+ points): Critical miscalibration

### Pass Rate
Percentage of cases where reasoning quality meets threshold (GEval score ≥ 0.7).

Target: 100%

## Usage

### Load Dataset (Python)

```python
import json

def load_benchmark():
    cases = []
    with open("data/benchmark_dataset.jsonl", "r") as f:
        for line in f:
            cases.append(json.loads(line))
    return cases

cases = load_benchmark()
print(f"Loaded {len(cases)} test cases")
```

### Run Evaluation

```bash
# Full evaluation with default model
uv run python eval_agent.py

# Specify custom benchmark path
uv run python eval_agent.py data/benchmark_dataset.jsonl

# Use different model
OLLAMA_MODEL=llama3.1:8b uv run python eval_agent.py
```

### Evaluation Output

Results are saved to `data/evaluations/TIMESTAMP_eval_MODEL.json` with:
- **Metadata**: Timestamp, model, benchmark path
- **Metrics**: MAE, pass rate, avg quality score, calibration breakdown
- **Detailed Results**: Per-case predictions, errors, reasoning, quality scores

## Dataset Design Principles

1. **Realistic Scenarios**: Job descriptions and resumes mirror real-world hiring
2. **Clear Ground Truth**: Human scores justified by explicit skill matches/gaps
3. **Edge Case Coverage**: Tests common failure modes (career transitions, transferable skills)
4. **Difficulty Distribution**: 
   - Easy: 3 cases (30%)
   - Medium: 3 cases (30%)
   - Hard: 4 cases (40%)
5. **Score Distribution**:
   - 0-20: 3 cases (very poor fit)
   - 21-50: 1 case (poor fit)
   - 51-70: 3 cases (moderate fit)
   - 71-100: 3 cases (strong fit)

## Maintenance

### Adding New Test Cases

1. Create realistic job description and resume
2. Assign ground truth score based on:
   - Required skills match rate
   - Experience level alignment
   - Career transition penalties
   - Domain expertise relevance
3. Classify difficulty and category
4. Add to `benchmark_dataset.jsonl` as new JSON line
5. Update this README with case details

### Updating Existing Cases

Changes to test cases should maintain consistency with:
- Historical evaluation reports
- Documented calibration improvements
- Ground truth score justification

Document changes in git commit messages and update evaluation baselines.

## Historical Context

This benchmark dataset was reconstructed from evaluation reports after the original was deleted. Key reference documents:
- `EVALUATION_RESULTS_2026-02-08_18-39-33.md`
- `CALIBRATION_IMPROVEMENTS_IMPLEMENTED.md`
- `EVALUATION_RESULTS_FINAL_SUMMARY.md`

The test cases represent lessons learned from calibration efforts and reflect common patterns in hiring scenarios that challenge LLM-based ranking systems.

## Model Performance Summary

| Model | MAE | Pass Rate | Best Use Case |
|-------|-----|-----------|---------------|
| DeepSeek-R1:8b | 6.4 | 90% | Production (accuracy-critical) |
| llama3.1:8b | 12.3 | 80% | Development (speed-focused) |

Target threshold: MAE ≤ 8.0

## References

- Evaluation framework: `eval_agent.py`
- Agent implementation: `agent.py`, `ranking_engine.py`
- Data models: `models.py`
- Setup instructions: `AGENTS.md`
