# Candidate Ranking Agent

A **local-first AI-powered candidate ranking system** that extracts structured profiles from resumes and ranks candidates against job descriptions using locally-hosted LLMs. No cloud APIs, no data leaves your machine.

## Features

- **Local-First Architecture**: Uses Ollama with DeepSeek-R1:8b running entirely on your hardware
- **Privacy-Preserving**: All data processing happens locally — sensitive candidate information never leaves your system
- **Structured Extraction**: Converts raw resume text into validated `CandidateProfile` objects (skills, experience, past titles)
- **Intelligent Ranking**: Produces 0-100 fit scores with detailed reasoning for hiring decisions
- **High Accuracy**: Achieves MAE ≤ 5.9 against human annotations (90% pass rate on benchmark)
- **Retry Logic**: Built-in exponential backoff for LLM reliability
- **Evaluation Framework**: Automated benchmarking with human-annotated test cases

## Architecture

```
Resume Text → [Extraction LLM] → CandidateProfile → [Ranking LLM] → RankingResult (fit_score + reasoning)
```

### Key Modules

- **`models.py`** — Pydantic v2 schemas for `CandidateProfile` (skills, years_of_experience, past_titles) and `RankingResult` (fit_score 0-100, reasoning)
- **`ranking_engine.py`** — Core LLM integration with extraction/ranking prompts, ChatOllama client setup, retry logic (tenacity), structured output parsing. Uses temperature 0.1 for extraction, 0.2 for ranking, 300s timeout for reasoning models
- **`agent.py`** — `CandidateAgent` orchestration class providing `extract_profile()`, `rank_candidate()` (end-to-end), and `rank_from_profile()` (pre-extracted)
- **`main.py`** — CLI entry point for profile extraction (args or stdin → JSON output)
- **`eval_agent.py`** — Evaluation framework: loads JSONL benchmark, runs all cases, calculates MAE/pass rate/calibration, saves timestamped JSON to `data/evaluations/`

### Data

- **`data/benchmark_dataset.jsonl`** — 10 human-annotated test cases (JSONL format) with case_id, job_description, resume_text, human_score, difficulty, and category

## Prerequisites

- **Python 3.10+** with `uv` package manager ([installation guide](https://docs.astral.sh/uv/))
- **Ollama** installed and running ([download](https://ollama.ai/))
- **~11GB VRAM** for DeepSeek-R1:8b (recommended model)
- **~8GB VRAM** minimum for llama3.1:8b (faster alternative)

## Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd candidate-ranking-agent

# 2. Install dependencies (creates .venv automatically)
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env if you need to customize Ollama settings

# 4. Pull the recommended LLM model
ollama pull deepseek-r1:8b

# 5. Extract a candidate profile from resume text
uv run python main.py "John Doe. Software Engineer with 5 years of Python experience..."

# 6. Rank a candidate (Python API example)
uv run python -c "
from agent import CandidateAgent

agent = CandidateAgent()
result = agent.rank_candidate(
    job_description='Looking for a Senior Python Developer with 5+ years...',
    resume_text='John Doe. Software Engineer with 5 years of Python...'
)
print(f'Fit Score: {result.fit_score}/100')
print(f'Reasoning: {result.reasoning}')
"
```

## Configuration

Create a `.env` file from `.env.example` and configure these variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server endpoint |
| `OLLAMA_MODEL` | `deepseek-r1:8b` | Model name for LLM operations |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `BENCHMARK_PATH` | `data/benchmark_dataset.jsonl` | Path to evaluation dataset |
| `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE` | `300` | Timeout for reasoning model calls |

## Usage

### CLI: Profile Extraction

Extract structured candidate profiles from raw resume text:

```bash
# From command-line argument
uv run python main.py "Jane Smith. Senior DevOps Engineer with 7 years of Kubernetes..."

# From file
uv run python main.py "$(cat resume.txt)"

# Output (JSON)
{
  "skills": ["Kubernetes", "Docker", "AWS", "Python", "Terraform"],
  "years_of_experience": 7.0,
  "past_titles": ["Senior DevOps Engineer", "DevOps Engineer", "Systems Administrator"],
  "raw_resume_text": "Jane Smith. Senior DevOps Engineer with 7 years of Kubernetes..."
}
```

### Python API: End-to-End Ranking

Rank candidates directly from resume text:

```python
from agent import CandidateAgent

agent = CandidateAgent()

# Rank candidate against job description
result = agent.rank_candidate(
    job_description="""
    Senior Backend Engineer
    Requirements:
    - 5+ years Python/Django experience
    - Strong knowledge of PostgreSQL and Redis
    - Experience with microservices architecture
    - AWS cloud infrastructure experience
    """,
    resume_text="""
    Alex Johnson
    Senior Software Engineer at TechCorp (2019-Present)
    - Built microservices platform using Python/Django
    - Designed PostgreSQL database schemas for 10M+ users
    - Deployed services to AWS ECS with Redis caching
    Skills: Python, Django, PostgreSQL, Redis, AWS, Docker
    """
)

print(f"Fit Score: {result.fit_score}/100")
print(f"Reasoning:\n{result.reasoning}")
```

### Python API: Ranking from Pre-Extracted Profile

If you already have a structured `CandidateProfile`, skip extraction:

```python
from agent import CandidateAgent
from models import CandidateProfile

agent = CandidateAgent()

# Create profile manually or from previous extraction
profile = CandidateProfile(
    skills=["Python", "Django", "PostgreSQL", "Redis", "AWS", "Docker"],
    years_of_experience=5.0,
    past_titles=["Senior Software Engineer", "Software Engineer"],
)

# Rank pre-extracted profile
result = agent.rank_from_profile(
    job_description="Senior Backend Engineer with Python/Django...",
    profile=profile
)

print(f"Fit Score: {result.fit_score}/100")
```

## Evaluation

Run automated evaluation against human-annotated benchmark:

```bash
# Run full benchmark evaluation (saves to timestamped JSON file)
uv run python eval_agent.py

# Run with custom benchmark path
uv run python eval_agent.py data/benchmark_dataset.jsonl

# Or set via environment variable
BENCHMARK_PATH=data/benchmark_dataset.jsonl uv run python eval_agent.py
```

### Evaluation Output

- **Console Summary**: Pass/fail status, scores, calibration MAE for each test case
- **JSON Output**: Detailed results saved to `data/evaluations/TIMESTAMP_eval_MODEL.json`
- **Metrics Tracked**: Mean Absolute Error (MAE), pass rate (within ±10 points), calibration breakdown by score range

### Model Comparison

| Model | MAE | Pass Rate | Speed (per candidate) | VRAM | Recommendation |
|-------|-----|-----------|----------------------|------|----------------|
| **DeepSeek-R1:8b** | **5.9** | **90%** | 10-20 minutes | ~11GB | ✅ **Recommended** — Best accuracy, achieves target MAE ≤8.0 |
| llama3.1:8b | 12.3 | 80% | 30-60 seconds | ~8GB | Alternative for speed-critical use cases (if MAE ~12 acceptable) |

**Current deployment**: DeepSeek-R1:8b (accuracy-first approach)

## Project Structure

```
candidate-ranking-agent/
├── agent.py                      # Orchestration layer (CandidateAgent)
├── models.py                     # Pydantic data models (CandidateProfile, RankingResult)
├── ranking_engine.py             # Core LLM integration (extraction/ranking prompts)
├── main.py                       # CLI entry point (profile extraction)
├── eval_agent.py                 # Evaluation framework (benchmarking)
├── .env.example                  # Configuration template
├── pyproject.toml                # Project dependencies (uv)
├── AGENTS.md                     # Developer guidelines (build/lint/test commands)
└── data/
    ├── benchmark_dataset.jsonl   # 10 human-annotated test cases
    └── evaluations/              # Saved evaluation results (timestamped JSON)
```

## Development

See **AGENTS.md** for:
- Build/lint/test commands
- Code style guidelines (type hints, error handling, logging)
- Jupyter notebook setup for interactive analysis
- Troubleshooting common issues

Quick dev setup:

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Optional: Add dev dependencies
uv add --dev mypy ruff

# Type checking
uv run mypy ranking_engine.py agent.py models.py

# Linting and formatting
uv run ruff check .
uv run ruff format .
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama server if not running
ollama serve
```

### Model Not Found

```bash
# List available models
ollama list

# Pull the recommended model
ollama pull deepseek-r1:8b
```

### PyArrow Import Error

If you see `AttributeError: module 'pyarrow' has no attribute '__version__'`:

```bash
# Reinstall pyarrow cleanly
uv pip uninstall pyarrow -y
uv pip install pyarrow

# Or rebuild environment
rm -rf .venv
uv sync
```

## License

License not yet specified. See LICENSE file for details (to be added).

## Contributing

Contributions are welcome! Please ensure:
- All functions have type hints
- Code follows style guidelines in AGENTS.md
- New features include evaluation test cases
- LLM prompts are documented with expected output formats

## Acknowledgments

Built with:
- [Ollama](https://ollama.ai/) for local LLM hosting
- [LangChain](https://www.langchain.com/) for LLM integration
- [Pydantic](https://docs.pydantic.dev/) for data validation
- [uv](https://docs.astral.sh/uv/) for dependency management
