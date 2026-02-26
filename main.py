"""
CLI entry point for candidate profile extraction.

Provides command-line interface for extracting structured candidate profiles
from raw resume text. Useful for testing and one-off extractions.
"""

from __future__ import annotations

import sys
import logging

from ranking_engine import extract_candidate_profile

logger = logging.getLogger(__name__)


def _usage() -> str:
    """Return usage instructions for the CLI."""
    return (
        "Usage: python main.py [RESUME_TEXT]\n"
        "  If RESUME_TEXT is omitted, reads resume text from stdin.\n"
        "  Outputs structured JSON profile to stdout.\n"
        "\n"
        "Environment variables:\n"
        "  OLLAMA_BASE_URL - Ollama server URL (default: http://localhost:11434)\n"
        "  OLLAMA_MODEL - Model name (default: deepseek-r1:8b)\n"
        "  LOG_LEVEL - Logging level (default: INFO)\n"
        "\n"
        "Examples:\n"
        '  python main.py "John Doe, 5 years Python experience..."\n'
        "  cat resume.txt | python main.py\n"
        '  echo "Resume text" | python main.py > profile.json\n'
    )


def main() -> None:
    """
    Main entry point for profile extraction CLI.

    Reads resume text from command line args or stdin, extracts profile,
    and outputs JSON to stdout.

    Exits with code 1 on extraction failure.
    """
    if "--help" in sys.argv or "-h" in sys.argv:
        print(_usage())
        return

    # Read input from args or stdin
    if len(sys.argv) > 1:
        raw_text = " ".join(sys.argv[1:])
    else:
        raw_text = sys.stdin.read()

    if not raw_text.strip():
        logger.error("No resume text provided")
        print("Error: No resume text provided", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        sys.exit(1)

    # Extract profile
    try:
        profile = extract_candidate_profile(raw_text)
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        print(f"Error: Extraction failed - {e}", file=sys.stderr)
        sys.exit(1)

    # Output JSON to stdout
    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
