"""
Ranking Engine: Resume processing and candidate ranking logic.

Core ranking functionality for processing resumes and job descriptions
via local LLM (Ollama). Contains extraction prompts, ranking prompts,
LLM client setup, and retry logic.

Default model: DeepSeek-R1 8B (MAE 6.4, achieves target ≤8.0).
"""

from __future__ import annotations

import logging
import os
import sys
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from models import CandidateProfile, RankingResult

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

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# DeepSeek-R1 8B: MAE 6.4, 90% pass rate, reasoning-focused model
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

# ---------------------------------------------------------------------------
# System prompt: extract only relevant professional data (think then output)
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a resume parser. Extract only professional information from resume text.

CRITICAL: Your final output MUST be valid JSON with this exact format:
{{"skills": ["skill1", "skill2"], "years_of_experience": <number>, "past_titles": ["title1", "title2"]}}

Think through the resume, then output ONLY the JSON object.

Extraction steps:
1. Identify all technical/professional skills (languages, frameworks, tools) - ignore hobbies
2. Determine total years of professional work experience (number only, can be decimal like 2.5)
3. List job titles in reverse chronological order (most recent first)

Rules:
- Skills: Only TECHNICAL and PROFESSIONAL skills, not soft skills or hobbies
- Years: Single number (e.g., 5 or 2.5). Use 0.0 ONLY if truly unclear or no experience stated
- Titles: Exact or normalized job titles, reverse chronological order
- Count WORK experience, not education

Output format example:
{{"skills": ["Python", "AWS", "Kubernetes"], "years_of_experience": 6.0, "past_titles": ["DevOps Engineer", "Junior DevOps"]}}

Now output ONLY the JSON object - no other text before or after."""

USER_PROMPT_TEMPLATE = """Resume text:
---
{resume_text}
---

Think step by step (what skills? how many years? what titles?), then output only the JSON object with keys: skills, years_of_experience, past_titles."""

RANKING_USER_TEMPLATE = """Job description:
---
{job_description}
---

Candidate profile:
- Skills: {skills}
- Years of experience: {years_of_experience}
- Past titles: {past_titles}

Think step by step about how well this candidate fits the role (what requirements does the job have? which ones match? which ones are missing?), then output only the JSON object with keys: fit_score (0-100) and reasoning."""

# ---------------------------------------------------------------------------
# LLM client with structured output
# ---------------------------------------------------------------------------


def _get_llm() -> ChatOllama:
    """Build ChatOllama client from environment (DevOps-ready) with timeout."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.1,
        timeout=300,  # 5 minute timeout for reasoning models
    ).with_structured_output(CandidateProfile)


def _get_ranking_llm() -> ChatOllama:
    """LLM for ranking (fit score + reasoning) with timeout."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.2,
        timeout=300,  # 5 minute timeout for reasoning models
    ).with_structured_output(RankingResult)


@retry(
    retry=retry_if_exception_type((ValueError, ConnectionError, Exception)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        "LLM call failed, retrying (attempt %s): %s",
        retry_state.attempt_number,
        retry_state.outcome.exception()
        if retry_state.outcome and retry_state.outcome.failed
        else None,
    ),
)
def extract_candidate_profile(raw_resume_text: str) -> CandidateProfile:
    """
    Clean raw resume text into a structured CandidateProfile using the local LLM.

    Args:
        raw_resume_text: Full text extracted from a resume (e.g. from PDF).

    Returns:
        CandidateProfile with skills, years_of_experience, past_titles.
        raw_resume_text on the result is not set by the LLM (to keep payload small);
        caller can attach it if needed.

    Raises:
        ValueError: If the LLM returns invalid or empty required fields.
        ConnectionError: If Ollama server is unreachable.
        Exception: For other LLM invocation failures.
    """
    if not raw_resume_text or not raw_resume_text.strip():
        logger.warning("Empty resume text provided; returning minimal profile.")
        return CandidateProfile(
            skills=[],
            years_of_experience=0.0,
            past_titles=[],
            raw_resume_text=raw_resume_text or None,
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACTION_SYSTEM_PROMPT),
            ("human", USER_PROMPT_TEMPLATE),
        ]
    )
    chain = prompt | _get_llm()

    logger.info("Calling LLM to extract candidate profile (model=%s).", OLLAMA_MODEL)
    try:
        result: CandidateProfile = chain.invoke({"resume_text": raw_resume_text})
    except Exception as e:
        logger.exception("LLM invocation failed: %s", e)
        raise

    if not isinstance(result, CandidateProfile):
        raise ValueError(f"Expected CandidateProfile, got {type(result)}")

    logger.info(
        "Extracted profile: %d skills, %.1f years exp, %d titles",
        len(result.skills),
        result.years_of_experience,
        len(result.past_titles),
    )
    # Optionally attach raw text for downstream use (LLM does not return it)
    result.raw_resume_text = raw_resume_text.strip()

    # Validation: Warn if extraction looks suspicious
    if result.years_of_experience == 0.0 and len(result.skills) > 5:
        logger.warning(
            "Suspicious extraction: 0 years but %d skills. Resume may have unclear experience format.",
            len(result.skills),
        )

    return result


# ---------------------------------------------------------------------------
# Reasoning: profile + JD -> fit score + reasoning
# ---------------------------------------------------------------------------

RANKING_SYSTEM_PROMPT = """You are a candidate-job fit assessor. Given a candidate profile and a job description, output a fit score (0-100) and reasoning.

CRITICAL: Your final output MUST be valid JSON with this exact format:
{{"fit_score": <number 0-100>, "reasoning": "<text>"}}

Think through the analysis, then output ONLY the JSON object.

Analysis steps:
1. List key JD requirements (skills, experience, seniority)
2. Match each requirement against the candidate's actual skills and experience
3. Only count gaps for skills NOT present in the candidate profile — do NOT invent gaps
4. Accept equivalent skills: "incident response" = "incident management", "led outages" implies "on-call experience"
5. Apply calibration rules below
6. Write reasoning citing specific matches and gaps

Calibration rules (apply consistently):
- Strong fit (85-100): Meets ALL or nearly all JD requirements with sufficient experience
- Moderate fit (55-84): Has many relevant skills but missing some requirements
- Weak fit (25-54): Some transferable skills but missing most core requirements
- Poor fit (0-24): Different career domain, missing nearly all required skills

Perfect match rule (IMPORTANT — apply carefully):
- If candidate has ALL required skills listed in JD + meets experience level → score 85-95
- Only penalize for skills explicitly required in JD that are genuinely absent from the profile
- Do NOT invent gaps — if the resume says they have a skill, they have it
- Accept synonyms and implied competencies (e.g., "SRE principles, SLIs, SLOs" implies on-call and incident management experience)
- Nice-to-have gaps → reduce by 5 max
- Verify: before scoring below 85, list which SPECIFIC required skills are truly missing

Experience gaps:
- 0-1 year short: -5 points
- 1-2 years short: -10 points  
- 2-3 years short: -15 points
- 3+ years short: -20 to -25 points
- State "X years vs Y+ required" in reasoning

Career transitions (candidate's domain differs from JD's domain):
- JD requires a specific language/framework (Java, Go, Spring Boot) that candidate has ZERO experience with → 10-25 maximum, even if role types sound similar
- Adjacent role with shared core skills (e.g., Data Scientist with Python/SQL → Data Engineer): 55-70
- Non-technical → Technical (e.g., Marketing → ML Engineer): 5-15

Reasoning requirements:
- Cite at least 3 specific JD requirements
- For each, state whether the candidate matches or not
- For 85+: Confirm ALL required skills are present
- For <50: List which core required skills are missing

Output format example:
{{"fit_score": 75, "reasoning": "Candidate has 4 years vs 5+ required (-5). Matches React, TypeScript, Jest, responsive design (core skills). Missing: AWS experience (required). Score reflects strong core match with experience gap and one missing requirement."}}

Now output ONLY the JSON object - no other text before or after."""


@retry(
    retry=retry_if_exception_type((ValueError, ConnectionError, Exception)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        "Ranking LLM call failed, retrying (attempt %s): %s",
        retry_state.attempt_number,
        retry_state.outcome.exception()
        if retry_state.outcome and retry_state.outcome.failed
        else None,
    ),
)
def rank_candidate_from_profile(
    profile: CandidateProfile,
    job_description_raw: str,
) -> RankingResult:
    """
    Compare a candidate profile to a job description and return fit score + reasoning.

    Args:
        profile: Structured CandidateProfile from extract_candidate_profile.
        job_description_raw: Full job description text.

    Returns:
        RankingResult with fit_score (0-100) and reasoning.

    Raises:
        ValueError: If the LLM returns invalid ranking result.
        ConnectionError: If Ollama server is unreachable.
        Exception: For other LLM invocation failures.
    """
    if not job_description_raw or not job_description_raw.strip():
        logger.warning("Empty job description; returning minimal score.")
        return RankingResult(
            fit_score=0,
            reasoning="No job description provided.",
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RANKING_SYSTEM_PROMPT),
            ("human", RANKING_USER_TEMPLATE),
        ]
    )
    chain = prompt | _get_ranking_llm()

    logger.info("Calling LLM for ranking (model=%s).", OLLAMA_MODEL)
    try:
        result: RankingResult = chain.invoke(
            {
                "job_description": job_description_raw.strip(),
                "skills": ", ".join(profile.skills)
                if profile.skills
                else "None listed",
                "years_of_experience": profile.years_of_experience,
                "past_titles": ", ".join(profile.past_titles)
                if profile.past_titles
                else "None listed",
            }
        )
    except Exception as e:
        logger.exception("Ranking invocation failed: %s", e)
        raise

    if not isinstance(result, RankingResult):
        raise ValueError(f"Expected RankingResult, got {type(result)}")

    logger.info("Ranking result: fit_score=%s", result.fit_score)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _usage() -> str:
    return (
        "Usage: python ranking_engine.py [RESUME_TEXT]\n"
        "  If RESUME_TEXT is omitted, reads resume text from stdin.\n"
        "  Env: OLLAMA_BASE_URL (default http://localhost:11434), OLLAMA_MODEL (default deepseek-r1:8b), LOG_LEVEL."
    )


def main() -> None:
    """CLI entry point for testing extraction directly."""
    if "--help" in sys.argv or "-h" in sys.argv:
        print(_usage())
        return

    if len(sys.argv) > 1:
        raw_text = " ".join(sys.argv[1:])
    else:
        raw_text = sys.stdin.read()

    try:
        profile = extract_candidate_profile(raw_text)
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        sys.exit(1)

    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
