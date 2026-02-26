"""
High-level agent for candidate ranking orchestration.

Provides a simple interface for extracting candidate profiles and ranking
them against job descriptions. Delegates to ranking_engine.py for LLM calls.
"""

from __future__ import annotations

import logging

from models import CandidateProfile, RankingResult
from ranking_engine import extract_candidate_profile, rank_candidate_from_profile

logger = logging.getLogger(__name__)


class CandidateAgent:
    """
    Orchestration layer for candidate ranking operations.

    Provides high-level methods for:
    - Extracting structured profiles from raw resumes
    - Ranking candidates against job descriptions
    - End-to-end pipeline from resume text to fit score
    """

    def __init__(self) -> None:
        """Initialize the candidate ranking agent."""
        logger.info("Initialized CandidateAgent")

    def extract_profile(self, resume_text: str) -> CandidateProfile:
        """
        Extract structured profile from raw resume text.

        Args:
            resume_text: Raw text from resume (PDF, plain text, etc.)

        Returns:
            CandidateProfile with skills, experience, and job titles.

        Raises:
            ValueError: If extraction fails or returns invalid data.
            ConnectionError: If LLM service is unreachable.
        """
        logger.info("Extracting profile from resume text (%d chars)", len(resume_text))
        return extract_candidate_profile(resume_text)

    def rank_candidate(
        self,
        job_description: str,
        resume_text: str,
    ) -> RankingResult:
        """
        Rank candidate's fit for a job (end-to-end pipeline).

        Extracts profile from resume, then ranks against job description.

        Args:
            job_description: Full job description text.
            resume_text: Raw resume text.

        Returns:
            RankingResult with fit_score (0-100) and reasoning.

        Raises:
            ValueError: If extraction or ranking fails.
            ConnectionError: If LLM service is unreachable.
        """
        logger.info(
            "Ranking candidate: JD %d chars, resume %d chars",
            len(job_description),
            len(resume_text),
        )

        # Step 1: Extract structured profile
        profile = self.extract_profile(resume_text)

        # Step 2: Rank against job description
        result = rank_candidate_from_profile(profile, job_description)

        logger.info(
            "Ranking complete: fit_score=%d, profile had %d skills, %.1f years exp",
            result.fit_score,
            len(profile.skills),
            profile.years_of_experience,
        )

        return result

    def rank_from_profile(
        self,
        job_description: str,
        profile: CandidateProfile,
    ) -> RankingResult:
        """
        Rank a pre-extracted candidate profile against a job description.

        Use this if you already have a CandidateProfile (e.g., from cache).

        Args:
            job_description: Full job description text.
            profile: Pre-extracted candidate profile.

        Returns:
            RankingResult with fit_score (0-100) and reasoning.

        Raises:
            ValueError: If ranking fails.
            ConnectionError: If LLM service is unreachable.
        """
        logger.info(
            "Ranking from pre-extracted profile: %d skills, %.1f years exp",
            len(profile.skills),
            profile.years_of_experience,
        )
        return rank_candidate_from_profile(profile, job_description)
