"""
Extract structured stances from press releases using LLM.

This module uses Claude via Archia API to analyze press releases
and extract politician stances on specific issues.

Usage:
    from RAG.extract_stances import extract_stance_from_text

    stance = extract_stance_from_text(
        text="Press release about healthcare...",
        issue_category="health_care"
    )
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Add parent directory to path to import schemas
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.memberOpinions import IssueStance, StanceStatus

# Load environment variables
load_dotenv()

# LLM Provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "archia")  # "archia" or "gemini"

# Archia API configuration
ARCHIA_API_KEY = os.getenv("ARCHIA")
ARCHIA_BASE_URL = os.getenv("ARCHIA_BASE_URL", "https://api.archia.app/v1")
ARCHIA_MODEL = os.getenv("ARCHIA_MODEL_ANTHROPIC", "priv-claude-3-5-haiku-20241022")

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

# Initialize Gemini client if available
_gemini_client = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Issue categories from memberOpinions.py
ISSUE_CATEGORIES = [
    "abortion",
    "budget_economy",
    "civil_rights",
    "corporations",
    "crime",
    "drugs",
    "education",
    "energy_oil",
    "environment",
    "families_children",
    "foreign_policy",
    "free_trade",
    "government_reform",
    "gun_control",
    "health_care",
    "homeland_security",
    "immigration",
    "jobs",
    "principles_values",
    "social_security",
    "tax_reform",
    "technology",
    "war_peace",
    "welfare_poverty",
]

# NOTE: Bill subject mapping moved to RAG/topic_matching.py
# Use: from RAG.topic_matching import map_bill_subjects_to_issues


def _extract_stance_archia(
    text: str,
    issue_category: str,
    source_url: Optional[str],
    prompt: str,
) -> IssueStance:
    """Extract stance using Archia API."""
    if not ARCHIA_API_KEY:
        raise ValueError("ARCHIA API key not found in environment")

    try:
        # Call Archia API
        headers = {
            "x-api-key": ARCHIA_API_KEY.strip(),
            "Authorization": f"Bearer {ARCHIA_API_KEY.strip()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        with httpx.Client(http2=True, timeout=30.0, headers=headers) as client:
            response = client.post(
                f"{ARCHIA_BASE_URL}/responses",
                json={
                    "model": ARCHIA_MODEL,
                    "input": prompt,
                }
            )
            response.raise_for_status()

        # Parse response
        result = response.json()

        # Check for API errors
        if result.get("status") == "failed":
            error = result.get("error", {})
            error_msg = error.get("message", "Unknown error")
            print(f"Archia API error: {error_msg}")
            return IssueStance(status="unknown", summary=None, source_url=source_url)

        # Extract content from Archia response format
        content = None
        if "output" in result and isinstance(result["output"], list):
            if result["output"] and "content" in result["output"][0]:
                content_list = result["output"][0]["content"]
                if content_list and isinstance(content_list, list):
                    content = content_list[0].get("text", "")

        if not content:
            print("No content in Archia response")
            return IssueStance(status="unknown", summary=None, source_url=source_url)

        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Parse JSON
        parsed = json.loads(content)

        return IssueStance(
            status=parsed.get("status", "unknown"),
            summary=parsed.get("summary"),
            source_url=source_url,
        )

    except Exception as e:
        print(f"Archia API error: {e}")
        return IssueStance(status="unknown", summary=None, source_url=source_url)


def _extract_stance_gemini(
    text: str,
    issue_category: str,
    source_url: Optional[str],
    prompt: str,
    max_tokens: int = 500,
) -> IssueStance:
    """Extract stance using Google Gemini API."""
    if not GEMINI_AVAILABLE:
        raise ValueError("google-genai package not installed. Run: uv pip install google-genai")
    if not _gemini_client:
        raise ValueError("GEMINI_API_KEY not found in environment")

    try:
        # Call Gemini API using new client
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "max_output_tokens": max_tokens,
                "temperature": 0.3,
            }
        )

        # Extract text from response
        content = response.text.strip()

        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Parse JSON
        parsed = json.loads(content)

        return IssueStance(
            status=parsed.get("status", "unknown"),
            summary=parsed.get("summary"),
            source_url=source_url,
        )

    except Exception as e:
        print(f"Gemini API error: {e}")
        return IssueStance(status="unknown", summary=None, source_url=source_url)


def extract_stance_from_text(
    text: str,
    issue_category: str,
    source_url: Optional[str] = None,
    max_tokens: int = 500,
) -> IssueStance:
    """
    Extract stance from text using configured LLM provider.

    Provider is selected via LLM_PROVIDER environment variable:
    - "archia" (default): Uses Archia API with Claude
    - "gemini": Uses Google Gemini API

    Args:
        text: Press release or statement text
        issue_category: Issue category to analyze (e.g., "health_care")
        source_url: Optional URL of the source
        max_tokens: Max tokens for LLM response

    Returns:
        IssueStance object with extracted stance
    """
    # Build prompt
    prompt = f"""Analyze this press release and extract the politician's stance on {issue_category.replace('_', ' ')}.

Press Release Text:
{text[:3000]}  # Limit text to avoid token limits

Determine:
1. Does this text express a clear position on {issue_category.replace('_', ' ')}?
2. If yes, what is their stance: supports, opposes, or mixed?
3. Provide a 1-2 sentence summary of their position.

Respond ONLY with valid JSON in this exact format:
{{
    "status": "supports" | "opposes" | "mixed" | "no_stance_recorded",
    "summary": "Brief summary of their position (or null if no stance)"
}}

If the text doesn't clearly address {issue_category.replace('_', ' ')}, return status "no_stance_recorded" with summary null.
Do not include any markdown formatting or code blocks, just the raw JSON."""

    # Route to appropriate provider
    if LLM_PROVIDER == "gemini":
        return _extract_stance_gemini(text, issue_category, source_url, prompt, max_tokens)
    else:
        return _extract_stance_archia(text, issue_category, source_url, prompt)


def extract_stances_from_press_release(
    title: str,
    body_text: str,
    url: Optional[str] = None,
    target_issues: Optional[List[str]] = None,
) -> dict[str, IssueStance]:
    """
    Extract stances on multiple issues from a single press release.

    Args:
        title: Press release title
        body_text: Press release body text
        url: Press release URL
        target_issues: List of issue categories to analyze (if None, analyzes all)

    Returns:
        Dictionary mapping issue categories to IssueStance objects
    """
    full_text = f"{title}\n\n{body_text}"

    # If no target issues specified, try to detect relevant ones
    if target_issues is None:
        # For now, just analyze a few common categories
        # In production, you might want smarter detection
        target_issues = ["health_care", "budget_economy", "immigration", "government_reform"]

    stances = {}
    for issue in target_issues:
        if issue not in ISSUE_CATEGORIES:
            print(f"Warning: '{issue}' is not a valid issue category")
            continue

        stance = extract_stance_from_text(
            text=full_text,
            issue_category=issue,
            source_url=url,
        )

        # Only include if a stance was found
        if stance.status != "no_stance_recorded":
            stances[issue] = stance

    return stances


def main():
    """Test stance extraction with example text."""
    # Example press release text
    test_text = """
    Burlison Announces Hearing on Making Housing More Affordable

    WASHINGTON — Subcommittee on Economic Growth, Energy Policy, and Regulatory Affairs
    Chairman Eric Burlison (R-Mo.) today announced a hearing on "Housing Affordability:
    Saving the American Dream."

    During the hearing, Chairman Burlison will examine the federal regulatory burden
    that is driving up housing costs for Americans. Years of Democratic leadership
    imposed burdensome federal regulations that have increased costs for families
    looking to purchase their first home or move into a bigger house.

    "For far too long, the federal government has imposed unnecessary regulations
    that drive up housing costs," said Chairman Burlison. "We need to remove these
    barriers and let the free market work."
    """

    print("=" * 80)
    print("Stance Extraction Test")
    print("=" * 80)

    # Test single issue extraction
    print("\nExtracting stance on 'government_reform'...")
    stance = extract_stance_from_text(
        text=test_text,
        issue_category="government_reform",
        source_url="https://example.com/press-release"
    )

    print(f"\nResult:")
    print(f"  Status: {stance.status}")
    print(f"  Summary: {stance.summary}")
    print(f"  Source: {stance.source_url}")

    print("\n" + "=" * 80)
    print("NOTE: Bill subject mapping has moved to RAG/topic_matching.py")
    print("Run: python RAG/topic_matching.py to test topic matching")
    print("=" * 80)


if __name__ == "__main__":
    main()
