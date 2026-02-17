"""
Build aggregated issue profile for a member by analyzing all press releases.

This script processes press releases sequentially, using an LLM to iteratively
update stance information across all issues defined in CandidateIssueProfile.

Usage:
    uv run python RAG/build_issue_profile.py --bioguide-id B001316
    uv run python RAG/build_issue_profile.py --bioguide-id B001316 --max-prs 40

Output:
    data/stances/{bioguide_id}_issue_profile.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.memberOpinions import CandidateIssueProfile

# Load environment
load_dotenv()

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STANCES_OUTPUT_DIR = DATA_DIR / "stances"
PRESS_RELEASES_FILE = DATA_DIR / "press_releases_by_bioguide.json"
CONGRESS_MEMBERS_FILE = DATA_DIR / "congress_members.json"

# Archia API configuration
ARCHIA_API_KEY = os.getenv("ARCHIA")
ARCHIA_BASE_URL = os.getenv("ARCHIA_BASE_URL", "https://api.archia.app/v1")
ARCHIA_MODEL = "gpt-5"

# Processing limits
DEFAULT_MAX_PRS = 40


def load_press_releases_data() -> Dict[str, Any]:
    """Load all press releases from consolidated JSON file."""
    if not PRESS_RELEASES_FILE.exists():
        raise FileNotFoundError(f"Press releases file not found: {PRESS_RELEASES_FILE}")

    with open(PRESS_RELEASES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("membersByBioguideId", {})


def load_member_name(bioguide_id: str) -> str:
    """Load member name from congress_members.json."""
    if not CONGRESS_MEMBERS_FILE.exists():
        return f"Member {bioguide_id}"

    with open(CONGRESS_MEMBERS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    members = data.get("members", [])
    for member in members:
        if member.get("bioguideId") == bioguide_id:
            return member.get("name", f"Member {bioguide_id}")

    return f"Member {bioguide_id}"


def call_archia_llm(prompt: str, max_retries: int = 3) -> str:
    """Call Archia API with prompt and return response text. Retries on timeout."""
    if not ARCHIA_API_KEY:
        raise ValueError("ARCHIA API key not found in environment")

    headers = {
        "x-api-key": ARCHIA_API_KEY.strip(),
        "Authorization": f"Bearer {ARCHIA_API_KEY.strip()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(http2=True, timeout=120.0, headers=headers) as client:
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
                    raise RuntimeError(f"Archia API error: {error_msg}")

                # Extract content from Archia response format
                content = None
                if "output" in result and isinstance(result["output"], list):
                    if result["output"] and "content" in result["output"][0]:
                        content_list = result["output"][0]["content"]
                        if content_list and isinstance(content_list, list):
                            content = content_list[0].get("text", "")

                if not content:
                    raise RuntimeError("No content in Archia response")

                return content

        except (httpx.TimeoutException, httpx.ReadTimeout) as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"    Timeout on attempt {attempt + 1}/{max_retries}, retrying...")
                continue
            else:
                raise RuntimeError(f"API timeout after {max_retries} attempts") from e

    # Should not reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Unexpected error in API call")


def build_update_prompt(
    pr_title: str,
    pr_body: str,
    pr_date: str,
    current_profile: CandidateIssueProfile,
) -> str:
    """Build minimal prompt for LLM to update issue profile."""

    # Convert current profile to dict
    current_state = current_profile.model_dump()

    prompt = f"""Analyze this press release and update the politician's issue profile.

CURRENT STATE:
{json.dumps(current_state, indent=2)}

PRESS RELEASE ({pr_date}):
{pr_title}

{pr_body[:4000]}

UPDATE RULES:
- For each issue CLEARLY addressed: update the summary text
- Make summaries comprehensive, incorporating all information seen across all PRs
- If an issue isn't addressed in this PR, keep its current summary
- Each summary should describe the politician's overall stance on that issue

Return ONLY valid JSON matching CandidateIssueProfile structure (24 issues as strings). No markdown."""

    return prompt


def process_member_prs(
    bioguide_id: str,
    press_releases: List[Dict[str, Any]],
    max_prs: int,
    verbose: bool = True
) -> CandidateIssueProfile:
    """Process press releases sequentially to build issue profile."""

    # Initialize empty profile
    profile = CandidateIssueProfile()

    # Limit number of PRs
    prs_to_process = press_releases[:max_prs]

    if verbose:
        print(f"Processing {len(prs_to_process)} press releases (max: {max_prs})...")
        print()

    # Process each PR
    for i, pr in enumerate(prs_to_process, 1):
        title = pr.get("title", "Untitled")
        body = pr.get("bodyText", "")
        date = pr.get("date", "Unknown")

        if verbose:
            print(f"[{i}/{len(prs_to_process)}] {date} - {title[:60]}...")

        # Build prompt
        prompt = build_update_prompt(
            pr_title=title,
            pr_body=body,
            pr_date=date,
            current_profile=profile,
        )

        try:
            # Call LLM
            response = call_archia_llm(prompt)

            if verbose:
                print("\n--- RAW LLM OUTPUT ---")
                print(response)
                print("--- END OUTPUT ---\n")

            # Extract JSON from response (handle markdown code blocks)
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()

            # Parse response
            try:
                updated_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"    JSON parse error: {e}")
                    print(f"    LLM response preview: {response[:200]}...")
                # Continue with previous state
                continue

            # Update profile
            new_profile = CandidateIssueProfile(**updated_data)

            if verbose:
                # Show which issues were updated
                changes = []
                old_dict = profile.model_dump()
                new_dict = new_profile.model_dump()

                for field_name in new_dict:
                    old_value = old_dict.get(field_name, "")
                    new_value = new_dict.get(field_name, "")

                    # Check if summary changed
                    if new_value != old_value and new_value:
                        changes.append(field_name)

                if changes:
                    print(f"    Updated: {', '.join(changes)}")
                else:
                    print(f"    No changes")

            # Store updated profile
            profile = new_profile

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            # Continue with previous state
            continue

        if verbose:
            print()

    return profile


def main():
    parser = argparse.ArgumentParser(
        description="Build aggregated issue profile from press releases",
        epilog="Uses LLM to iteratively update stance information across all issues"
    )

    parser.add_argument(
        "--bioguide-id",
        required=True,
        help="Member bioguide ID (e.g., B001316)"
    )
    parser.add_argument(
        "--max-prs",
        type=int,
        default=DEFAULT_MAX_PRS,
        help=f"Maximum press releases to process (default: {DEFAULT_MAX_PRS})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STANCES_OUTPUT_DIR,
        help=f"Output directory (default: {STANCES_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Load press releases
    if verbose:
        print("Loading press releases data...")
    press_releases_data = load_press_releases_data()

    # Get member data
    member_data = press_releases_data.get(args.bioguide_id)
    if not member_data:
        print(f"Error: No press releases found for {args.bioguide_id}")
        return

    press_releases = member_data.get("pressReleases", [])
    if not press_releases:
        print(f"Error: Member {args.bioguide_id} has no press releases")
        return

    member_name = load_member_name(args.bioguide_id)

    if verbose:
        print(f"Member: {member_name} ({args.bioguide_id})")
        print(f"Total press releases available: {len(press_releases)}")
        print()
        print("=" * 80)
        print("BUILDING ISSUE PROFILE")
        print("=" * 80)
        print()

    # Process press releases
    try:
        profile = process_member_prs(
            bioguide_id=args.bioguide_id,
            press_releases=press_releases,
            max_prs=args.max_prs,
            verbose=verbose
        )

        # Save output
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = args.output_dir / f"{args.bioguide_id}_issue_profile.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(profile.model_dump(), f, indent=2, ensure_ascii=False)

        if verbose:
            print("=" * 80)
            print("FINAL PROFILE SUMMARY")
            print("=" * 80)

            # Count issues with summaries
            profile_dict = profile.model_dump()
            issues_with_summaries = []

            for issue_name, summary in profile_dict.items():
                if summary and summary.strip():
                    issues_with_summaries.append({
                        "issue": issue_name,
                        "summary": summary[:150] + "..." if len(summary) > 150 else summary
                    })

            print(f"\nIssues with stances: {len(issues_with_summaries)}/24")
            print()

            for item in issues_with_summaries:
                print(f"{item['issue']}:")
                print(f"  {item['summary']}")
                print()

        print(f"✓ Saved to: {output_file}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
