"""
Build aggregated issue profile for a member by analyzing voting records.

This script processes voting records in batches, using an LLM to iteratively
update stance information across all issues defined in CandidateIssueProfile.

IMPORTANT: Bill summaries are truncated to 30,000 characters (~7,500 tokens) to
prevent token explosion. Analysis shows the longest bill summary is 300,596 chars
(~75,149 tokens). The 30,000 char limit (40% of max) preserves substantial detail
from even the longest bills while managing costs (median summary is only 290 tokens).

Usage:
    # Single member
    uv run python RAG/build_voting_issue_profile.py --bioguide-id C001136
    uv run python RAG/build_voting_issue_profile.py --bioguide-id C001136 --max-votes 30

    # Batch from file
    uv run python RAG/build_voting_issue_profile.py --from-file sample_politicians.txt
    uv run python RAG/build_voting_issue_profile.py --from-file sample_politicians.txt --max-votes 20

Output:
    data/stances/{bioguide_id}_voting_issue_profile.json

Cost Estimates (GPT-4 pricing, ~$0.01/1K input, ~$0.03/1K output):
    - 10 votes (default): ~$0.25 per member
    - 30 votes: ~$0.83 per member
    - 30 people (30 votes each): ~$25.00 total
    - All votes (~238 avg): ~$6.50 per member
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
from dataset.votingProfile import VotingProfile

# Load environment
load_dotenv()

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VOTING_PROFILES_DIR = DATA_DIR / "voting_profiles"
STANCES_OUTPUT_DIR = DATA_DIR / "stances"
CONGRESS_MEMBERS_FILE = DATA_DIR / "congress_members.json"

# Archia API configuration
ARCHIA_API_KEY = os.getenv("ARCHIA")
ARCHIA_BASE_URL = os.getenv("ARCHIA_BASE_URL", "https://api.archia.app/v1")
ARCHIA_MODEL = "gpt-5"

# Gemini API configuration (fallback)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Processing limits
DEFAULT_MAX_VOTES = 10
DEFAULT_BATCH_SIZE = 3  # Reduced from 5 to avoid API timeouts
SUMMARY_TRUNCATE_CHARS = 30000  # ~7,500 tokens (40% of longest bill summary)


def load_voting_profile(bioguide_id: str) -> VotingProfile:
    """Load voting profile from JSON file."""
    profile_file = VOTING_PROFILES_DIR / f"{bioguide_id}.json"

    if not profile_file.exists():
        raise FileNotFoundError(f"Voting profile not found: {profile_file}")

    with open(profile_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return VotingProfile(**data)


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


def call_gemini_llm(prompt: str) -> str:
    """Call Gemini API with prompt and return response text."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment")

    try:
        import google.genai as genai

        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if response.text:
            return response.text
        else:
            raise RuntimeError("No content in Gemini response")

    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}") from e


def call_archia_llm(prompt: str, max_retries: int = 3) -> str:
    """Call Archia API with prompt and return response text. Retries on timeout and server errors."""
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
            with httpx.Client(http2=True, timeout=180.0, headers=headers) as client:
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
                print(f"    Timeout on attempt {attempt + 1}/{max_retries}, retrying in 5s...")
                import time
                time.sleep(5)
                continue
            else:
                raise RuntimeError(f"API timeout after {max_retries} attempts") from e

        except httpx.HTTPStatusError as e:
            # Retry on 5xx server errors (including 504 Gateway Timeout)
            if e.response.status_code >= 500:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"    Server error {e.response.status_code} on attempt {attempt + 1}/{max_retries}, retrying in 10s...")
                    import time
                    time.sleep(10)
                    continue
                else:
                    raise RuntimeError(f"Server error after {max_retries} attempts: {e}") from e
            else:
                # Don't retry on 4xx client errors
                raise

    # Should not reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Unexpected error in API call")


def call_llm_with_fallback(prompt: str) -> str:
    """Call LLM with Archia first, fallback to Gemini if Archia fails."""
    # Try Archia first
    try:
        if ARCHIA_API_KEY:
            return call_archia_llm(prompt)
        else:
            print("    Archia API key not found, using Gemini fallback...")
    except Exception as e:
        print(f"    Archia failed: {e}")
        print("    Falling back to Gemini...")

    # Fallback to Gemini
    try:
        return call_gemini_llm(prompt)
    except Exception as e:
        raise RuntimeError(f"Both Archia and Gemini failed. Last error: {e}") from e


def truncate_summary(summary: str, max_chars: int = SUMMARY_TRUNCATE_CHARS) -> str:
    """Truncate summary to prevent token explosion."""
    if not summary or len(summary) <= max_chars:
        return summary
    return summary[:max_chars] + "..."


def build_update_prompt(
    vote_batch: List[Dict[str, Any]],
    current_profile: CandidateIssueProfile,
) -> str:
    """Build prompt for LLM to update issue profile based on voting batch."""

    # Convert current profile to dict
    current_state = current_profile.model_dump()

    # Format votes for prompt
    votes_text = []
    for i, vote in enumerate(vote_batch, 1):
        bill_id = vote.get("bill_id", "")
        title = vote.get("title", "")
        summary = truncate_summary(vote.get("summary", ""))
        vote_cast = vote.get("vote", "")
        date = vote.get("date", "")

        vote_text = f"""VOTE {i}:
Bill: {bill_id}
Date: {date}
Title: {title}
Summary: {summary}
Vote Cast: {vote_cast}
"""
        votes_text.append(vote_text)

    votes_section = "\n".join(votes_text)

    prompt = f"""Analyze these voting records and update the politician's issue profile.

CURRENT STATE:
{json.dumps(current_state, indent=2)}

VOTING RECORDS ({len(vote_batch)} votes):
{votes_section}

UPDATE RULES:
- For each issue CLEARLY addressed by these votes: update the summary text
- Make summaries comprehensive, incorporating all information seen across all votes
- If an issue isn't addressed in these votes, keep its current summary
- Each summary should describe the politician's overall stance on that issue
- Consider the vote cast (Yea/Nay) in context of what the bill does

Return ONLY valid JSON matching CandidateIssueProfile structure (24 issues as strings). No markdown."""

    return prompt


def process_voting_records(
    bioguide_id: str,
    voting_profile: VotingProfile,
    max_votes: int,
    batch_size: int,
    verbose: bool = True
) -> CandidateIssueProfile:
    """Process voting records in batches to build issue profile."""

    # Initialize empty profile
    profile = CandidateIssueProfile()

    # Filter votes with summaries and limit
    votes_with_summaries = [
        v.model_dump() for v in voting_profile.votes
        if v.summary
    ]
    votes_to_process = votes_with_summaries[:max_votes]

    if verbose:
        print(f"Processing {len(votes_to_process)} votes (max: {max_votes}, batch size: {batch_size})...")
        print()

    # Process in batches
    num_batches = (len(votes_to_process) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(votes_to_process))
        batch = votes_to_process[start_idx:end_idx]

        if verbose:
            print(f"[Batch {batch_idx + 1}/{num_batches}] Processing votes {start_idx + 1}-{end_idx}...")

        # Build prompt
        prompt = build_update_prompt(
            vote_batch=batch,
            current_profile=profile,
        )

        try:
            # Call LLM (with fallback)
            response = call_llm_with_fallback(prompt)

            if verbose:
                print("\n--- RAW LLM OUTPUT ---")
                print(response[:500] + "..." if len(response) > 500 else response)
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


def parse_politician_file(file_path: Path) -> List[str]:
    """Parse politician file and extract bioguide IDs.

    Expected format: LastName, FirstName MiddleInitial. (BIOGUIDE_ID)
    Example: Conaway, Herbert C. (C001136)
    """
    bioguide_ids = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Extract bioguide ID from parentheses
            if '(' in line and ')' in line:
                bioguide_id = line.split('(')[1].split(')')[0].strip()
                if bioguide_id:
                    bioguide_ids.append(bioguide_id)

    return bioguide_ids


def main():
    parser = argparse.ArgumentParser(
        description="Build aggregated issue profile from voting records",
        epilog="Uses LLM to iteratively update stance information across all issues"
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--bioguide-id",
        help="Single member bioguide ID (e.g., C001136)"
    )
    input_group.add_argument(
        "--from-file",
        type=Path,
        help="File containing politician names and bioguide IDs (e.g., sample_politicians.txt)"
    )
    parser.add_argument(
        "--max-votes",
        type=int,
        default=DEFAULT_MAX_VOTES,
        help=f"Maximum votes to process (default: {DEFAULT_MAX_VOTES})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Votes per API call (default: {DEFAULT_BATCH_SIZE})"
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

    # Determine bioguide IDs to process
    if args.from_file:
        bioguide_ids = parse_politician_file(args.from_file)
        if verbose:
            print(f"Loaded {len(bioguide_ids)} members from {args.from_file}")
            print()
    else:
        bioguide_ids = [args.bioguide_id]

    # Process each member
    success_count = 0
    error_count = 0

    for idx, bioguide_id in enumerate(bioguide_ids, 1):
        if len(bioguide_ids) > 1 and verbose:
            print(f"\n{'=' * 80}")
            print(f"PROCESSING MEMBER {idx}/{len(bioguide_ids)}: {bioguide_id}")
            print(f"{'=' * 80}\n")

        try:
            # Load voting profile
            if verbose:
                print("Loading voting profile...")
            voting_profile = load_voting_profile(bioguide_id)

            member_name = load_member_name(bioguide_id)

            if verbose:
                print(f"Member: {member_name} ({bioguide_id})")
                print(f"Total votes in profile: {voting_profile.total_votes}")
                votes_with_summaries = len([v for v in voting_profile.votes if v.summary])
                print(f"Votes with summaries: {votes_with_summaries}")
                print()
                print("=" * 80)
                print("BUILDING ISSUE PROFILE FROM VOTING RECORDS")
                print("=" * 80)
                print()

            # Process voting records
            profile = process_voting_records(
                bioguide_id=bioguide_id,
                voting_profile=voting_profile,
                max_votes=args.max_votes,
                batch_size=args.batch_size,
                verbose=verbose
            )

            # Save output
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_file = args.output_dir / f"{bioguide_id}_voting_issue_profile.json"

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
            success_count += 1

        except Exception as e:
            print(f"\n✗ Error processing {bioguide_id}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            error_count += 1
            continue

    # Final summary for batch processing
    if len(bioguide_ids) > 1:
        print(f"\n{'=' * 80}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'=' * 80}")
        print(f"Successful: {success_count}/{len(bioguide_ids)}")
        print(f"Errors: {error_count}/{len(bioguide_ids)}")

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
