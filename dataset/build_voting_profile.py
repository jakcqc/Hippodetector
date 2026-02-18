"""
Build voting profiles from consolidated bills data.

PURPOSE:
    Produces minimal VotingProfile objects optimized for LLM stance extraction.
    The VotingProfile format contains only essential fields (bill title, summary,
    vote cast, date) to minimize input tokens when feeding to LLMs for stance
    analysis. This avoids the overhead of full bill details and API responses.

WORKFLOW:
    1. Extract House votes from congress_bill_summaries_last_10_years.zip
    2. Filter for House chamber votes only (700 bills, 1,015 vote events)
    3. Use bioguideId directly from vote records (no name matching needed)
    4. Create VotingProfile with minimal but sufficient context
    5. Save to data/voting_profiles/ for downstream LLM processing

OUTPUT FORMAT:
    VotingProfile {
        bioguide_id: str
        name: str
        congress: int
        total_votes: int
        votes: [VoteRecord {bill_id, title, summary, vote, date}]
    }

Usage:
    # Single member
    uv run python dataset/build_voting_profile.py --bioguide-id B001316

    # All members in one pass
    uv run python dataset/build_voting_profile.py --all
"""

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from votingProfile import VotingProfile, VoteRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VOTING_PROFILES_DIR = DATA_DIR / "voting_profiles"
BILLS_ZIP = DATA_DIR / "congress_bill_summaries_last_10_years.zip"


def get_member_info(bioguide_id: str, name: str, party: str, state: str) -> Dict[str, str]:
    """Create member info dict from vote record data."""
    return {
        "bioguide_id": bioguide_id,
        "name": name,
        "state": state,
        "party": party,
    }


def load_bills_data() -> Dict[str, Any]:
    """Load bills data from zip file."""
    if not BILLS_ZIP.exists():
        raise FileNotFoundError(f"Bills zip file not found: {BILLS_ZIP}")

    print(f"Loading bills data from {BILLS_ZIP.name}...")
    with zipfile.ZipFile(BILLS_ZIP, 'r') as zf:
        # Get the JSON file name from zip
        json_files = [f for f in zf.namelist() if f.endswith('.json')]
        if not json_files:
            raise ValueError("No JSON file found in zip")

        json_file = json_files[0]
        with zf.open(json_file) as f:
            data = json.load(f)

    print(f"Loaded {len(data.get('bills', []))} bills")
    return data


def extract_bill_summary(summaries: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the most recent bill summary text."""
    if not summaries:
        return None

    # Summaries are usually ordered by date, take the most recent with text
    for summary in reversed(summaries):
        text = summary.get("text", "")
        if text:
            # Strip HTML tags if present
            import re
            text = re.sub(r'<[^>]+>', '', text)
            return text.strip()

    return None


def build_all_profiles(
    bills_data: Dict[str, Any],
    target_bioguide: Optional[str] = None
) -> Dict[str, VotingProfile]:
    """
    Build voting profiles for all House members (or single member if specified).

    Args:
        bills_data: Loaded bills data from zip
        target_bioguide: Optional specific bioguide ID to extract

    Returns:
        Dict of bioguide_id -> VotingProfile
    """
    # Initialize vote collection for each member
    member_votes = defaultdict(list)
    member_info_cache = {}  # bioguide_id -> {name, party, state}

    # Single pass through all bills
    bills = bills_data.get("bills", [])
    print(f"Processing {len(bills)} bills...")

    for i, bill in enumerate(bills, 1):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(bills)} bills...")

        # Build bill identifier
        congress = bill.get("congress", "")
        bill_type = bill.get("billType", "")
        bill_number = bill.get("billNumber", "")
        bill_id = f"{congress}-{bill_type}-{bill_number}"

        title = bill.get("title", "")
        summary = extract_bill_summary(bill.get("summaries", []))

        # Process all vote events for this bill (filter for House only)
        for vote_event in bill.get("votes", []):
            # Skip non-House votes
            if vote_event.get("chamber") != "House":
                continue

            vote_date = vote_event.get("recordedAt", "")

            # Check each member's vote
            for member_vote in vote_event.get("members", []):
                bioguide_id = member_vote.get("bioguideId", "")

                # Skip if no bioguide ID
                if not bioguide_id:
                    continue

                # If filtering for specific member, skip others
                if target_bioguide and bioguide_id != target_bioguide:
                    continue

                # Cache member info on first encounter
                if bioguide_id not in member_info_cache:
                    member_info_cache[bioguide_id] = {
                        "name": member_vote.get("name", ""),
                        "party": member_vote.get("party", ""),
                        "state": member_vote.get("state", ""),
                    }

                vote_record = VoteRecord(
                    bill_id=bill_id,
                    title=title,
                    summary=summary,
                    subjects=[],  # Not available in this dataset
                    vote=member_vote.get("voteCast", ""),
                    date=vote_date
                )
                member_votes[bioguide_id].append(vote_record)

    # Build VotingProfile objects
    profiles = {}
    for bioguide_id, votes in member_votes.items():
        if not votes:
            continue

        member_info = member_info_cache[bioguide_id]

        # Infer congress from votes (use most common)
        congresses = [int(v.bill_id.split('-')[0]) for v in votes if '-' in v.bill_id]
        congress = max(set(congresses), key=congresses.count) if congresses else 119

        profile = VotingProfile(
            bioguide_id=bioguide_id,
            name=member_info["name"],
            congress=congress,
            total_votes=len(votes),
            votes=votes
        )

        profiles[bioguide_id] = profile

    return profiles


def save_profile(profile: VotingProfile, output_dir: Path) -> None:
    """Save voting profile to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{profile.bioguide_id}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(profile.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"  Saved: {output_file.name} ({profile.total_votes} votes)")


def main():
    parser = argparse.ArgumentParser(
        description="Build voting profiles from consolidated bills data"
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--bioguide-id",
        type=str,
        help="Build profile for single member (e.g., B001316)"
    )
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Build profiles for all members in one pass"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VOTING_PROFILES_DIR,
        help=f"Output directory (default: {VOTING_PROFILES_DIR})"
    )

    args = parser.parse_args()

    # Load data
    bills_data = load_bills_data()

    # Build profiles
    target_bioguide = args.bioguide_id if not args.all else None
    profiles = build_all_profiles(bills_data, target_bioguide)

    # Save profiles
    print(f"\nSaving {len(profiles)} profile(s) to {args.output_dir}...")
    for profile in profiles.values():
        save_profile(profile, args.output_dir)

    print(f"\n✓ Complete! Generated {len(profiles)} voting profile(s)")

    # Summary statistics
    if profiles:
        total_votes = sum(p.total_votes for p in profiles.values())
        avg_votes = total_votes / len(profiles)
        print(f"  Total votes across all profiles: {total_votes}")
        print(f"  Average votes per member: {avg_votes:.0f}")


if __name__ == "__main__":
    main()
