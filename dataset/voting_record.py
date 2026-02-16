"""
Script to fetch a politician's voting record from the Congress API.

Usage:
    python voting_record.py --bioguide-id B000944
    python voting_record.py --name "Sherrod Brown"
    python voting_record.py --bioguide-id B000944 --congress 118
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


BASE_URL = "https://api.congress.gov/v3"
DEFAULT_LIMIT = 250


def load_env_file(env_path: Path) -> None:
    """Load environment variables from a .env file."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def ensure_api_key() -> str:
    """Get the Congress API key from environment variables."""
    key = (
        os.getenv("CONGRESS_API")
        or os.getenv("CONGRES_API")
        or os.getenv("CONGRESS_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "Missing Congress API key. Set CONGRESS_API in your environment or .env file."
        )
    return key


def with_api_key(url: str, api_key: str) -> str:
    """Add API key to URL query parameters."""
    parsed = urlparse(url)
    params: Dict[str, str] = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params["api_key"] = api_key
    updated_query = urlencode(params)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            updated_query,
            parsed.fragment,
        )
    )


def request_json(url: str) -> Dict:
    """Make a GET request and return JSON response."""
    request = Request(
        url,
        headers={
            "User-Agent": "Hippodetector/1.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def search_member_by_name(name: str, api_key: str) -> Optional[Dict]:
    """
    Search for a member by name.
    Returns the first matching member or None.
    """
    url = f"{BASE_URL}/member?format=json&limit=250"
    payload = request_json(with_api_key(url, api_key))

    members = payload.get("members", [])
    name_lower = name.lower()

    # Try to find exact or close match
    for member in members:
        member_name = member.get("name", "").lower()
        if name_lower in member_name or member_name in name_lower:
            return member

    return None


def fetch_house_votes_for_congress(
    congress: int,
    api_key: str,
    limit: int = DEFAULT_LIMIT
) -> List[Dict]:
    """Fetch all House roll call votes for a specific congress."""
    votes_url = f"{BASE_URL}/house-vote/{congress}?format=json&limit={limit}"
    votes: List[Dict] = []
    next_url = votes_url

    print(f"  Fetching House votes for Congress {congress}...")
    while next_url:
        payload = request_json(with_api_key(next_url, api_key))
        votes.extend(payload.get("houseRollCallVotes", []))
        next_url = payload.get("pagination", {}).get("next")
        print(f"    Fetched {len(votes)} votes so far...")

    return votes


def fetch_member_votes_for_roll_call(
    congress: int,
    session: int,
    roll_call: int,
    api_key: str
) -> List[Dict]:
    """Fetch all member votes for a specific House roll call vote."""
    url = f"{BASE_URL}/house-vote/{congress}/{session}/{roll_call}/members?format=json&limit=250"
    all_results: List[Dict] = []
    next_url = url

    while next_url:
        payload = request_json(with_api_key(next_url, api_key))
        vote_data = payload.get("houseRollCallVoteMemberVotes", {})
        all_results.extend(vote_data.get("results", []))
        next_url = payload.get("pagination", {}).get("next")

    return all_results


def fetch_member_votes(
    bioguide_id: str,
    api_key: str,
    congress: Optional[int] = None,
    chamber: str = "house",
    max_votes: Optional[int] = None
) -> Tuple[List[Dict], Dict]:
    """
    Fetch all votes for a specific member.

    Args:
        bioguide_id: The bioguide ID of the member (e.g., 'B000944')
        api_key: Congress API key
        congress: Optional congress number to filter by (e.g., 119 for 119th Congress)
        chamber: Chamber to fetch votes for ("house" or "senate")
        max_votes: Maximum number of roll call votes to check (for faster results)

    Returns:
        Tuple of (list of votes, member info dict)
    """
    # First, get member info
    member_url = f"{BASE_URL}/member/{bioguide_id}?format=json"
    member_data = request_json(with_api_key(member_url, api_key))
    member_info = member_data.get("member", {})

    # Determine congress if not specified
    if not congress:
        # Get the latest term to determine current congress
        terms = member_info.get("terms", {}).get("item", [])
        if terms:
            congress = terms[-1].get("congress")

    if not congress:
        raise ValueError("Could not determine congress number for member")

    # Only House votes are supported currently (beta API)
    if chamber.lower() != "house":
        print("Note: Only House votes are currently supported in the Congress.gov API")
        return [], member_info

    # Fetch all House roll call votes for the congress
    all_votes = fetch_house_votes_for_congress(congress, api_key)

    # Limit the number of votes to check if specified
    if max_votes and max_votes < len(all_votes):
        # Take the most recent votes (they're sorted by date)
        all_votes = all_votes[:max_votes]
        print(f"  Limited to checking the {max_votes} most recent roll call votes")

    # Now fetch member votes for each roll call
    member_votes = []
    print(f"\n  Checking {len(all_votes)} roll call votes for member {bioguide_id}...")

    for i, vote in enumerate(all_votes, 1):
        if i % 25 == 0:
            print(f"    Processed {i}/{len(all_votes)} votes...")

        session = vote.get("sessionNumber")
        roll_call = vote.get("rollCallNumber")

        # Fetch member votes for this roll call
        member_vote_results = fetch_member_votes_for_roll_call(
            congress, session, roll_call, api_key
        )

        # Find our member's vote
        for member_vote in member_vote_results:
            if member_vote.get("bioguideID") == bioguide_id:
                # Combine vote info with roll call info
                combined_vote = {
                    "congress": congress,
                    "session": session,
                    "rollCall": roll_call,
                    "date": vote.get("startDate"),
                    "legislation": {
                        "type": vote.get("legislationType") or vote.get("amendmentType"),
                        "number": vote.get("legislationNumber") or vote.get("amendmentNumber"),
                        "url": vote.get("legislationUrl"),
                    },
                    "question": vote.get("voteQuestion", ""),
                    "result": vote.get("result"),
                    "voteType": vote.get("voteType"),
                    "memberVote": member_vote.get("voteCast"),
                    "party": member_vote.get("voteParty"),
                    "state": member_vote.get("voteState"),
                }
                member_votes.append(combined_vote)
                break

    return member_votes, member_info


def format_vote_summary(votes: List[Dict]) -> Dict[str, int]:
    """Generate a summary of vote positions."""
    summary = {
        "Yea": 0,
        "Nay": 0,
        "Aye": 0,
        "No": 0,
        "Present": 0,
        "Not Voting": 0,
        "Other": 0,
    }

    for vote in votes:
        position = vote.get("memberVote", "Other")
        if position in summary:
            summary[position] += 1
        else:
            summary["Other"] += 1

    return summary


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch a politician's voting record from Congress API"
    )
    parser.add_argument(
        "--bioguide-id",
        type=str,
        help="Bioguide ID of the member (e.g., B000944 for Sherrod Brown)",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the member to search for",
    )
    parser.add_argument(
        "--congress",
        type=int,
        help="Congress number to filter by (e.g., 119 for 119th Congress). Defaults to member's current congress.",
    )
    parser.add_argument(
        "--chamber",
        type=str,
        choices=["house", "senate"],
        default="house",
        help="Chamber to fetch votes from (currently only 'house' is supported)",
    )
    parser.add_argument(
        "--max-votes",
        type=int,
        help="Maximum number of recent roll call votes to check (default: all votes). Use lower values for faster results (e.g., 50 for recent votes only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: data/votes_<bioguide_id>.json)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.bioguide_id and not args.name:
        parser.error("Either --bioguide-id or --name is required")

    # Load environment and API key
    load_env_file(Path(".env"))
    api_key = ensure_api_key()

    # Get bioguide ID
    bioguide_id = args.bioguide_id

    if not bioguide_id and args.name:
        print(f"Searching for member: {args.name}")
        member = search_member_by_name(args.name, api_key)
        if not member:
            print(f"Error: Could not find member with name '{args.name}'")
            return
        bioguide_id = member.get("bioguideId")
        print(f"Found: {member.get('name')} (BioguideID: {bioguide_id})")

    # Fetch votes
    print(f"\nFetching voting record for {bioguide_id}...")
    if args.congress:
        print(f"Filtering by Congress: {args.congress}")

    votes, member_info = fetch_member_votes(
        bioguide_id, api_key, args.congress, args.chamber, args.max_votes
    )

    # Generate summary
    summary = format_vote_summary(votes)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"data/votes_{bioguide_id}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get chamber from terms
    chamber_value = None
    if member_info.get("terms"):
        terms_data = member_info.get("terms")
        if isinstance(terms_data, dict) and "item" in terms_data:
            terms_list = terms_data["item"]
        elif isinstance(terms_data, list):
            terms_list = terms_data
        else:
            terms_list = []

        if terms_list:
            chamber_value = terms_list[-1].get("chamber")

    # Create output payload
    output_payload = {
        "fetchedAtUtc": datetime.now(timezone.utc).isoformat(),
        "member": {
            "bioguideId": bioguide_id,
            "name": member_info.get("name"),
            "state": member_info.get("state"),
            "party": member_info.get("partyName"),
            "chamber": chamber_value,
        },
        "congress": args.congress,
        "voteSummary": summary,
        "totalVotes": len(votes),
        "votes": votes,
    }

    # Write to file
    output_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Member: {member_info.get('name')}")
    print(f"Party: {member_info.get('partyName')}")
    print(f"State: {member_info.get('state')}")
    print(f"{'='*60}")
    print(f"\nVoting Summary:")
    print(f"  Total Votes: {len(votes)}")
    for position, count in summary.items():
        if count > 0:
            percentage = (count / len(votes) * 100) if votes else 0
            print(f"  {position}: {count} ({percentage:.1f}%)")
    print(f"\nDetailed voting record saved to: {output_path}")


if __name__ == "__main__":
    main()
