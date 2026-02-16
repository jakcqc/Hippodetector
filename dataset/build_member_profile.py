"""
Build complete member profile by aggregating votes, bills, and press releases.

Usage:
    python build_member_profile.py --bioguide-id B001316
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BILLS_CACHE_DIR = DATA_DIR / "bills_cache"
MEMBERS_DIR = DATA_DIR / "members"


def load_json(file_path: Path) -> Any:
    """Load JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def find_member_metadata(bioguide_id: str) -> Optional[Dict[str, Any]]:
    """Find member metadata from congress_members.json."""
    congress_members_file = DATA_DIR / "congress_members.json"
    if not congress_members_file.exists():
        print(f"Warning: {congress_members_file} not found")
        return None

    data = load_json(congress_members_file)
    members = data.get("members", [])

    for member in members:
        if member.get("bioguideId") == bioguide_id:
            return member

    return None


def load_voting_record(bioguide_id: str) -> Dict[str, Any]:
    """Load voting record from data/votes_{bioguideId}.json."""
    votes_file = DATA_DIR / f"votes_{bioguide_id}.json"
    return load_json(votes_file)


def load_bill_from_cache(bill_id: str) -> Optional[Dict[str, Any]]:
    """Load bill from cache."""
    cache_file = BILLS_CACHE_DIR / f"{bill_id}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    return None


def extract_bills_from_votes(votes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract and load bill details for all votes."""
    bill_ids = set()
    bills_map = {}

    # Collect unique bill IDs
    for vote in votes:
        legislation = vote.get("legislation")
        if not legislation or not isinstance(legislation, dict):
            continue

        bill_type = (legislation.get("type") or "").upper()
        bill_number = legislation.get("number") or ""
        congress = vote.get("congress")

        if bill_type and bill_number and congress:
            bill_id = f"{congress}-{bill_type.lower()}-{bill_number}"
            bill_ids.add(bill_id)

    # Load bill details from cache
    missing_bills = []
    for bill_id in sorted(bill_ids):
        bill_data = load_bill_from_cache(bill_id)
        if bill_data:
            bills_map[bill_id] = bill_data
        else:
            missing_bills.append(bill_id)

    if missing_bills:
        print(f"Warning: {len(missing_bills)} bills not found in cache. Run fetch_bill_details.py first.")
        print(f"Missing bills: {', '.join(missing_bills[:5])}" + ("..." if len(missing_bills) > 5 else ""))

    return list(bills_map.values())


def transform_vote_record(vote: Dict[str, Any]) -> Dict[str, Any]:
    """Transform vote record to our schema format."""
    legislation = vote.get("legislation")

    # Handle None or missing legislation
    if legislation and isinstance(legislation, dict):
        bill_type = (legislation.get("type") or "").upper()
        bill_number = legislation.get("number") or ""
        bill_url = legislation.get("url", "")
    else:
        bill_type = ""
        bill_number = ""
        bill_url = ""

    congress = vote.get("congress")
    bill_id = f"{congress}-{bill_type.lower()}-{bill_number}" if bill_type and bill_number and congress else None

    return {
        "congress": congress,
        "session": vote.get("session"),
        "rollCall": vote.get("rollCall"),
        "date": vote.get("date"),
        "billType": bill_type,
        "billNumber": bill_number,
        "billUrl": bill_url,
        "question": vote.get("question", ""),
        "result": vote.get("result", ""),
        "voteType": vote.get("voteType", ""),
        "memberVote": vote.get("memberVote", ""),
        "billId": bill_id
    }


def find_press_releases(bioguide_id: str) -> List[Dict[str, Any]]:
    """Find press releases for this member."""
    # Look for press releases in multiple possible locations
    possible_files = [
        DATA_DIR / f"{bioguide_id.lower()}_press_releases.json",
        DATA_DIR / f"press_releases_{bioguide_id}.json",
        DATA_DIR / "press_releases_by_bioguide.json",  # All members combined
    ]

    for file_path in possible_files:
        if not file_path.exists():
            continue

        data = load_json(file_path)

        # Check if it's the combined format
        members_map = data.get("membersByBioguideId", {})
        if members_map and bioguide_id in members_map:
            member_data = members_map[bioguide_id]
            releases = member_data.get("pressReleases", [])
            return transform_press_releases(releases)

        # Check if it's a single-member file
        if isinstance(data, dict) and "pressReleases" in data:
            return transform_press_releases(data["pressReleases"])

        # Check if it's just an array of releases
        if isinstance(data, list):
            return transform_press_releases(data)

    print(f"Warning: No press releases found for {bioguide_id}")
    return []


def transform_press_releases(releases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform press releases to our schema format."""
    transformed = []

    for release in releases:
        if not isinstance(release, dict):
            continue

        # Generate a simple ID from date and title
        date = release.get("date", "")
        title = release.get("title", "")
        release_id = f"pr-{date}-{title[:20].lower().replace(' ', '-')}" if date and title else None

        transformed.append({
            "id": release_id,
            "title": title,
            "date": date,
            "publishedTime": release.get("publishedTime", ""),
            "url": release.get("url", ""),
            "bodyText": release.get("bodyText", ""),
            "bodyHtml": release.get("bodyHtml", ""),
            "topics": release.get("topics", []),
            "relatedBills": release.get("relatedBills", [])
        })

    return transformed


def build_metadata(bioguide_id: str, member_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build metadata section."""
    if not member_data:
        return {
            "bioguideId": bioguide_id,
            "name": "Unknown",
            "firstName": "",
            "lastName": "",
            "state": "",
            "district": "",
            "party": "",
            "partyCode": "",
            "chamber": "",
            "currentMember": False,
            "profileUrl": "",
            "imageUrl": ""
        }

    # Extract party info
    party_name = member_data.get("partyName", "")
    party_code = "R" if "Republican" in party_name else "D" if "Democratic" in party_name else "I"

    # Parse name
    full_name = member_data.get("name", "")
    name_parts = full_name.split(", ")
    last_name = name_parts[0] if name_parts else ""
    first_name = name_parts[1] if len(name_parts) > 1 else ""

    # Determine chamber
    terms = member_data.get("terms", {}).get("item", [])
    chamber = terms[0].get("chamber", "") if terms else ""

    # Build profile URL
    last_name_lower = last_name.lower()
    profile_url = f"https://{last_name_lower}.house.gov" if chamber == "House of Representatives" else ""

    # Get image URL from depiction
    depiction = member_data.get("depiction", {})
    image_url = depiction.get("imageUrl", "")

    return {
        "bioguideId": bioguide_id,
        "name": full_name,
        "firstName": first_name,
        "lastName": last_name,
        "state": member_data.get("state", ""),
        "district": str(member_data.get("district", "")),
        "party": party_name,
        "partyCode": party_code,
        "chamber": chamber,
        "currentMember": True,  # Assume true if they're in the system
        "profileUrl": profile_url,
        "imageUrl": image_url
    }


def build_member_profile(bioguide_id: str) -> Dict[str, Any]:
    """Build complete member profile."""
    print(f"Building profile for {bioguide_id}...")

    # Load member metadata
    print("  Loading member metadata...")
    member_metadata = find_member_metadata(bioguide_id)

    # Load voting record
    print("  Loading voting record...")
    voting_data = load_voting_record(bioguide_id)
    votes = voting_data.get("votes", [])

    # Transform votes
    print("  Transforming votes...")
    transformed_votes = [transform_vote_record(v) for v in votes]

    # Extract and load bills
    print("  Loading bill details...")
    bills = extract_bills_from_votes(votes)

    # Find press releases
    print("  Loading press releases...")
    press_releases = find_press_releases(bioguide_id)

    # Build profile
    profile = {
        "metadata": build_metadata(bioguide_id, member_metadata),
        "dataCollection": {
            "fetchedAt": datetime.now(timezone.utc).isoformat(),
            "votesCount": len(transformed_votes),
            "billsCount": len(bills),
            "pressReleasesCount": len(press_releases),
            "votesSource": f"https://clerk.house.gov/members/{bioguide_id}",
            "billsSource": "https://api.congress.gov/v3/bill",
            "pressReleasesSource": profile_url if (profile_url := build_metadata(bioguide_id, member_metadata)["profileUrl"]) else ""
        },
        "votes": transformed_votes,
        "bills": bills,
        "pressReleases": press_releases
    }

    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Build complete member profile")
    parser.add_argument(
        "--bioguide-id",
        required=True,
        help="Bioguide ID of the member (e.g., B001316)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MEMBERS_DIR,
        help=f"Output directory (default: {MEMBERS_DIR})"
    )

    args = parser.parse_args()

    # Build profile
    try:
        profile = build_member_profile(args.bioguide_id)

        # Save to file
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = args.output_dir / f"{args.bioguide_id}.json"

        output_file.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print(f"\n✓ Profile saved to: {output_file}")
        print(f"  Votes: {profile['dataCollection']['votesCount']}")
        print(f"  Bills: {profile['dataCollection']['billsCount']}")
        print(f"  Press Releases: {profile['dataCollection']['pressReleasesCount']}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()