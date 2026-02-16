"""
Fetch full bill details from Congress.gov API.

Usage:
    python fetch_bill_details.py --from-votes data/votes_B001316.json
    python fetch_bill_details.py --bill-ids 119-hr-3424 119-s-1234
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://api.congress.gov/v3"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BILLS_CACHE_DIR = DATA_DIR / "bills_cache"


def print_progress_bar(current: int, total: int, prefix: str = "", suffix: str = "", bar_length: int = 40) -> None:
    """Print a simple text-based progress bar."""
    if total == 0:
        return

    percent = current / total
    filled_length = int(bar_length * percent)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)

    # Truncate suffix if too long
    max_suffix_len = 30
    if len(suffix) > max_suffix_len:
        suffix = suffix[:max_suffix_len-3] + "..."

    # Print with carriage return to overwrite the same line
    sys.stdout.write(f"\r{prefix} [{bar}] {percent*100:.1f}% ({current}/{total}) {suffix}".ljust(120))
    sys.stdout.flush()

    # Print newline when complete
    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def load_env_file(env_path: Path) -> None:
    """Load environment variables from a .env file."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            import os
            os.environ.setdefault(key, value)


def get_api_key() -> str:
    """Get Congress API key from environment."""
    import os
    load_env_file(PROJECT_ROOT / ".env")
    api_key = os.getenv("CONGRESS_API")
    if not api_key:
        raise ValueError(
            "CONGRESS_API key not found. "
            "Please set it in your .env file or as an environment variable."
        )
    return api_key


def parse_bill_id(bill_id: str) -> tuple[int, str, str]:
    """
    Parse bill ID in format: congress-type-number (e.g., '119-hr-3424').

    Returns:
        (congress, bill_type, bill_number)
    """
    parts = bill_id.lower().split("-")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid bill ID format: {bill_id}. "
            "Expected format: congress-type-number (e.g., 119-hr-3424)"
        )

    congress = int(parts[0])
    bill_type = parts[1].upper()
    bill_number = parts[2]

    return congress, bill_type, bill_number


def request_api(url: str, api_key: str) -> Dict[str, Any]:
    """Make an API request with proper headers."""
    params = {"api_key": api_key, "format": "json"}
    full_url = f"{url}?{urlencode(params)}"
    req = Request(
        full_url,
        headers={
            "User-Agent": "Hippodetector/1.0",
            "Accept": "application/json",
        }
    )
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bill_summaries(congress: int, bill_type: str, bill_number: str, api_key: str) -> List[Dict[str, Any]]:
    """Fetch bill summaries from Congress.gov API."""
    url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}/summaries"
    try:
        data = request_api(url, api_key)
        return data.get("summaries", [])
    except Exception:
        return []


def fetch_bill_subjects(congress: int, bill_type: str, bill_number: str, api_key: str) -> List[Dict[str, Any]]:
    """Fetch bill subjects from Congress.gov API."""
    url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}/subjects"
    try:
        data = request_api(url, api_key)
        subjects_data = data.get("subjects", {})
        legislative_subjects = subjects_data.get("legislativeSubjects", [])
        return legislative_subjects if isinstance(legislative_subjects, list) else []
    except Exception:
        return []


def fetch_bill_details(congress: int, bill_type: str, bill_number: str, api_key: str) -> Dict[str, Any]:
    """Fetch full bill details from Congress.gov API (including summaries and subjects)."""
    url = f"{BASE_URL}/bill/{congress}/{bill_type.lower()}/{bill_number}"

    try:
        # Fetch basic bill data
        bill_data = request_api(url, api_key)
        bill = bill_data.get("bill", {})

        # Fetch summaries and subjects (additional API calls)
        time.sleep(0.1)  # Small delay between requests
        summaries = fetch_bill_summaries(congress, bill_type, bill_number, api_key)

        time.sleep(0.1)
        subjects = fetch_bill_subjects(congress, bill_type, bill_number, api_key)

        # Add to bill data
        bill["_summaries"] = summaries
        bill["_subjects"] = subjects

        return bill
    except Exception as e:
        print(f"Error fetching bill {congress}-{bill_type}-{bill_number}: {e}")
        return {}


def extract_bill_summary(bill_data: Dict[str, Any]) -> Optional[str]:
    """Extract bill summary from bill data."""
    # Use the summaries we fetched separately
    summaries = bill_data.get("_summaries", [])
    if summaries and isinstance(summaries, list):
        # Get the most recent summary
        for summary in summaries:
            if isinstance(summary, dict) and summary.get("text"):
                return summary.get("text", "")
    return None


def build_bill_record(bill_data: Dict[str, Any], bill_id: str) -> Dict[str, Any]:
    """Build bill record in our schema format."""
    # Extract subjects from the separately fetched data
    subjects = bill_data.get("_subjects", [])
    subject_names = [
        s.get("name") for s in subjects
        if isinstance(s, dict) and s.get("name")
    ]

    return {
        "billId": bill_id,
        "congress": bill_data.get("congress"),
        "type": bill_data.get("type"),
        "number": bill_data.get("number"),
        "title": bill_data.get("title", ""),
        "summary": extract_bill_summary(bill_data),
        "latestAction": bill_data.get("latestAction"),
        "subjects": subject_names,
        "cosponsors": bill_data.get("cosponsors", {}).get("count", 0),
        "url": bill_data.get("legislationUrl", "")
    }


def extract_bill_ids_from_votes(votes_file: Path) -> List[str]:
    """Extract unique bill IDs from a votes JSON file.

    Note: Excludes amendments (HAMDT, SAMDT) as they use a different API endpoint.
    """
    if not votes_file.exists():
        raise FileNotFoundError(f"Votes file not found: {votes_file}")

    votes_data = json.loads(votes_file.read_text(encoding="utf-8"))
    votes = votes_data.get("votes", [])

    # Amendment types use /amendment/ endpoint, not /bill/
    AMENDMENT_TYPES = {"HAMDT", "SAMDT"}

    bill_ids = set()
    skipped_amendments = 0

    for vote in votes:
        if not isinstance(vote, dict):
            continue

        legislation = vote.get("legislation")
        if not legislation or not isinstance(legislation, dict):
            continue

        bill_type = legislation.get("type") or ""
        bill_number = legislation.get("number") or ""
        congress = vote.get("congress")

        # Skip amendments for now (they need different endpoint)
        if bill_type.upper() in AMENDMENT_TYPES:
            skipped_amendments += 1
            continue

        if bill_type and bill_number and congress:
            bill_id = f"{congress}-{bill_type.lower()}-{bill_number}"
            bill_ids.add(bill_id)

    if skipped_amendments > 0:
        print(f"  Note: Skipped {skipped_amendments} amendments (use different API endpoint)")

    return sorted(bill_ids)


def load_cached_bill(bill_id: str) -> Optional[Dict[str, Any]]:
    """Load bill from cache if it exists."""
    cache_file = BILLS_CACHE_DIR / f"{bill_id}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    return None


def save_bill_to_cache(bill_id: str, bill_record: Dict[str, Any]) -> None:
    """Save bill record to cache."""
    BILLS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = BILLS_CACHE_DIR / f"{bill_id}.json"
    cache_file.write_text(json.dumps(bill_record, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch full bill details from Congress.gov API")
    parser.add_argument(
        "--from-votes",
        type=Path,
        help="Extract bill IDs from a votes JSON file"
    )
    parser.add_argument(
        "--bill-ids",
        nargs="+",
        help="Bill IDs in format: congress-type-number (e.g., 119-hr-3424)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )

    args = parser.parse_args()

    if not args.from_votes and not args.bill_ids:
        parser.error("Either --from-votes or --bill-ids must be specified")

    # Get bill IDs
    if args.from_votes:
        print(f"Extracting bill IDs from {args.from_votes}...")
        bill_ids = extract_bill_ids_from_votes(args.from_votes)
        print(f"Found {len(bill_ids)} unique bills")
    else:
        bill_ids = args.bill_ids

    if not bill_ids:
        print("No bills to fetch")
        return

    # Get API key
    api_key = get_api_key()

    # Fetch bills
    fetched = 0
    cached = 0
    errors = 0
    total_bills = len(bill_ids)

    print(f"\nProcessing {total_bills} bills...")

    for idx, bill_id in enumerate(bill_ids, 1):
        # Check cache first
        cached_bill = load_cached_bill(bill_id)
        if cached_bill:
            cached += 1
            print_progress_bar(idx, total_bills, prefix="  Progress", suffix=f"[CACHED] {bill_id}")
            continue

        # Fetch from API
        try:
            congress, bill_type, bill_number = parse_bill_id(bill_id)
            print_progress_bar(idx, total_bills, prefix="  Progress", suffix=f"[FETCH] {bill_id}")

            bill_data = fetch_bill_details(congress, bill_type, bill_number, api_key)

            if bill_data:
                bill_record = build_bill_record(bill_data, bill_id)
                save_bill_to_cache(bill_id, bill_record)
                fetched += 1
            else:
                errors += 1

            # Rate limiting
            if idx < total_bills:
                time.sleep(args.delay)

        except Exception:
            errors += 1
            print_progress_bar(idx, total_bills, prefix="  Progress", suffix=f"[ERROR] {bill_id}")

    print(f"\nSummary:")
    print(f"  Fetched: {fetched}")
    print(f"  Cached: {cached}")
    print(f"  Errors: {errors}")
    print(f"\nBills cached at: {BILLS_CACHE_DIR}")


if __name__ == "__main__":
    main()