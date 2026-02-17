import json
from pathlib import Path
from typing import Any, Dict, List

MEMBERS_DIR = Path("data/members")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def merge_votes_and_bills(votes: List[Dict[str, Any]], bills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bills_by_id: Dict[str, Dict[str, Any]] = {
        bill.get("billId"): bill for bill in bills if bill.get("billId")
    }

    merged: List[Dict[str, Any]] = []
    for vote in votes:
        bill_id = vote.get("billId")
        merged.append({
            **vote,
            "bill": bills_by_id.get(bill_id)
        })

    return merged


def process_member_file(path: Path) -> None:
    profile = load_json(path)

    votes = profile.get("votes", [])
    bills = profile.get("bills", [])

    merged_votes = merge_votes_and_bills(votes, bills)

    profile["votes"] = merged_votes
    profile.pop("bills", None)

    save_json(profile, path)

    print(f"{path.name}: merged {len(merged_votes)} votes and overwrote file")


def main() -> None:
    if not MEMBERS_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {MEMBERS_DIR}")

    member_files = sorted(MEMBERS_DIR.glob("*.json"))
    if not member_files:
        print(f"No member files found in {MEMBERS_DIR}")
        return

    for member_file in member_files:
        process_member_file(member_file)


if __name__ == "__main__":
    main()
