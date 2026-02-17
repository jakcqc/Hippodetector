"""
Build a compact bills dataset containing only bills that have votes.

If the same bill appears more than once in the source input, it is merged into a
single bill object with a `versions` array. Each version is marked with
`versionNumber`.

Usage:
    python dataset/build_voted_bills_compact.py
    python dataset/build_voted_bills_compact.py --input data/congress_bill_summaries_last_1_years.json --output data/congress_bills_voted_compact_last_1_year.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "congress_bill_summaries_last_1_years.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "congress_bills_voted_compact_last_1_year.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bill_key(bill: dict[str, Any]) -> str:
    congress = str(bill.get("congress") or "").strip()
    bill_type = str(bill.get("billType") or "").strip().upper()
    bill_number = str(bill.get("billNumber") or "").strip()
    return f"{congress}-{bill_type}-{bill_number}"


def vote_key(vote: dict[str, Any]) -> str:
    chamber = str(vote.get("chamber") or "")
    congress = str(vote.get("congress") or "")
    session = str(vote.get("sessionNumber") or "")
    roll = str(vote.get("rollNumber") or "")
    recorded_at = str(vote.get("recordedAt") or "")
    return f"{chamber}|{congress}|{session}|{roll}|{recorded_at}"


def _append_clean(values: set[str], raw: Any) -> None:
    text = str(raw or "").strip()
    if text:
        values.add(text)


def extract_proposer_metadata(bill: dict[str, Any]) -> tuple[list[str], list[str]]:
    parties: set[str] = set()
    states: set[str] = set()

    sponsor_like = []
    if isinstance(bill.get("sponsors"), list):
        sponsor_like.extend(bill.get("sponsors", []))
    if isinstance(bill.get("sponsor"), dict):
        sponsor_like.append(bill.get("sponsor"))
    if isinstance(bill.get("introducedBy"), dict):
        sponsor_like.append(bill.get("introducedBy"))

    for sponsor in sponsor_like:
        if not isinstance(sponsor, dict):
            continue
        _append_clean(parties, sponsor.get("party"))
        _append_clean(parties, sponsor.get("partyName"))
        _append_clean(states, sponsor.get("state"))
        _append_clean(states, sponsor.get("stateCode"))

    if not parties:
        parties.add("Unknown")
    if not states:
        states.add("Unknown")
    return sorted(parties), sorted(states)


def build_compact(source_payload: dict[str, Any], source_file: Path) -> dict[str, Any]:
    source_bills = source_payload.get("bills", [])
    if not isinstance(source_bills, list):
        raise ValueError("Source payload does not contain a valid 'bills' list.")

    merged_by_key: dict[str, dict[str, Any]] = {}
    voted_input_rows = 0
    total_vote_events = 0

    for idx, bill in enumerate(source_bills):
        if not isinstance(bill, dict):
            continue

        votes = bill.get("votes", [])
        if not isinstance(votes, list) or not votes:
            continue

        voted_input_rows += 1
        total_vote_events += len(votes)
        key = bill_key(bill)

        if key not in merged_by_key:
            proposer_parties, proposer_states = extract_proposer_metadata(bill)
            merged_by_key[key] = {
                "billId": key,
                "congress": bill.get("congress"),
                "billType": bill.get("billType"),
                "billNumber": bill.get("billNumber"),
                "title": bill.get("title"),
                "latestUpdateDate": bill.get("updateDate"),
                "summaries": bill.get("summaries", []),
                "proposerParties": proposer_parties,
                "proposerStates": proposer_states,
                "votes": [],
                "versions": [],
            }

        merged_bill = merged_by_key[key]
        merged_bill["title"] = bill.get("title") or merged_bill.get("title")
        current_parties = set(str(v) for v in (merged_bill.get("proposerParties") or []))
        current_states = set(str(v) for v in (merged_bill.get("proposerStates") or []))
        more_parties, more_states = extract_proposer_metadata(bill)
        current_parties.update(more_parties)
        current_states.update(more_states)
        if len(current_parties) > 1 and "Unknown" in current_parties:
            current_parties.discard("Unknown")
        if len(current_states) > 1 and "Unknown" in current_states:
            current_states.discard("Unknown")
        merged_bill["proposerParties"] = sorted(current_parties) if current_parties else ["Unknown"]
        merged_bill["proposerStates"] = sorted(current_states) if current_states else ["Unknown"]

        update_date = bill.get("updateDate")
        latest_update = merged_bill.get("latestUpdateDate")
        if isinstance(update_date, str) and (
            not isinstance(latest_update, str) or update_date > latest_update
        ):
            merged_bill["latestUpdateDate"] = update_date
            merged_bill["summaries"] = bill.get("summaries", [])

        existing_vote_keys = {vote_key(v) for v in merged_bill["votes"] if isinstance(v, dict)}
        for vote in votes:
            if not isinstance(vote, dict):
                continue
            if vote_key(vote) not in existing_vote_keys:
                merged_bill["votes"].append(vote)
                existing_vote_keys.add(vote_key(vote))

        version_number = len(merged_bill["versions"]) + 1
        merged_bill["versions"].append(
            {
                "versionNumber": version_number,
                "sourceBillIndex": idx,
                "updateDate": bill.get("updateDate"),
                "title": bill.get("title"),
                "summaries": bill.get("summaries", []),
                "votes": votes,
            }
        )

    compact_bills = sorted(
        merged_by_key.values(),
        key=lambda b: (
            str(b.get("congress") or ""),
            str(b.get("billType") or ""),
            int(str(b.get("billNumber") or "0")) if str(b.get("billNumber") or "").isdigit() else 0,
        ),
    )
    for bill in compact_bills:
        bill["versionCount"] = len(bill.get("versions", []))
        bill["voteCount"] = len(bill.get("votes", []))

    return {
        "generatedAtUtc": utc_now_iso(),
        "sourceFile": str(source_file),
        "sourceSavedAtUtc": source_payload.get("savedAtUtc"),
        "sourceWindowFromUtc": source_payload.get("windowFromUtc"),
        "sourceWindowToUtc": source_payload.get("windowToUtc"),
        "totalBillsInSource": len(source_bills),
        "votedBillRowsInSource": voted_input_rows,
        "votedBillsCount": len(compact_bills),
        "voteEventsCount": sum(len(b.get("votes", [])) for b in compact_bills),
        "inputVoteEventsCountRaw": total_vote_events,
        "bills": compact_bills,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact voted-bills dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Source JSON file (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output JSON file (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    source_payload = load_json(args.input)
    if not isinstance(source_payload, dict):
        raise ValueError("Input JSON must be an object with a 'bills' array.")

    compact_payload = build_compact(source_payload, args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact_payload, ensure_ascii=False), encoding="utf-8")

    print(f"Saved compact voted bills JSON to: {args.output}")
    print(f"Voted bills: {compact_payload['votedBillsCount']}")
    print(f"Vote events: {compact_payload['voteEventsCount']}")


if __name__ == "__main__":
    main()
