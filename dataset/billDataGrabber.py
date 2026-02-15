import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


BASE_URL = "https://api.congress.gov/v3/bill"
DEFAULT_LIMIT = 250
YEARS_BACK = 1
MAX_WORKERS = 8  # Safe parallelism for Congress API
FETCH_SUBJECTS = False  # Toggle if you really need subjects

DEFAULT_OUTPUT_PATH_TEMPLATE = "data/congress_bill_summaries_last_{years}_years.json"
CHECKPOINT_EVERY = 25
CHECKPOINT_SECONDS = 30
PROGRESS_EVERY = 50


logger = logging.getLogger("billDataGrabber")


# ------------------ ENV ------------------

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_env_file(env_path: Path) -> None:
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
    key = (
        os.getenv("CONGRESS_API")
        or os.getenv("CONGRES_API")
        or os.getenv("CONGRESS_API_KEY")
    )
    if not key:
        raise RuntimeError("Missing Congress API key.")
    return key


# ------------------ HTTP ------------------

def with_api_key(url: str, api_key: str) -> str:
    parsed = urlparse(url)
    params: Dict[str, str] = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params["api_key"] = api_key
    updated_query = urlencode(params)
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path,
         parsed.params, updated_query, parsed.fragment)
    )


def request_json(url: str, retries: int = 5, backoff: float = 1.5) -> Dict:
    for attempt in range(retries):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "Hippodetector/2.0",
                    "Accept": "application/json",
                },
            )
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as e:
            if e.code in (502, 503, 504):
                wait = backoff ** attempt
                logger.warning(f"Server error {e.code}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise

        except URLError as e:
            wait = backoff ** attempt
            logger.warning(f"Network error: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
            continue

    raise RuntimeError("Max retries exceeded")


def request_text(url: str, retries: int = 5, backoff: float = 1.5) -> str:
    for attempt in range(retries):
        try:
            request = Request(
                url,
                headers={"User-Agent": "Hippodetector/2.0"},
            )
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as e:
            if e.code in (502, 503, 504):
                wait = backoff ** attempt
                logger.warning(f"Server error {e.code}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise
        except URLError as e:
            wait = backoff ** attempt
            logger.warning(f"Network error: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
            continue
    raise RuntimeError("Max retries exceeded")


# ------------------ FETCH BILL LIST ------------------

def fetch_all_bills(api_key: str, from_datetime: str, to_datetime: str) -> List[Dict]:
    next_url = (
        f"{BASE_URL}?format=json&limit={DEFAULT_LIMIT}&sort=updateDate+desc"
        f"&fromDateTime={from_datetime}&toDateTime={to_datetime}"
    )

    bills: List[Dict] = []

    while next_url:
        payload = request_json(with_api_key(next_url, api_key))
        bills.extend(payload.get("bills", []))
        next_url = payload.get("pagination", {}).get("next")

    return bills


# ------------------ FETCH SUBRESOURCES ------------------

def with_limit(url: str, limit: int = DEFAULT_LIMIT) -> str:
    parsed = urlparse(url)
    params: Dict[str, str] = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params["format"] = "json"
    params["limit"] = str(limit)
    updated_query = urlencode(params)
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path,
         parsed.params, updated_query, parsed.fragment)
    )


def fetch_paginated_items(api_key: str, first_url: str, key: str) -> List[Dict]:
    items: List[Dict] = []
    next_url = with_limit(first_url, DEFAULT_LIMIT)

    while next_url:
        payload = request_json(with_api_key(next_url, api_key))
        batch = payload.get(key, [])
        if isinstance(batch, list):
            items.extend(batch)
        next_url = payload.get("pagination", {}).get("next")

    return items


def parse_senate_vote_xml(xml_text: str) -> Dict:
    root = ET.fromstring(xml_text)
    members: List[Dict] = []
    for node in root.findall(".//members/member"):
        members.append(
            {
                "name": (node.findtext("member_full") or "").strip(),
                "lastName": (node.findtext("last_name") or "").strip(),
                "firstName": (node.findtext("first_name") or "").strip(),
                "party": (node.findtext("party") or "").strip(),
                "state": (node.findtext("state") or "").strip(),
                "voteCast": (node.findtext("vote_cast") or "").strip(),
                "lisMemberId": (node.findtext("lis_member_id") or "").strip(),
            }
        )
    return {
        "question": (root.findtext("vote_question_text") or "").strip(),
        "result": (root.findtext("vote_result_text") or "").strip(),
        "voteDate": (root.findtext("vote_date") or "").strip(),
        "members": members,
    }


def parse_house_vote_xml(xml_text: str) -> Dict:
    root = ET.fromstring(xml_text)
    meta = root.find("vote-metadata")
    members: List[Dict] = []
    for node in root.findall(".//vote-data/recorded-vote"):
        legislator = node.find("legislator")
        if legislator is None:
            continue
        members.append(
            {
                "name": (legislator.text or "").strip(),
                "bioguideId": (legislator.get("name-id") or "").strip(),
                "party": (legislator.get("party") or "").strip(),
                "state": (legislator.get("state") or "").strip(),
                "voteCast": (node.findtext("vote") or "").strip(),
            }
        )
    return {
        "question": (meta.findtext("vote-question") if meta is not None else "") or "",
        "result": (meta.findtext("vote-result") if meta is not None else "") or "",
        "voteDate": (meta.findtext("action-date") if meta is not None else "") or "",
        "members": members,
    }


def summarize_vote_counts(members: List[Dict]) -> Dict[str, int]:
    counts = {"yea": 0, "nay": 0, "present": 0, "notVoting": 0, "other": 0}
    for member in members:
        raw_vote = str(member.get("voteCast") or "").strip().lower()
        if raw_vote in {"yea", "aye", "yes"}:
            counts["yea"] += 1
        elif raw_vote in {"nay", "no"}:
            counts["nay"] += 1
        elif raw_vote in {"present"}:
            counts["present"] += 1
        elif raw_vote in {"not voting", "not voting."}:
            counts["notVoting"] += 1
        else:
            counts["other"] += 1
    return counts


def fetch_vote_members(recorded_vote: Dict) -> List[Dict]:
    vote_url = str(recorded_vote.get("url") or "").strip()
    if not vote_url:
        return []

    try:
        xml_text = request_text(vote_url)
        parsed = urlparse(vote_url)
        if "senate.gov" in parsed.netloc.lower():
            vote_data = parse_senate_vote_xml(xml_text)
        elif "clerk.house.gov" in parsed.netloc.lower():
            vote_data = parse_house_vote_xml(xml_text)
        else:
            return []

        members = vote_data.get("members", [])
        if not isinstance(members, list):
            members = []

        event = {
            "chamber": recorded_vote.get("chamber"),
            "congress": recorded_vote.get("congress"),
            "sessionNumber": recorded_vote.get("sessionNumber"),
            "rollNumber": recorded_vote.get("rollNumber"),
            "recordedAt": recorded_vote.get("date"),
            "sourceUrl": vote_url,
            "question": vote_data.get("question"),
            "result": vote_data.get("result"),
            "voteDate": vote_data.get("voteDate"),
            "voteCounts": summarize_vote_counts(members),
            "members": members,
        }
        return [event]
    except Exception as e:
        logger.warning(f"Failed parsing vote XML at {vote_url}: {e}")
        return []


def enrich_bill(api_key: str, bill: Dict) -> Dict | None:
    try:
        detail_url = bill.get("url")
        if not detail_url:
            return None

        detail_payload = request_json(with_api_key(str(detail_url), api_key))
        bill_detail = detail_payload.get("bill", {})

        congress = str(bill_detail.get("congress"))
        bill_type = str(bill_detail.get("type"))
        bill_number = str(bill_detail.get("number"))

        summaries_info = bill_detail.get("summaries", {})
        summaries: List[Dict] = []
        if isinstance(summaries_info, dict) and summaries_info.get("url"):
            summaries = fetch_paginated_items(
                api_key=api_key,
                first_url=str(summaries_info["url"]),
                key="summaries",
            )

        vote_events: List[Dict] = []
        seen_vote_urls: set[str] = set()
        actions_info = bill_detail.get("actions", {})
        if isinstance(actions_info, dict) and actions_info.get("url"):
            actions = fetch_paginated_items(
                api_key=api_key,
                first_url=str(actions_info["url"]),
                key="actions",
            )
            for action in actions:
                for recorded_vote in action.get("recordedVotes", []) or []:
                    vote_url = str(recorded_vote.get("url") or "").strip()
                    if not vote_url or vote_url in seen_vote_urls:
                        continue
                    seen_vote_urls.add(vote_url)
                    vote_events.extend(fetch_vote_members(recorded_vote))

        result = {
            "congress": congress,
            "billType": bill_type,
            "billNumber": bill_number,
            "title": bill_detail.get("title") or bill.get("title"),
            "updateDate": bill_detail.get("updateDate") or bill.get("updateDate"),
            "summaries": summaries,
            "votes": vote_events,
        }

        if FETCH_SUBJECTS:
            subjects_info = bill_detail.get("subjects", {})
            if isinstance(subjects_info, dict) and subjects_info.get("url"):
                subjects_payload = request_json(
                    with_api_key(with_limit(str(subjects_info["url"]), DEFAULT_LIMIT), api_key)
                )
                result["subjects"] = subjects_payload.get("subjects", {})

        return result

    except Exception as e:
        congress = str(bill.get("congress"))
        bill_type = str(bill.get("type"))
        bill_number = str(bill.get("number"))
        logger.warning(f"Failed on {congress}-{bill_type}-{bill_number}: {e}")
        return None


# ------------------ CHECKPOINT / RESUME ------------------

def bill_key(congress: str, bill_type: str, bill_number: str) -> str:
    return f"{str(congress).strip()}-{str(bill_type).strip().lower()}-{str(bill_number).strip()}"


def bill_key_from_bill(bill: Dict) -> str:
    return bill_key(
        str(bill.get("congress", "")),
        str(bill.get("type", "")),
        str(bill.get("number", "")),
    )


def bill_key_from_enriched_bill(bill: Dict) -> str:
    return bill_key(
        str(bill.get("congress", "")),
        str(bill.get("billType", "")),
        str(bill.get("billNumber", "")),
    )


def atomic_write_json(path: Path, payload: Dict) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def build_output_payload(
    run_started_utc: datetime,
    years_back: int,
    from_datetime: str,
    to_datetime: str,
    fetched_bills_count: int,
    enriched_bills: List[Dict],
    is_partial: bool,
    resumed_count: int,
) -> Dict:
    processed_count = len(enriched_bills)
    return {
        "runStartedAtUtc": run_started_utc.isoformat(),
        "savedAtUtc": datetime.now(timezone.utc).isoformat(),
        "isPartial": is_partial,
        "yearsBack": years_back,
        "windowFromUtc": from_datetime,
        "windowToUtc": to_datetime,
        "fetchedBillsCount": fetched_bills_count,
        "resumedBillsCount": resumed_count,
        "enrichedBillsCount": processed_count,
        "remainingBillsCount": max(0, fetched_bills_count - processed_count),
        "bills": enriched_bills,
    }


def load_resume_state(
    output_path: Path,
    years_back: int,
    from_datetime: str,
    to_datetime: str,
) -> Tuple[List[Dict], set[str]]:
    if not output_path.exists():
        logger.info("No existing data file found. Starting fresh run.")
        return [], set()

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Could not parse existing output file at {output_path}: {e}")
        return [], set()

    existing_years_back = payload.get("yearsBack")
    existing_from = payload.get("windowFromUtc")
    existing_to = payload.get("windowToUtc")

    if existing_years_back != years_back or existing_from != from_datetime or existing_to != to_datetime:
        logger.info(
            "Existing data file window does not match this run. Starting fresh. "
            f"(existing years/window: {existing_years_back} {existing_from} -> {existing_to})"
        )
        return [], set()

    bills = payload.get("bills", [])
    if not isinstance(bills, list):
        logger.warning("Existing data file has invalid 'bills' format. Starting fresh.")
        return [], set()

    clean_bills: List[Dict] = []
    processed_keys: set[str] = set()
    for item in bills:
        if not isinstance(item, dict):
            continue
        key = bill_key_from_enriched_bill(item)
        if key in processed_keys:
            continue
        processed_keys.add(key)
        clean_bills.append(item)

    logger.info(f"Resuming from existing output file: {len(clean_bills)} bills already saved.")
    return clean_bills, processed_keys


# ------------------ MAIN ------------------

def main() -> None:
    configure_logging()
    load_env_file(Path(".env"))
    api_key = ensure_api_key()

    run_started_utc = datetime.now(timezone.utc)
    window_start_utc = run_started_utc - timedelta(days=365 * YEARS_BACK)

    from_datetime = window_start_utc.strftime("%Y-%m-%dT00:00:00Z")
    to_datetime = run_started_utc.strftime("%Y-%m-%dT00:00:00Z")

    output_path = Path(DEFAULT_OUTPUT_PATH_TEMPLATE.format(years=YEARS_BACK))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enriched_bills, processed_bill_keys = load_resume_state(
        output_path=output_path,
        years_back=YEARS_BACK,
        from_datetime=from_datetime,
        to_datetime=to_datetime,
    )
    resumed_count = len(enriched_bills)

    logger.info("Fetching bill identities...")
    bills = fetch_all_bills(api_key, from_datetime, to_datetime)
    logger.info(f"Total bills returned: {len(bills)}")

    bills_to_process = [bill for bill in bills if bill_key_from_bill(bill) not in processed_bill_keys]
    logger.info(f"Bills remaining after resume check: {len(bills_to_process)}")

    def persist(is_partial: bool) -> None:
        payload = build_output_payload(
            run_started_utc=run_started_utc,
            years_back=YEARS_BACK,
            from_datetime=from_datetime,
            to_datetime=to_datetime,
            fetched_bills_count=len(bills),
            enriched_bills=enriched_bills,
            is_partial=is_partial,
            resumed_count=resumed_count,
        )
        atomic_write_json(output_path, payload)

    if not bills_to_process:
        persist(is_partial=False)
        logger.info(f"No remaining bills to process. Output is up to date at {output_path}.")
        return

    logger.info("Fetching summaries and vote/member data in parallel...")
    completed = 0
    failed = 0
    added_this_run = 0
    last_checkpoint_monotonic = time.monotonic()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(enrich_bill, api_key, bill) for bill in bills_to_process]

        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
            except Exception as e:
                failed += 1
                logger.exception(f"Unhandled enrichment worker exception: {e}")
                result = None

            if result:
                key = bill_key_from_enriched_bill(result)
                if key and key not in processed_bill_keys:
                    processed_bill_keys.add(key)
                    enriched_bills.append(result)
                    added_this_run += 1
            else:
                failed += 1

            if completed % PROGRESS_EVERY == 0:
                logger.info(
                    f"Progress: {completed}/{len(bills_to_process)} processed, "
                    f"{added_this_run} added, {failed} failed."
                )

            elapsed_since_checkpoint = time.monotonic() - last_checkpoint_monotonic
            should_checkpoint = (
                added_this_run > 0
                and (
                    added_this_run % CHECKPOINT_EVERY == 0
                    or elapsed_since_checkpoint >= CHECKPOINT_SECONDS
                )
            )
            if should_checkpoint:
                persist(is_partial=True)
                last_checkpoint_monotonic = time.monotonic()
                logger.info(
                    f"Checkpoint saved: {len(enriched_bills)} total bills persisted to {output_path}."
                )

    persist(is_partial=False)
    logger.info(
        f"Completed run. Saved {len(enriched_bills)} total bills "
        f"({added_this_run} newly added this run) to {output_path}. "
        f"Failures this run: {failed}."
    )


if __name__ == "__main__":
    main()
