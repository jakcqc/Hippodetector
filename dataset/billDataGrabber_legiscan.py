import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://api.legiscan.com/"
DEFAULT_STATE = "US"  # US Congress in LegiScan
YEARS_BACK = 1
MAX_WORKERS = 6

# Vote/member detail behavior
FETCH_MEMBER_DETAILS_FOR_VOTES = True

# Save/resume behavior
DEFAULT_OUTPUT_PATH_TEMPLATE = "data/legiscan_bill_summaries_last_{years}_years.json"
CHECKPOINT_EVERY = 25
CHECKPOINT_SECONDS = 30
PROGRESS_EVERY = 50

logger = logging.getLogger("billDataGrabberLegiScan")


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
        os.getenv("LEGISCAN_API_KEY")
        or os.getenv("LEGISCAN_API")
        or os.getenv("LEGISCAN_KEY")
    )
    if not key:
        raise RuntimeError(
            "Missing LegiScan API key. Set LEGISCAN_API_KEY (or LEGISCAN_API / LEGISCAN_KEY)."
        )
    return key


def request_json(url: str, retries: int = 5, backoff: float = 1.5) -> Dict[str, Any]:
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
            if e.code in (429, 500, 502, 503, 504):
                wait = backoff ** attempt
                logger.warning(f"HTTP {e.code} for {url}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise
        except URLError as e:
            wait = backoff ** attempt
            logger.warning(f"Network error: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
            continue

    raise RuntimeError("Max retries exceeded")


def legiscan_call(api_key: str, op: str, **params: Any) -> Dict[str, Any]:
    q: Dict[str, Any] = {"key": api_key, "op": op}
    q.update(params)
    url = f"{BASE_URL}?{urlencode(q)}"
    payload = request_json(url)

    status = str(payload.get("status", "")).upper()
    if status != "OK":
        alert = payload.get("alert")
        message = ""
        if isinstance(alert, dict):
            message = str(alert.get("message") or "")
        raise RuntimeError(f"LegiScan {op} failed: status={status} message={message}")
    return payload


def normalize_date(date_text: str) -> str:
    text = str(date_text or "").strip()
    if not text or text == "0000-00-00":
        return ""
    return text


def within_window(date_text: str, from_dt: datetime, to_dt: datetime) -> bool:
    text = normalize_date(date_text)
    if not text:
        return True
    try:
        dt = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return from_dt <= dt <= to_dt
    except ValueError:
        return True


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def bill_key(state: str, session_id: Any, bill_number: str, bill_id: Any) -> str:
    return f"{str(state).upper()}-{str(session_id)}-{str(bill_number).strip()}-{str(bill_id)}"


def bill_key_from_result(item: Dict[str, Any]) -> str:
    return bill_key(
        str(item.get("state", "")),
        item.get("sessionId", ""),
        str(item.get("billNumber", "")),
        item.get("billId", ""),
    )


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def extract_master_bills(masterlist_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    # getMasterList returns metadata fields and numeric keys for bills
    masterlist = masterlist_payload.get("masterlist")
    if not isinstance(masterlist, dict):
        return []

    out: List[Dict[str, Any]] = []
    for key, value in masterlist.items():
        if not isinstance(value, dict):
            continue
        if "bill_id" not in value:
            continue
        out.append(value)
    return out


def map_vote_counts(vote_texts: List[str]) -> Dict[str, int]:
    counts = {"yea": 0, "nay": 0, "notVoting": 0, "absent": 0, "other": 0}
    for raw in vote_texts:
        v = str(raw or "").strip().lower()
        if v in {"yea", "yes", "aye"}:
            counts["yea"] += 1
        elif v in {"nay", "no"}:
            counts["nay"] += 1
        elif v in {"nv", "not voting", "abstain", "abstention"}:
            counts["notVoting"] += 1
        elif v in {"absent", "excused"}:
            counts["absent"] += 1
        else:
            counts["other"] += 1
    return counts


def slim_person(person: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "peopleId": person.get("people_id"),
        "name": person.get("name"),
        "firstName": person.get("first_name"),
        "middleName": person.get("middle_name"),
        "lastName": person.get("last_name"),
        "suffix": person.get("suffix"),
        "party": person.get("party"),
        "role": person.get("role"),
        "district": person.get("district"),
        "state": person.get("state"),
        "stateId": person.get("state_id"),
    }


def get_person_cached(
    api_key: str,
    people_id: Any,
    people_cache: Dict[str, Dict[str, Any]],
    people_lock: Lock,
) -> Dict[str, Any] | None:
    pid = str(people_id or "").strip()
    if not pid:
        return None

    with people_lock:
        cached = people_cache.get(pid)
    if cached is not None:
        return cached

    try:
        person_payload = legiscan_call(api_key, "getPerson", id=pid)
        person = person_payload.get("person")
        if not isinstance(person, dict):
            return None
        compact = slim_person(person)
        with people_lock:
            people_cache[pid] = compact
        return compact
    except Exception as e:
        logger.warning(f"Failed getPerson for people_id={pid}: {e}")
        return None


def build_vote_event(
    api_key: str,
    vote_stub: Dict[str, Any],
    people_cache: Dict[str, Dict[str, Any]],
    people_lock: Lock,
) -> Dict[str, Any] | None:
    roll_call_id = vote_stub.get("roll_call_id")
    if roll_call_id is None:
        return None

    roll_payload = legiscan_call(api_key, "getRollCall", id=roll_call_id)
    roll = roll_payload.get("roll_call")
    if not isinstance(roll, dict):
        return None

    member_votes: List[Dict[str, Any]] = []
    vote_texts: List[str] = []

    for v in roll.get("votes", []) or []:
        if not isinstance(v, dict):
            continue

        people_id = v.get("people_id")
        vote_text = str(v.get("vote_text") or "").strip()
        vote_texts.append(vote_text)

        member: Dict[str, Any] = {
            "peopleId": people_id,
            "voteText": vote_text,
            "voteId": v.get("vote_id"),
        }

        if FETCH_MEMBER_DETAILS_FOR_VOTES and people_id is not None:
            person_info = get_person_cached(api_key, people_id, people_cache, people_lock)
            if person_info:
                member["member"] = person_info

        member_votes.append(member)

    vote_counts_from_members = map_vote_counts(vote_texts)

    return {
        "rollCallId": roll.get("roll_call_id"),
        "date": normalize_date(roll.get("date")),
        "description": roll.get("desc"),
        "chamber": roll.get("chamber"),
        "chamberId": roll.get("chamber_id"),
        "passed": bool(to_int(roll.get("passed"))),
        "totals": {
            "yea": to_int(roll.get("yea")),
            "nay": to_int(roll.get("nay")),
            "notVoting": to_int(roll.get("nv")),
            "absent": to_int(roll.get("absent")),
            "total": to_int(roll.get("total")),
        },
        "voteCountsFromMembers": vote_counts_from_members,
        "members": member_votes,
    }


def enrich_bill(
    api_key: str,
    bill_id: Any,
    from_dt: datetime,
    to_dt: datetime,
    people_cache: Dict[str, Dict[str, Any]],
    people_lock: Lock,
) -> Dict[str, Any] | None:
    try:
        payload = legiscan_call(api_key, "getBill", id=bill_id)
        bill = payload.get("bill")
        if not isinstance(bill, dict):
            return None

        status_date = normalize_date(bill.get("status_date"))
        if status_date and not within_window(status_date, from_dt, to_dt):
            return None

        vote_events: List[Dict[str, Any]] = []
        for vote_stub in bill.get("votes", []) or []:
            if not isinstance(vote_stub, dict):
                continue
            try:
                event = build_vote_event(api_key, vote_stub, people_cache, people_lock)
                if event:
                    vote_events.append(event)
            except Exception as e:
                logger.warning(
                    f"Failed roll call fetch for bill_id={bill.get('bill_id')} "
                    f"roll_call_id={vote_stub.get('roll_call_id')}: {e}"
                )

        sponsors: List[Dict[str, Any]] = []
        for s in bill.get("sponsors", []) or []:
            if not isinstance(s, dict):
                continue
            sponsors.append(
                {
                    "peopleId": s.get("people_id"),
                    "name": s.get("name"),
                    "firstName": s.get("first_name"),
                    "lastName": s.get("last_name"),
                    "party": s.get("party"),
                    "role": s.get("role"),
                    "district": s.get("district"),
                    "sponsorTypeId": s.get("sponsor_type_id"),
                    "sponsorOrder": s.get("sponsor_order"),
                    "committeeSponsor": bool(to_int(s.get("committee_sponsor"))),
                }
            )

        return {
            "source": "legiscan",
            "state": bill.get("state"),
            "sessionId": bill.get("session_id"),
            "session": bill.get("session"),
            "billId": bill.get("bill_id"),
            "billNumber": bill.get("bill_number"),
            "billType": bill.get("bill_type"),
            "title": bill.get("title"),
            # LegiScan's bill-level summary-like field is "description"
            "description": bill.get("description"),
            "status": bill.get("status"),
            "statusDate": status_date,
            "url": bill.get("url"),
            "stateLink": bill.get("state_link"),
            "sponsors": sponsors,
            "votes": vote_events,
        }
    except Exception as e:
        logger.warning(f"Failed bill enrich for bill_id={bill_id}: {e}")
        return None


def load_resume_state(
    output_path: Path,
    years_back: int,
    state: str,
    from_iso_date: str,
    to_iso_date: str,
) -> Tuple[List[Dict[str, Any]], set[str], Dict[str, Dict[str, Any]]]:
    if not output_path.exists():
        logger.info("No existing data file found. Starting fresh run.")
        return [], set(), {}

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Could not parse existing output file at {output_path}: {e}")
        return [], set(), {}

    if (
        payload.get("yearsBack") != years_back
        or str(payload.get("state", "")).upper() != state.upper()
        or payload.get("windowFromDate") != from_iso_date
        or payload.get("windowToDate") != to_iso_date
    ):
        logger.info("Existing save file scope/window does not match current run. Starting fresh.")
        return [], set(), {}

    bills = payload.get("bills")
    if not isinstance(bills, list):
        logger.warning("Existing save file has invalid bills format. Starting fresh.")
        return [], set(), {}

    clean_bills: List[Dict[str, Any]] = []
    keys: set[str] = set()
    for item in bills:
        if not isinstance(item, dict):
            continue
        key = bill_key_from_result(item)
        if key in keys:
            continue
        keys.add(key)
        clean_bills.append(item)

    people_cache_in = payload.get("peopleById")
    people_cache: Dict[str, Dict[str, Any]] = {}
    if isinstance(people_cache_in, dict):
        for k, v in people_cache_in.items():
            if isinstance(v, dict):
                people_cache[str(k)] = v

    logger.info(
        f"Resuming from existing save: {len(clean_bills)} bills, {len(people_cache)} cached members."
    )
    return clean_bills, keys, people_cache


def build_output_payload(
    run_started_utc: datetime,
    years_back: int,
    state: str,
    from_iso_date: str,
    to_iso_date: str,
    fetched_master_count: int,
    resumed_count: int,
    bills: List[Dict[str, Any]],
    people_cache: Dict[str, Dict[str, Any]],
    is_partial: bool,
) -> Dict[str, Any]:
    return {
        "source": "legiscan",
        "runStartedAtUtc": run_started_utc.isoformat(),
        "savedAtUtc": datetime.now(timezone.utc).isoformat(),
        "isPartial": is_partial,
        "state": state,
        "yearsBack": years_back,
        "windowFromDate": from_iso_date,
        "windowToDate": to_iso_date,
        "fetchedMasterListCount": fetched_master_count,
        "resumedBillsCount": resumed_count,
        "enrichedBillsCount": len(bills),
        "remainingBillsCount": max(0, fetched_master_count - len(bills)),
        "cachedPeopleCount": len(people_cache),
        "peopleById": people_cache,
        "bills": bills,
    }


def main() -> None:
    configure_logging()
    load_env_file(Path(".env"))
    api_key = ensure_api_key()

    run_started_utc = datetime.now(timezone.utc)
    from_dt = run_started_utc - timedelta(days=365 * YEARS_BACK)
    to_dt = run_started_utc
    from_iso_date = from_dt.strftime("%Y-%m-%d")
    to_iso_date = to_dt.strftime("%Y-%m-%d")

    output_path = Path(DEFAULT_OUTPUT_PATH_TEMPLATE.format(years=YEARS_BACK))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bills, processed_keys, people_cache = load_resume_state(
        output_path=output_path,
        years_back=YEARS_BACK,
        state=DEFAULT_STATE,
        from_iso_date=from_iso_date,
        to_iso_date=to_iso_date,
    )
    resumed_count = len(bills)
    people_lock = Lock()

    logger.info(f"Fetching LegiScan master list for state={DEFAULT_STATE}...")
    master_payload = legiscan_call(api_key, "getMasterList", state=DEFAULT_STATE)
    master_bills = extract_master_bills(master_payload)
    logger.info(f"Master list bill count: {len(master_bills)}")

    work_items: List[Dict[str, Any]] = []
    for item in master_bills:
        key = bill_key(
            DEFAULT_STATE,
            item.get("session_id", ""),
            str(item.get("number", "")),
            item.get("bill_id", ""),
        )
        if key in processed_keys:
            continue
        work_items.append(item)

    logger.info(f"Bills remaining after resume check: {len(work_items)}")

    def persist(is_partial: bool) -> None:
        with people_lock:
            payload = build_output_payload(
                run_started_utc=run_started_utc,
                years_back=YEARS_BACK,
                state=DEFAULT_STATE,
                from_iso_date=from_iso_date,
                to_iso_date=to_iso_date,
                fetched_master_count=len(master_bills),
                resumed_count=resumed_count,
                bills=bills,
                people_cache=people_cache,
                is_partial=is_partial,
            )
        atomic_write_json(output_path, payload)

    if not work_items:
        persist(is_partial=False)
        logger.info(f"No remaining bills to process. Output is up to date at {output_path}.")
        return

    completed = 0
    failed = 0
    added_this_run = 0
    last_checkpoint_at = time.monotonic()

    logger.info("Fetching bill details, summaries, and roll call member votes...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                enrich_bill,
                api_key,
                item.get("bill_id"),
                from_dt,
                to_dt,
                people_cache,
                people_lock,
            ): item
            for item in work_items
        }

        for future in as_completed(futures):
            completed += 1
            item = futures[future]
            try:
                result = future.result()
            except Exception as e:
                failed += 1
                logger.exception(
                    f"Unhandled worker failure for bill_id={item.get('bill_id')}: {e}"
                )
                result = None

            if result:
                key = bill_key_from_result(result)
                if key and key not in processed_keys:
                    processed_keys.add(key)
                    bills.append(result)
                    added_this_run += 1
            else:
                failed += 1

            if completed % PROGRESS_EVERY == 0:
                with people_lock:
                    people_count = len(people_cache)
                logger.info(
                    f"Progress: {completed}/{len(work_items)} processed, "
                    f"{added_this_run} added, {failed} failed, {people_count} members cached."
                )

            elapsed = time.monotonic() - last_checkpoint_at
            should_checkpoint = added_this_run > 0 and (
                added_this_run % CHECKPOINT_EVERY == 0 or elapsed >= CHECKPOINT_SECONDS
            )
            if should_checkpoint:
                persist(is_partial=True)
                last_checkpoint_at = time.monotonic()
                logger.info(
                    f"Checkpoint saved: {len(bills)} bills, {len(people_cache)} people at {output_path}."
                )

    persist(is_partial=False)
    logger.info(
        f"Completed run. Saved {len(bills)} total bills ({added_this_run} newly added), "
        f"{len(people_cache)} cached members. Failures this run: {failed}."
    )


if __name__ == "__main__":
    main()
