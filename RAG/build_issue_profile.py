"""
Build aggregated issue profiles by analyzing member press releases.

This script processes press releases sequentially, using an LLM to iteratively
update stance information across all issues defined in CandidateIssueProfile.
It is resumable across runs via a progress JSON file.

Usage:
    uv run python RAG/build_issue_profile.py
    uv run python RAG/build_issue_profile.py --max-prs 40
    uv run python RAG/build_issue_profile.py --bioguide-id B001316

Output:
    data/stances/{bioguide_id}_issue_profile.json
    data/stances/issue_profile_progress.json
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

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
ARCHIA_MODEL = "gpt-5-mini"
ARCHIA_READ_TIMEOUT_SECONDS = float(os.getenv("ARCHIA_READ_TIMEOUT_SECONDS", "330"))
ARCHIA_CONNECT_TIMEOUT_SECONDS = float(os.getenv("ARCHIA_CONNECT_TIMEOUT_SECONDS", "30"))
ARCHIA_WRITE_TIMEOUT_SECONDS = float(os.getenv("ARCHIA_WRITE_TIMEOUT_SECONDS", "30"))
ARCHIA_POOL_TIMEOUT_SECONDS = float(os.getenv("ARCHIA_POOL_TIMEOUT_SECONDS", "30"))

# Processing limits
DEFAULT_MAX_PRS = 40
DEFAULT_PROGRESS_FILENAME = "issue_profile_progress.json"
DEFAULT_CHECKPOINT_EVERY = 5
DEFAULT_PR_RETRIES = 3


def load_press_releases_data() -> Dict[str, Any]:
    """Load all press releases from consolidated JSON file."""
    if not PRESS_RELEASES_FILE.exists():
        raise FileNotFoundError(f"Press releases file not found: {PRESS_RELEASES_FILE}")

    with open(PRESS_RELEASES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("membersByBioguideId", {})


@lru_cache(maxsize=None)
def load_member_name(bioguide_id: str) -> str:
    """Load member name from congress_members.json."""
    if not CONGRESS_MEMBERS_FILE.exists():
        return f"Member {bioguide_id}"

    with open(CONGRESS_MEMBERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    members = data.get("members", [])
    for member in members:
        if member.get("bioguideId") == bioguide_id:
            return member.get("name", f"Member {bioguide_id}")

    return f"Member {bioguide_id}"


def utc_now_iso() -> str:
    """Return UTC timestamp in ISO 8601 with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def default_progress_state(max_prs: int) -> Dict[str, Any]:
    """Create default progress state."""
    return {
        "schemaVersion": 1,
        "maxPressReleasesThreshold": max_prs,
        "updatedAt": utc_now_iso(),
        "completedBioguideIds": [],
        "skippedMembers": {},
        "failedMembers": {},
        "memberCheckpoints": {},
    }


def load_progress_state(progress_file: Path, max_prs: int) -> Dict[str, Any]:
    """Load progress state from disk (or initialize default)."""
    progress = default_progress_state(max_prs=max_prs)
    if not progress_file.exists():
        return progress

    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Progress file is invalid JSON: {progress_file}") from e

    if not isinstance(raw, dict):
        return progress

    completed = raw.get("completedBioguideIds", [])
    skipped = raw.get("skippedMembers", {})
    failed = raw.get("failedMembers", {})
    checkpoints = raw.get("memberCheckpoints", {})

    if isinstance(completed, list):
        progress["completedBioguideIds"] = sorted({str(x) for x in completed if x})
    if isinstance(skipped, dict):
        progress["skippedMembers"] = skipped
    if isinstance(failed, dict):
        progress["failedMembers"] = failed
    if isinstance(checkpoints, dict):
        progress["memberCheckpoints"] = checkpoints

    return progress


def save_progress_state(
    progress_file: Path,
    progress: Dict[str, Any],
    completed_members: Set[str],
    skipped_members: Dict[str, Any],
    failed_members: Dict[str, Any],
    member_checkpoints: Dict[str, Any],
    max_prs: int,
) -> None:
    """Persist progress state to disk atomically when possible.

    On Windows, antivirus/indexers/editors can briefly lock the destination file.
    In that case, retry replace and fall back to direct write.
    """
    progress["maxPressReleasesThreshold"] = max_prs
    progress["completedBioguideIds"] = sorted(completed_members)
    progress["skippedMembers"] = dict(sorted(skipped_members.items()))
    progress["failedMembers"] = dict(sorted(failed_members.items()))
    progress["memberCheckpoints"] = dict(sorted(member_checkpoints.items()))
    progress["updatedAt"] = utc_now_iso()

    progress_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = progress_file.with_suffix(progress_file.suffix + ".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

    last_error: PermissionError | None = None
    for attempt in range(5):
        try:
            temp_file.replace(progress_file)
            return
        except PermissionError as e:
            last_error = e
            time.sleep(0.2 * (attempt + 1))

    # Fallback: direct write if atomic replace keeps failing.
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        return
    except PermissionError as e:
        raise PermissionError(
            f"Unable to write progress file due to file lock: {progress_file}"
        ) from (last_error or e)


def clean_optional_text(value: Any, default: str = "") -> str:
    """Normalize optional values into safe strings."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def member_has_usable_press_release_text(press_releases: List[Dict[str, Any]]) -> bool:
    """Return True if at least one PR has non-empty body text."""
    for pr in press_releases:
        body = clean_optional_text(pr.get("bodyText"), "").strip()
        if body:
            return True
    return False


def parse_profile_payload(payload: Dict[str, Any]) -> CandidateIssueProfile:
    """Parse profile payload while tolerating legacy string-only issue values."""
    try:
        return CandidateIssueProfile(**payload)
    except Exception:
        normalized: Dict[str, Any] = {}
        for issue_name, value in payload.items():
            if isinstance(value, str):
                normalized[issue_name] = {"summary": value, "evidence": 0}
            else:
                normalized[issue_name] = value
        return CandidateIssueProfile(**normalized)


def get_evidence_count(issue_value: Any) -> int:
    """Extract evidence integer from issue payload/model."""
    if isinstance(issue_value, dict):
        raw = issue_value.get("evidence", 0)
    else:
        raw = getattr(issue_value, "evidence", 0)

    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def get_summary_text(issue_value: Any) -> str:
    """Extract summary string from issue payload/model."""
    if isinstance(issue_value, dict):
        raw = issue_value.get("summary", "")
    else:
        raw = getattr(issue_value, "summary", "")
    return clean_optional_text(raw, "")


def evidence_regressions(
    old_profile: CandidateIssueProfile,
    new_profile: CandidateIssueProfile,
) -> List[Tuple[str, int, int]]:
    """Return issue evidence regressions where new evidence < old evidence."""
    regressions: List[Tuple[str, int, int]] = []
    old_dict = old_profile.model_dump()
    new_dict = new_profile.model_dump()

    for issue_name in old_dict:
        old_evidence = get_evidence_count(old_dict.get(issue_name))
        new_evidence = get_evidence_count(new_dict.get(issue_name))
        if new_evidence < old_evidence:
            regressions.append((issue_name, old_evidence, new_evidence))

    return regressions


def find_existing_completed_members(output_dir: Path) -> Set[str]:
    """Discover members with an existing issue profile output file."""
    completed: Set[str] = set()
    suffix = "_issue_profile.json"
    if not output_dir.exists():
        return completed

    for profile_file in output_dir.glob(f"*{suffix}"):
        name = profile_file.name
        if name.endswith(suffix):
            completed.add(name[:-len(suffix)])

    return completed


def save_issue_profile(output_dir: Path, bioguide_id: str, profile: CandidateIssueProfile) -> Path:
    """Save generated issue profile to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{bioguide_id}_issue_profile.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(), f, indent=2, ensure_ascii=False)
    return output_file


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
            timeout = httpx.Timeout(
                connect=ARCHIA_CONNECT_TIMEOUT_SECONDS,
                read=ARCHIA_READ_TIMEOUT_SECONDS,
                write=ARCHIA_WRITE_TIMEOUT_SECONDS,
                pool=ARCHIA_POOL_TIMEOUT_SECONDS,
            )
            with httpx.Client(http2=True, timeout=timeout, headers=headers) as client:
                response = client.post(
                    f"{ARCHIA_BASE_URL}/responses",
                    json={
                        "model": ARCHIA_MODEL,
                        "input": prompt,
                    },
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

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code is not None and status_code >= 500 and attempt < max_retries - 1:
                print(
                    f"    Server error {status_code} on attempt {attempt + 1}/{max_retries}, retrying..."
                )
                continue
            raise
        except (httpx.TimeoutException, httpx.ReadTimeout) as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"    Timeout on attempt {attempt + 1}/{max_retries}, retrying...")
                continue
            raise RuntimeError(
                f"API timeout after {max_retries} attempts "
                f"(read timeout {ARCHIA_READ_TIMEOUT_SECONDS:.0f}s)"
            ) from e

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
- CandidateIssueProfile has 24 issue objects: {{"summary": str, "evidence": int}}
- For each issue CLEARLY addressed: update summary and evidence
- Never decrease evidence for any issue; keep or increase it
- If an issue is not addressed in this PR, keep both summary and evidence unchanged
- Make summaries cumulative across all processed PRs

Return ONLY valid JSON matching CandidateIssueProfile structure. No markdown."""

    return prompt


def process_member_prs(
    bioguide_id: str,
    press_releases: List[Dict[str, Any]],
    max_prs: int,
    start_index: int = 0,
    initial_profile: CandidateIssueProfile | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    pr_retry_attempts: int = DEFAULT_PR_RETRIES,
    checkpoint_callback: Callable[[CandidateIssueProfile, int, int], None] | None = None,
    verbose: bool = True,
) -> CandidateIssueProfile:
    """Process press releases sequentially to build issue profile.

    Args:
        start_index: zero-based index to resume from.
        checkpoint_every: save checkpoint every N sampled PRs.
        pr_retry_attempts: retries per PR (always using old stable profile as base).
        checkpoint_callback: callback(profile, next_index, sampled_count).
    """
    _ = bioguide_id

    profile = initial_profile or CandidateIssueProfile()

    # Limit number of PRs
    prs_to_process = press_releases[:max_prs]
    total_prs = len(prs_to_process)

    if start_index < 0 or start_index > total_prs:
        raise ValueError(
            f"Invalid start_index {start_index} for {bioguide_id}; expected 0..{total_prs}"
        )

    if verbose:
        print(f"Processing {total_prs} press releases (max: {max_prs})...")
        if start_index > 0:
            print(f"Resuming at index {start_index + 1}/{total_prs}")
        print()

    sampled_count = start_index
    last_checkpoint_index = -1

    for idx in range(start_index, total_prs):
        pr = prs_to_process[idx]
        display_index = idx + 1
        title = clean_optional_text(pr.get("title"), "Untitled")
        body = clean_optional_text(pr.get("bodyText"), "")
        date = clean_optional_text(pr.get("date"), "Unknown")

        if not body.strip():
            if verbose:
                print(f"[{display_index}/{total_prs}] {date} - {title[:60]}...")
                print("    Skipped: empty bodyText")
                print()
            sampled_count = display_index
            if checkpoint_callback and sampled_count % checkpoint_every == 0:
                checkpoint_callback(profile, display_index, sampled_count)
                last_checkpoint_index = display_index
            continue

        if verbose:
            print(f"[{display_index}/{total_prs}] {date} - {title[:60]}...")

        stable_profile = profile
        pr_succeeded = False

        for attempt in range(1, pr_retry_attempts + 1):
            try:
                prompt = build_update_prompt(
                    pr_title=title,
                    pr_body=body,
                    pr_date=date,
                    current_profile=stable_profile,
                )

                response = call_archia_llm(prompt)

                if verbose:
                    print("\n--- RAW LLM OUTPUT ---")
                    print(response)
                    print("--- END OUTPUT ---\n")

                json_text = response
                if "```json" in response:
                    json_text = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_text = response.split("```")[1].split("```")[0].strip()

                updated_data = json.loads(json_text)
                if not isinstance(updated_data, dict):
                    raise ValueError("LLM response JSON is not an object")

                new_profile = parse_profile_payload(updated_data)

                regressions = evidence_regressions(stable_profile, new_profile)
                if regressions:
                    preview = ", ".join(
                        f"{issue} {old}->{new}" for issue, old, new in regressions[:5]
                    )
                    raise ValueError(
                        "Evidence regression detected; rejecting update "
                        f"({len(regressions)} issues): {preview}"
                    )

                if verbose:
                    changes = []
                    old_dict = stable_profile.model_dump()
                    new_dict = new_profile.model_dump()

                    for field_name in new_dict:
                        old_value = old_dict.get(field_name, {})
                        new_value = new_dict.get(field_name, {})
                        old_summary = get_summary_text(old_value)
                        new_summary = get_summary_text(new_value)
                        old_evidence = get_evidence_count(old_value)
                        new_evidence = get_evidence_count(new_value)

                        if new_summary != old_summary or new_evidence != old_evidence:
                            changes.append(field_name)

                    if changes:
                        print(f"    Updated: {', '.join(changes)}")
                    else:
                        print("    No changes")

                profile = new_profile
                pr_succeeded = True
                break

            except KeyboardInterrupt:
                if checkpoint_callback:
                    checkpoint_callback(profile, idx, sampled_count)
                raise
            except Exception as e:
                if verbose:
                    print(f"    Attempt {attempt}/{pr_retry_attempts} failed: {e}")
                if attempt < pr_retry_attempts:
                    time.sleep(1.0 * attempt)

        if not pr_succeeded and verbose:
            print("    Keeping previous profile state for this PR.")

        sampled_count = display_index
        if checkpoint_callback and sampled_count % checkpoint_every == 0:
            checkpoint_callback(profile, display_index, sampled_count)
            last_checkpoint_index = display_index
        if verbose:
            print()

    if checkpoint_callback and last_checkpoint_index != total_prs:
        checkpoint_callback(profile, total_prs, sampled_count)

    return profile


def collect_candidates(
    press_releases_data: Dict[str, Any],
    bioguide_id: str | None,
) -> List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Return candidate members that have at least one press release."""
    if bioguide_id:
        member_data = press_releases_data.get(bioguide_id)
        if not member_data:
            return []
        candidates = [(bioguide_id, member_data)]
    else:
        candidates = sorted(press_releases_data.items(), key=lambda item: item[0])

    members_with_press_releases: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
    for candidate_id, member_data in candidates:
        raw_press_releases = member_data.get("pressReleases", [])
        if not isinstance(raw_press_releases, list) or not raw_press_releases:
            continue

        press_releases = [item for item in raw_press_releases if isinstance(item, dict)]
        if press_releases:
            members_with_press_releases.append((candidate_id, member_data, press_releases))

    return members_with_press_releases


def main():
    parser = argparse.ArgumentParser(
        description="Build aggregated issue profile from press releases",
        epilog="Uses LLM to iteratively update stance information across all issues",
    )

    parser.add_argument(
        "--bioguide-id",
        help="Optional member bioguide ID (if omitted, process all members with press releases)",
    )
    parser.add_argument(
        "--max-prs",
        type=int,
        default=DEFAULT_MAX_PRS,
        help=(
            f"Members with more than this many total press releases are skipped "
            f"(default: {DEFAULT_MAX_PRS})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STANCES_OUTPUT_DIR,
        help=f"Output directory (default: {STANCES_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help=f"Progress file path (default: <output-dir>/{DEFAULT_PROGRESS_FILENAME})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help=f"Save in-progress member checkpoint every N sampled PRs (default: {DEFAULT_CHECKPOINT_EVERY})",
    )
    parser.add_argument(
        "--pr-retries",
        type=int,
        default=DEFAULT_PR_RETRIES,
        help=f"Retry attempts per PR when API/validation fails (default: {DEFAULT_PR_RETRIES})",
    )

    args = parser.parse_args()

    if args.max_prs <= 0:
        parser.error("--max-prs must be greater than 0")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be greater than 0")
    if args.pr_retries <= 0:
        parser.error("--pr-retries must be greater than 0")

    verbose = not args.quiet

    if verbose:
        print("Loading press releases data...")
    press_releases_data = load_press_releases_data()
    candidates = collect_candidates(press_releases_data=press_releases_data, bioguide_id=args.bioguide_id)

    if not candidates:
        if args.bioguide_id:
            if args.bioguide_id in press_releases_data:
                print(f"Error: Member {args.bioguide_id} has no press releases")
            else:
                print(f"Error: Member {args.bioguide_id} not found in press release data")
        else:
            print("No members with press releases were found.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = args.progress_file or (args.output_dir / DEFAULT_PROGRESS_FILENAME)
    progress = load_progress_state(progress_file=progress_file, max_prs=args.max_prs)

    completed_members = set(progress.get("completedBioguideIds", []))
    skipped_members = dict(progress.get("skippedMembers", {}))
    failed_members = dict(progress.get("failedMembers", {}))
    member_checkpoints = dict(progress.get("memberCheckpoints", {}))

    # If issue-profile files already exist, treat them as completed.
    existing_completed = find_existing_completed_members(args.output_dir)
    newly_discovered_completed = existing_completed - completed_members
    if newly_discovered_completed:
        completed_members.update(newly_discovered_completed)
        save_progress_state(
            progress_file=progress_file,
            progress=progress,
            completed_members=completed_members,
            skipped_members=skipped_members,
            failed_members=failed_members,
            member_checkpoints=member_checkpoints,
            max_prs=args.max_prs,
        )

    if verbose:
        print(f"Members with press releases: {len(candidates)}")
        print(f"Already completed: {len(completed_members)}")
        print(f"Max press release threshold: {args.max_prs}")
        print(f"Checkpoint interval: every {args.checkpoint_every} sampled PRs")
        print(f"PR retry attempts: {args.pr_retries}")
        print(f"Progress file: {progress_file}")
        print()
        print("=" * 80)
        print("BUILDING ISSUE PROFILES")
        print("=" * 80)
        print()

    processed_this_run = 0
    skipped_over_limit_this_run = 0
    skipped_no_text_this_run = 0
    failed_this_run = 0
    already_completed_this_run = 0
    total_candidates = len(candidates)

    try:
        for index, (bioguide_id, member_data, press_releases) in enumerate(candidates, start=1):
            member_name = member_data.get("name") or load_member_name(bioguide_id)
            total_press_releases = len(press_releases)

            if bioguide_id in completed_members:
                already_completed_this_run += 1
                if bioguide_id in member_checkpoints:
                    member_checkpoints.pop(bioguide_id, None)
                if verbose:
                    print(f"[{index}/{total_candidates}] Skipping {member_name} ({bioguide_id}) - already completed")
                continue

            if total_press_releases > args.max_prs:
                skipped_over_limit_this_run += 1
                skipped_members[bioguide_id] = {
                    "reason": "press_release_count_above_limit",
                    "memberName": member_name,
                    "totalPressReleases": total_press_releases,
                    "maxAllowedPressReleases": args.max_prs,
                    "updatedAt": utc_now_iso(),
                }
                failed_members.pop(bioguide_id, None)
                member_checkpoints.pop(bioguide_id, None)
                save_progress_state(
                    progress_file=progress_file,
                    progress=progress,
                    completed_members=completed_members,
                    skipped_members=skipped_members,
                    failed_members=failed_members,
                    member_checkpoints=member_checkpoints,
                    max_prs=args.max_prs,
                )
                if verbose:
                    print(
                        f"[{index}/{total_candidates}] Skipping {member_name} ({bioguide_id}) - "
                        f"{total_press_releases} press releases exceeds max {args.max_prs}"
                    )
                continue

            if not member_has_usable_press_release_text(press_releases):
                skipped_no_text_this_run += 1
                skipped_members[bioguide_id] = {
                    "reason": "no_usable_press_release_text",
                    "memberName": member_name,
                    "totalPressReleases": total_press_releases,
                    "updatedAt": utc_now_iso(),
                }
                failed_members.pop(bioguide_id, None)
                member_checkpoints.pop(bioguide_id, None)
                save_progress_state(
                    progress_file=progress_file,
                    progress=progress,
                    completed_members=completed_members,
                    skipped_members=skipped_members,
                    failed_members=failed_members,
                    member_checkpoints=member_checkpoints,
                    max_prs=args.max_prs,
                )
                if verbose:
                    print(
                        f"[{index}/{total_candidates}] Skipping {member_name} ({bioguide_id}) - "
                        "no usable press-release body text"
                    )
                continue

            if verbose:
                print("=" * 80)
                print(f"[{index}/{total_candidates}] Processing {member_name} ({bioguide_id})")
                print(f"Total press releases: {total_press_releases}")
                print("=" * 80)
                print()

            checkpoint_state = member_checkpoints.get(bioguide_id, {})
            start_index = 0
            initial_profile = CandidateIssueProfile()
            if isinstance(checkpoint_state, dict):
                checkpoint_total = checkpoint_state.get("totalPressReleases")
                checkpoint_next = checkpoint_state.get("nextPressReleaseIndex")
                checkpoint_profile_raw = checkpoint_state.get("profile", {})

                try:
                    checkpoint_next_int = int(checkpoint_next)
                except (TypeError, ValueError):
                    checkpoint_next_int = 0

                if (
                    isinstance(checkpoint_total, int)
                    and checkpoint_total == total_press_releases
                    and 0 <= checkpoint_next_int <= total_press_releases
                ):
                    start_index = checkpoint_next_int
                    if isinstance(checkpoint_profile_raw, dict):
                        initial_profile = parse_profile_payload(checkpoint_profile_raw)
                    if verbose and start_index > 0:
                        print(
                            f"Resuming from checkpoint at PR {start_index + 1}/{total_press_releases}"
                        )
                else:
                    # Source changed or invalid checkpoint. Restart this member cleanly.
                    member_checkpoints.pop(bioguide_id, None)
                    save_progress_state(
                        progress_file=progress_file,
                        progress=progress,
                        completed_members=completed_members,
                        skipped_members=skipped_members,
                        failed_members=failed_members,
                        member_checkpoints=member_checkpoints,
                        max_prs=args.max_prs,
                    )

            if start_index >= total_press_releases:
                output_file = save_issue_profile(
                    output_dir=args.output_dir,
                    bioguide_id=bioguide_id,
                    profile=initial_profile,
                )
                completed_members.add(bioguide_id)
                skipped_members.pop(bioguide_id, None)
                failed_members.pop(bioguide_id, None)
                member_checkpoints.pop(bioguide_id, None)
                save_progress_state(
                    progress_file=progress_file,
                    progress=progress,
                    completed_members=completed_members,
                    skipped_members=skipped_members,
                    failed_members=failed_members,
                    member_checkpoints=member_checkpoints,
                    max_prs=args.max_prs,
                )
                processed_this_run += 1
                if verbose:
                    print(f"[OK] Checkpoint already complete. Saved to: {output_file}")
                continue

            def save_member_checkpoint(
                profile_snapshot: CandidateIssueProfile,
                next_index: int,
                sampled_count: int,
            ) -> None:
                member_checkpoints[bioguide_id] = {
                    "memberName": member_name,
                    "totalPressReleases": total_press_releases,
                    "nextPressReleaseIndex": next_index,
                    "sampledCount": sampled_count,
                    "checkpointEvery": args.checkpoint_every,
                    "profile": profile_snapshot.model_dump(),
                    "updatedAt": utc_now_iso(),
                }
                save_progress_state(
                    progress_file=progress_file,
                    progress=progress,
                    completed_members=completed_members,
                    skipped_members=skipped_members,
                    failed_members=failed_members,
                    member_checkpoints=member_checkpoints,
                    max_prs=args.max_prs,
                )

            try:
                profile = process_member_prs(
                    bioguide_id=bioguide_id,
                    press_releases=press_releases,
                    max_prs=args.max_prs,
                    start_index=start_index,
                    initial_profile=initial_profile,
                    checkpoint_every=args.checkpoint_every,
                    pr_retry_attempts=args.pr_retries,
                    checkpoint_callback=save_member_checkpoint,
                    verbose=verbose,
                )
                output_file = save_issue_profile(
                    output_dir=args.output_dir,
                    bioguide_id=bioguide_id,
                    profile=profile,
                )

                completed_members.add(bioguide_id)
                skipped_members.pop(bioguide_id, None)
                failed_members.pop(bioguide_id, None)
                member_checkpoints.pop(bioguide_id, None)
                save_progress_state(
                    progress_file=progress_file,
                    progress=progress,
                    completed_members=completed_members,
                    skipped_members=skipped_members,
                    failed_members=failed_members,
                    member_checkpoints=member_checkpoints,
                    max_prs=args.max_prs,
                )

                processed_this_run += 1
                if verbose:
                    issues_with_summaries = sum(
                        1
                        for issue in profile.model_dump().values()
                        if get_summary_text(issue).strip()
                    )
                    print(f"[OK] Saved to: {output_file}")
                    print(f"Issues with stances: {issues_with_summaries}/24")
                    print()

            except KeyboardInterrupt:
                # checkpoint callback already saved recent state; force one last save.
                save_progress_state(
                    progress_file=progress_file,
                    progress=progress,
                    completed_members=completed_members,
                    skipped_members=skipped_members,
                    failed_members=failed_members,
                    member_checkpoints=member_checkpoints,
                    max_prs=args.max_prs,
                )
                raise
            except Exception as e:
                failed_this_run += 1
                failed_members[bioguide_id] = {
                    "memberName": member_name,
                    "totalPressReleases": total_press_releases,
                    "error": str(e),
                    "updatedAt": utc_now_iso(),
                }
                save_progress_state(
                    progress_file=progress_file,
                    progress=progress,
                    completed_members=completed_members,
                    skipped_members=skipped_members,
                    failed_members=failed_members,
                    member_checkpoints=member_checkpoints,
                    max_prs=args.max_prs,
                )
                print(f"[ERROR] Failed processing {member_name} ({bioguide_id}): {e}")
                if verbose:
                    traceback.print_exc()
                print()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Resume by rerunning the script.")
        print(f"Progress saved to: {progress_file}")
        return

    print("=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)
    print(f"Candidates with press releases considered: {total_candidates}")
    print(f"Already completed before/while run: {already_completed_this_run}")
    print(f"Processed this run: {processed_this_run}")
    print(f"Skipped (press releases > {args.max_prs}): {skipped_over_limit_this_run}")
    print(f"Skipped (no usable body text): {skipped_no_text_this_run}")
    print(f"Failed this run: {failed_this_run}")
    print(f"Total completed so far: {len(completed_members)}")
    print(f"Progress file: {progress_file}")


if __name__ == "__main__":
    main()
