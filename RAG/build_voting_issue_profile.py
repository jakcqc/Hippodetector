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
import random
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.memberOpinions import CandidateIssueProfile, IssueSummary
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
ARCHIA_MODEL = os.getenv("ARCHIA_MODEL_OPENAI", "gpt-5-mini")

# Archia timeout configuration (following build_issue_profile.py pattern)
ARCHIA_READ_TIMEOUT_SECONDS = float(os.getenv("ARCHIA_READ_TIMEOUT_SECONDS", "330"))
ARCHIA_CONNECT_TIMEOUT_SECONDS = float(os.getenv("ARCHIA_CONNECT_TIMEOUT_SECONDS", "10"))
ARCHIA_WRITE_TIMEOUT_SECONDS = float(os.getenv("ARCHIA_WRITE_TIMEOUT_SECONDS", "30"))
ARCHIA_POOL_TIMEOUT_SECONDS = float(os.getenv("ARCHIA_POOL_TIMEOUT_SECONDS", "10"))

# Retry backoff configuration
ARCHIA_504_BACKOFF_BASE_SECONDS = float(os.getenv("ARCHIA_504_BACKOFF_BASE_SECONDS", "12"))
ARCHIA_5XX_BACKOFF_BASE_SECONDS = float(os.getenv("ARCHIA_5XX_BACKOFF_BASE_SECONDS", "4"))
ARCHIA_TIMEOUT_BACKOFF_BASE_SECONDS = float(os.getenv("ARCHIA_TIMEOUT_BACKOFF_BASE_SECONDS", "6"))
ARCHIA_RETRY_JITTER_MAX_SECONDS = float(os.getenv("ARCHIA_RETRY_JITTER_MAX_SECONDS", "2"))
ARCHIA_RETRY_BACKOFF_CAP_SECONDS = float(os.getenv("ARCHIA_RETRY_BACKOFF_CAP_SECONDS", "180"))

# Streaming support
ARCHIA_USE_STREAMING = os.getenv("ARCHIA_USE_STREAMING", "1").strip().lower() not in {"0", "false", "no"}

# Gemini API configuration (fallback)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Processing limits
DEFAULT_MAX_VOTES = 10
DEFAULT_BATCH_SIZE = 3  # Reduced from 5 to avoid API timeouts
DEFAULT_CHECKPOINT_EVERY = 5  # Save checkpoint every N vote batches
DEFAULT_BATCH_RETRIES = 3  # Retries per vote batch
SUMMARY_TRUNCATE_CHARS = 30000  # ~7,500 tokens (40% of longest bill summary)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def default_progress_state(max_votes: int) -> Dict[str, Any]:
    """Create default progress state structure."""
    return {
        "schemaVersion": 1,
        "maxVotesThreshold": max_votes,
        "updatedAt": utc_now_iso(),
        "completedBioguideIds": [],
        "skippedMembers": {},
        "failedMembers": {},
        "memberCheckpoints": {},
    }


def save_progress_state(
    progress_file: Path,
    progress: Dict[str, Any],
    completed_members: Set[str],
    skipped_members: Dict[str, Any],
    failed_members: Dict[str, Any],
    member_checkpoints: Dict[str, Any],
    max_votes: int,
) -> None:
    """Persist progress state to disk atomically when possible."""
    progress["maxVotesThreshold"] = max_votes
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

    # Fallback: direct write if atomic replace keeps failing
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


def parse_profile_payload(payload: Dict[str, Any]) -> CandidateIssueProfile:
    """Parse profile payload while tolerating legacy string-only issue values."""
    try:
        return CandidateIssueProfile(**payload)
    except Exception:
        # Convert legacy string values to IssueSummary format
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
    """Discover members with an existing voting issue profile output file."""
    completed: Set[str] = set()
    suffix = "_voting_issue_profile.json"
    if not output_dir.exists():
        return completed

    for profile_file in output_dir.glob(f"*{suffix}"):
        name = profile_file.name
        if name.endswith(suffix):
            completed.add(name[:-len(suffix)])

    return completed


def save_issue_profile(output_dir: Path, bioguide_id: str, profile: CandidateIssueProfile) -> Path:
    """Save generated voting issue profile to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{bioguide_id}_voting_issue_profile.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(), f, indent=2, ensure_ascii=False)
    return output_file


def _retry_backoff_seconds(base_seconds: float, attempt: int) -> float:
    """Compute exponential backoff with jitter for retry attempt index."""
    exp = max(0, attempt)
    raw = base_seconds * (2 ** exp)
    jitter = random.uniform(0.0, max(0.0, ARCHIA_RETRY_JITTER_MAX_SECONDS))
    return min(ARCHIA_RETRY_BACKOFF_CAP_SECONDS, raw + jitter)


def load_voting_profile(bioguide_id: str) -> VotingProfile:
    """Load voting profile from JSON file."""
    profile_file = VOTING_PROFILES_DIR / f"{bioguide_id}.json"

    if not profile_file.exists():
        raise FileNotFoundError(f"Voting profile not found: {profile_file}")

    with open(profile_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return VotingProfile(**data)


@lru_cache(maxsize=512)
def load_member_name(bioguide_id: str) -> str:
    """Load member name from congress_members.json (cached)."""
    if not CONGRESS_MEMBERS_FILE.exists():
        return f"Member {bioguide_id}"

    with open(CONGRESS_MEMBERS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    members = data.get("members", [])
    for member in members:
        if member.get("bioguideId") == bioguide_id:
            return member.get("name", f"Member {bioguide_id}")

    return f"Member {bioguide_id}"


def _extract_content_from_response(result: Dict[str, Any]) -> str:
    """Extract text from Archia/OpenAI-style responses payload."""
    collected: List[str] = []

    output_text = result.get("output_text")
    if isinstance(output_text, str):
        if output_text.strip():
            collected.append(output_text)
    elif isinstance(output_text, list):
        for chunk in output_text:
            if isinstance(chunk, str) and chunk.strip():
                collected.append(chunk)
            elif isinstance(chunk, dict):
                text_value = chunk.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    collected.append(text_value)

    output = result.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text_value = part.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    collected.append(text_value)

    text = "".join(collected).strip()
    return text


def _extract_text_from_stream_event(event: Dict[str, Any]) -> str:
    """Extract any text delta from a streaming event payload."""
    collected: List[str] = []

    direct_delta = event.get("delta")
    if isinstance(direct_delta, str):
        collected.append(direct_delta)

    direct_text = event.get("text")
    if isinstance(direct_text, str):
        collected.append(direct_text)

    output_text = event.get("output_text")
    if isinstance(output_text, str):
        collected.append(output_text)
    elif isinstance(output_text, list):
        for chunk in output_text:
            if isinstance(chunk, str):
                collected.append(chunk)
            elif isinstance(chunk, dict):
                text_value = chunk.get("text")
                if isinstance(text_value, str):
                    collected.append(text_value)

    choices = event.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            delta = first_choice.get("delta", {})
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    collected.append(content)
                elif isinstance(content, list):
                    for piece in content:
                        if isinstance(piece, str):
                            collected.append(piece)
                        elif isinstance(piece, dict):
                            piece_text = piece.get("text")
                            if isinstance(piece_text, str):
                                collected.append(piece_text)

    return "".join(collected)


def _read_streamed_response(
    client: httpx.Client,
    url: str,
    payload: Dict[str, Any],
) -> str:
    """Read SSE response stream and assemble final text."""
    stream_payload = dict(payload)
    stream_payload["stream"] = True

    text_chunks: List[str] = []
    final_result: Dict[str, Any] | None = None

    with client.stream("POST", url, json=stream_payload) as response:
        response.raise_for_status()

        for raw_line in response.iter_lines():
            if raw_line is None:
                continue

            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(":"):
                continue

            if line.startswith("data:"):
                payload_text = line[5:].strip()
            else:
                payload_text = line

            if not payload_text or payload_text == "[DONE]":
                continue

            try:
                event = json.loads(payload_text)
            except json.JSONDecodeError:
                continue

            if not isinstance(event, dict):
                continue

            event_type = event.get("type")
            if event_type in {"error", "response.error"}:
                err = event.get("error", {})
                if isinstance(err, dict):
                    msg = err.get("message", "Unknown streaming error")
                else:
                    msg = str(err)
                raise RuntimeError(f"Archia streaming error: {msg}")

            # Some providers emit full response object near the end
            maybe_response = event.get("response")
            if isinstance(maybe_response, dict):
                final_result = maybe_response

            # Or they emit final payload directly
            if "output" in event and isinstance(event.get("output"), list):
                final_result = event

            delta_text = _extract_text_from_stream_event(event)
            if delta_text:
                text_chunks.append(delta_text)

    if final_result:
        text = _extract_content_from_response(final_result)
        if text:
            return text

    joined = "".join(text_chunks).strip()
    if joined:
        return joined

    raise RuntimeError("No content in Archia streaming response")


def _read_non_stream_response(
    client: httpx.Client,
    url: str,
    payload: Dict[str, Any],
) -> str:
    """Read standard JSON response and extract text."""
    response = client.post(url, json=payload)
    response.raise_for_status()

    result = response.json()
    if result.get("status") == "failed":
        error = result.get("error", {})
        error_msg = error.get("message", "Unknown error")
        raise RuntimeError(f"Archia API error: {error_msg}")

    content = _extract_content_from_response(result)
    if content:
        return content

    raise RuntimeError("No content in Archia response")


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
    """Call Archia API with prompt and return response text."""
    if not ARCHIA_API_KEY:
        raise ValueError("ARCHIA API key not found in environment")

    headers = {
        "x-api-key": ARCHIA_API_KEY.strip(),
        "Authorization": f"Bearer {ARCHIA_API_KEY.strip()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    last_error = None
    url = f"{ARCHIA_BASE_URL}/responses"
    payload = {
        "model": ARCHIA_MODEL,
        "input": prompt,
    }

    for attempt in range(max_retries):
        try:
            timeout = httpx.Timeout(
                connect=ARCHIA_CONNECT_TIMEOUT_SECONDS,
                read=ARCHIA_READ_TIMEOUT_SECONDS,
                write=ARCHIA_WRITE_TIMEOUT_SECONDS,
                pool=ARCHIA_POOL_TIMEOUT_SECONDS,
            )
            with httpx.Client(http2=True, timeout=timeout, headers=headers) as client:
                if ARCHIA_USE_STREAMING:
                    try:
                        return _read_streamed_response(client=client, url=url, payload=payload)
                    except httpx.HTTPStatusError as e:
                        status_code = e.response.status_code if e.response is not None else None
                        # If streaming is unsupported, fall back to non-stream mode
                        if status_code in {400, 404, 405, 415, 422}:
                            return _read_non_stream_response(client=client, url=url, payload=payload)
                        raise
                return _read_non_stream_response(client=client, url=url, payload=payload)

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code is not None and status_code >= 500 and attempt < max_retries - 1:
                if status_code == 504:
                    sleep_seconds = _retry_backoff_seconds(
                        ARCHIA_504_BACKOFF_BASE_SECONDS, attempt
                    )
                else:
                    sleep_seconds = _retry_backoff_seconds(
                        ARCHIA_5XX_BACKOFF_BASE_SECONDS, attempt
                    )
                print(
                    f"    Server error {status_code} on attempt {attempt + 1}/{max_retries}, "
                    f"backing off {sleep_seconds:.1f}s before retry..."
                )
                time.sleep(sleep_seconds)
                continue
            raise
        except (httpx.TimeoutException, httpx.ReadTimeout) as e:
            last_error = e
            if attempt < max_retries - 1:
                sleep_seconds = _retry_backoff_seconds(ARCHIA_TIMEOUT_BACKOFF_BASE_SECONDS, attempt)
                print(
                    f"    Timeout on attempt {attempt + 1}/{max_retries}, "
                    f"backing off {sleep_seconds:.1f}s before retry..."
                )
                time.sleep(sleep_seconds)
                continue
            raise RuntimeError(
                f"API timeout after {max_retries} attempts "
                f"(read timeout {ARCHIA_READ_TIMEOUT_SECONDS:.0f}s)"
            ) from e

    # Should not reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Unexpected error in API call")




def truncate_summary(summary: str, max_chars: int = SUMMARY_TRUNCATE_CHARS) -> str:
    """Truncate summary to prevent token explosion."""
    if not summary or len(summary) <= max_chars:
        return summary
    return summary[:max_chars] + "..."


def build_update_prompt(
    vote_batch: List[Dict[str, Any]],
    current_profile: CandidateIssueProfile,
    batch_number: int,
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

VOTING RECORDS (Batch {batch_number}, {len(vote_batch)} votes):
{votes_section}

UPDATE RULES:
- CandidateIssueProfile has 24 issue objects: {{"summary": str, "evidence": int}}
- For each issue CLEARLY addressed: update summary and evidence
- Never decrease evidence for any issue; keep or increase it
- Evidence = total number of votes seen so far that inform this issue
- If an issue is not addressed in this batch, keep both summary and evidence unchanged
- Make summaries cumulative across all processed vote batches
- Consider the vote cast (Yea/Nay) in context of what the bill does

Return ONLY valid JSON matching CandidateIssueProfile structure. No markdown."""

    return prompt


def process_voting_records(
    bioguide_id: str,
    voting_profile: VotingProfile,
    max_votes: int,
    batch_size: int,
    start_batch: int = 0,
    initial_profile: CandidateIssueProfile | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    batch_retry_attempts: int = DEFAULT_BATCH_RETRIES,
    checkpoint_callback: Callable[[CandidateIssueProfile, int, int], None] | None = None,
    verbose: bool = True
) -> CandidateIssueProfile:
    """Process voting records in batches to build issue profile.

    Args:
        bioguide_id: Member bioguide ID
        voting_profile: VotingProfile with votes to process
        max_votes: Maximum votes to process
        batch_size: Votes per batch
        start_batch: zero-based batch index to resume from
        initial_profile: Initial profile state (for resuming)
        checkpoint_every: save checkpoint every N batches
        batch_retry_attempts: retries per batch (always using old stable profile as base)
        checkpoint_callback: callback(profile, next_batch_idx, processed_count)
        verbose: print progress output
    """
    _ = bioguide_id

    profile = initial_profile or CandidateIssueProfile()

    # Filter votes with summaries and limit
    votes_with_summaries = [
        v.model_dump() for v in voting_profile.votes
        if v.summary
    ]
    votes_to_process = votes_with_summaries[:max_votes]

    # Calculate total batches
    num_batches = (len(votes_to_process) + batch_size - 1) // batch_size

    if start_batch < 0 or start_batch > num_batches:
        raise ValueError(
            f"Invalid start_batch {start_batch} for {bioguide_id}; expected 0..{num_batches}"
        )

    if verbose:
        print(f"Processing {len(votes_to_process)} votes (max: {max_votes}, batch size: {batch_size})...")
        if start_batch > 0:
            print(f"Resuming at batch {start_batch + 1}/{num_batches}")
        print()

    processed_count = start_batch * batch_size
    last_checkpoint_batch = -1

    for batch_idx in range(start_batch, num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(votes_to_process))
        batch = votes_to_process[start_idx:end_idx]

        display_batch = batch_idx + 1

        if verbose:
            print(f"[Batch {display_batch}/{num_batches}] Processing votes {start_idx + 1}-{end_idx}...")

        stable_profile = profile
        batch_succeeded = False

        for attempt in range(1, batch_retry_attempts + 1):
            try:
                prompt = build_update_prompt(
                    vote_batch=batch,
                    current_profile=stable_profile,
                    batch_number=display_batch,
                )

                response = call_archia_llm(prompt)

                if verbose:
                    print("\n--- RAW LLM OUTPUT ---")
                    print(response[:500] + "..." if len(response) > 500 else response)
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
                batch_succeeded = True
                break

            except KeyboardInterrupt:
                if checkpoint_callback:
                    checkpoint_callback(profile, batch_idx, processed_count)
                raise
            except Exception as e:
                if verbose:
                    print(f"    Attempt {attempt}/{batch_retry_attempts} failed: {e}")
                if attempt < batch_retry_attempts:
                    time.sleep(1.0 * attempt)

        if not batch_succeeded and verbose:
            print("    Keeping previous profile state for this batch.")

        processed_count = end_idx
        if checkpoint_callback and display_batch % checkpoint_every == 0:
            checkpoint_callback(profile, display_batch, processed_count)
            last_checkpoint_batch = display_batch
        if verbose:
            print()

    if checkpoint_callback and last_checkpoint_batch != num_batches:
        checkpoint_callback(profile, num_batches, processed_count)

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
        "--progress-file",
        type=Path,
        default=None,
        help="Progress tracking file (default: {output_dir}/voting_issue_profile_progress.json)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help=f"Save checkpoint every N batches (default: {DEFAULT_CHECKPOINT_EVERY})"
    )
    parser.add_argument(
        "--batch-retries",
        type=int,
        default=DEFAULT_BATCH_RETRIES,
        help=f"Retry attempts per batch (default: {DEFAULT_BATCH_RETRIES})"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup progress file
    if args.progress_file is None:
        args.progress_file = args.output_dir / "voting_issue_profile_progress.json"

    # Load or initialize progress state
    if args.progress_file.exists():
        with open(args.progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
        if verbose:
            print(f"Loaded progress from {args.progress_file}")
    else:
        progress = default_progress_state(args.max_votes)
        if verbose:
            print("Starting fresh (no progress file found)")

    completed_members = set(progress.get("completedBioguideIds", []))
    skipped_members = progress.get("skippedMembers", {})
    failed_members = progress.get("failedMembers", {})
    member_checkpoints = progress.get("memberCheckpoints", {})

    # Determine bioguide IDs to process
    if args.from_file:
        all_bioguide_ids = parse_politician_file(args.from_file)
        if verbose:
            print(f"Loaded {len(all_bioguide_ids)} members from {args.from_file}")
    else:
        all_bioguide_ids = [args.bioguide_id]

    # Filter out already completed members
    bioguide_ids = [bid for bid in all_bioguide_ids if bid not in completed_members]

    if verbose:
        if len(bioguide_ids) < len(all_bioguide_ids):
            print(f"Skipping {len(all_bioguide_ids) - len(bioguide_ids)} already completed members")
        print(f"Processing {len(bioguide_ids)} members")
        print()

    # Process each member
    success_count = len(completed_members)
    error_count = len(failed_members)

    for idx, bioguide_id in enumerate(bioguide_ids, 1):
        if verbose:
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

            # Check for checkpoint
            checkpoint_data = member_checkpoints.get(bioguide_id, {})
            start_batch = checkpoint_data.get("nextBatchIndex", 0)
            initial_profile = None
            if start_batch > 0:
                try:
                    initial_profile = parse_profile_payload(checkpoint_data.get("profile", {}))
                    if verbose:
                        print(f"Resuming from batch {start_batch + 1}")
                        print()
                except Exception:
                    start_batch = 0
                    if verbose:
                        print("Checkpoint corrupted, starting fresh")
                        print()

            # Create checkpoint callback
            def save_checkpoint(profile: CandidateIssueProfile, next_batch: int, processed: int):
                member_checkpoints[bioguide_id] = {
                    "nextBatchIndex": next_batch,
                    "processedCount": processed,
                    "profile": profile.model_dump(),
                    "updatedAt": utc_now_iso(),
                }
                save_progress_state(
                    args.progress_file,
                    progress,
                    completed_members,
                    skipped_members,
                    failed_members,
                    member_checkpoints,
                    args.max_votes,
                )

            # Process voting records
            profile = process_voting_records(
                bioguide_id=bioguide_id,
                voting_profile=voting_profile,
                max_votes=args.max_votes,
                batch_size=args.batch_size,
                start_batch=start_batch,
                initial_profile=initial_profile,
                checkpoint_every=args.checkpoint_every,
                batch_retry_attempts=args.batch_retries,
                checkpoint_callback=save_checkpoint,
                verbose=verbose
            )

            # Save final output
            output_file = save_issue_profile(args.output_dir, bioguide_id, profile)

            # Update progress
            completed_members.add(bioguide_id)
            if bioguide_id in member_checkpoints:
                del member_checkpoints[bioguide_id]
            if bioguide_id in failed_members:
                del failed_members[bioguide_id]

            save_progress_state(
                args.progress_file,
                progress,
                completed_members,
                skipped_members,
                failed_members,
                member_checkpoints,
                args.max_votes,
            )

            if verbose:
                print("=" * 80)
                print("FINAL PROFILE SUMMARY")
                print("=" * 80)

                # Count issues with stances
                profile_dict = profile.model_dump()
                issues_with_stances = []

                for issue_name, issue_value in profile_dict.items():
                    summary = get_summary_text(issue_value)
                    evidence = get_evidence_count(issue_value)
                    if summary.strip():
                        preview = summary[:150] + "..." if len(summary) > 150 else summary
                        issues_with_stances.append({
                            "issue": issue_name,
                            "summary": preview,
                            "evidence": evidence
                        })

                print(f"\nIssues with stances: {len(issues_with_stances)}/24")
                print()

                for item in issues_with_stances:
                    print(f"{item['issue']} (evidence: {item['evidence']}):")
                    print(f"  {item['summary']}")
                    print()

            print(f"✓ Saved to: {output_file}")
            success_count += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Progress saved.")
            sys.exit(130)
        except Exception as e:
            print(f"\n✗ Error processing {bioguide_id}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

            failed_members[bioguide_id] = {
                "error": str(e),
                "failedAt": utc_now_iso(),
            }
            save_progress_state(
                args.progress_file,
                progress,
                completed_members,
                skipped_members,
                failed_members,
                member_checkpoints,
                args.max_votes,
            )
            error_count += 1
            continue

    # Final summary
    if len(all_bioguide_ids) > 1:
        print(f"\n{'=' * 80}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'=' * 80}")
        print(f"Successful: {success_count}/{len(all_bioguide_ids)}")
        print(f"Errors: {error_count}/{len(all_bioguide_ids)}")
        print(f"Progress saved to: {args.progress_file}")

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
