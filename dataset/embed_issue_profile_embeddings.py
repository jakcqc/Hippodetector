"""
Embed issue-profile summaries into per-topic JSON array files.

Each output file represents one tracked issue topic and stores an array of:
{
  "bioguideId": "...",
  "summary": "...",
  "evidence": 0,
  "sourceIssueFile": "...",
  "summaryHash": "...",
  "updatedAt": "...",
  "embedding": [...]
}

Usage examples:
  uv run dataset/embed_issue_profile_embeddings.py
  uv run dataset/embed_issue_profile_embeddings.py --bioguide-id C001136
  uv run dataset/embed_issue_profile_embeddings.py --topic health_care --topic immigration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Add project root so local modules can be imported when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from LLM.hf_embedding_gemma import EmbeddingGemmaClient, MODEL_NAME
from dataset.memberOpinions import CandidateIssueProfile


ISSUE_FILE_SUFFIXES = ("_issue_profile.json", "_issues_profile.json")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "stances"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "issue_profile_topic_embeddings"
MAX_SAFE_LENGTH = 512


@dataclass(frozen=True)
class TopicCandidate:
    topic: str
    bioguide_id: str
    source_issue_file: str
    summary: str
    evidence: int
    text_for_embedding: str
    summary_hash: str


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def configure_stdio_utf8() -> None:
    # Some Windows shells default to cp1252 and fail on Unicode status logs.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def normalize_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def to_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def stable_summary_hash(topic: str, bioguide_id: str, summary: str) -> str:
    key = "||".join([topic, bioguide_id, summary])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def get_issue_topics() -> list[str]:
    if hasattr(CandidateIssueProfile, "model_fields"):
        keys = list(getattr(CandidateIssueProfile, "model_fields").keys())
        if keys:
            return keys
    if hasattr(CandidateIssueProfile, "__fields__"):
        keys = list(getattr(CandidateIssueProfile, "__fields__").keys())
        if keys:
            return keys
    raise RuntimeError("Unable to discover issue topics from CandidateIssueProfile.")


def parse_profile_file(path: Path) -> tuple[str, Path] | None:
    name = path.name
    for suffix in ISSUE_FILE_SUFFIXES:
        if name.endswith(suffix):
            bioguide_id = name[: -len(suffix)].strip().upper()
            if bioguide_id:
                return bioguide_id, path
    return None


def discover_issue_profile_files(
    input_dir: Path,
    requested_ids: set[str],
) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    for path in sorted(input_dir.glob("*.json")):
        parsed = parse_profile_file(path)
        if parsed is None:
            continue
        bioguide_id, issue_file = parsed
        if requested_ids and bioguide_id not in requested_ids:
            continue
        discovered.append((bioguide_id, issue_file))
    return discovered


def load_existing_topic_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = load_json(path)
    except Exception:
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        return [item for item in payload["entries"] if isinstance(item, dict)]
    return []


def has_embedding(entry: dict[str, Any]) -> bool:
    embedding = entry.get("embedding")
    return isinstance(embedding, list) and len(embedding) > 0


def collect_topic_candidates(
    profile_files: list[tuple[str, Path]],
    topics: list[str],
    include_empty: bool,
) -> tuple[dict[str, dict[str, TopicCandidate]], dict[str, set[str]], int, int]:
    candidates_by_topic: dict[str, dict[str, TopicCandidate]] = {topic: {} for topic in topics}
    members_by_topic: dict[str, set[str]] = {topic: set() for topic in topics}
    source_entries = 0
    skipped_empty = 0

    for bioguide_id, issue_file in profile_files:
        payload = load_json(issue_file)
        if not isinstance(payload, dict):
            continue

        for topic in topics:
            source_entries += 1
            members_by_topic[topic].add(bioguide_id)
            topic_payload = payload.get(topic) if isinstance(payload.get(topic), dict) else {}

            summary = normalize_string(topic_payload.get("summary"))
            evidence = max(to_int(topic_payload.get("evidence"), 0), 0)

            if summary:
                text_for_embedding = summary
            elif include_empty:
                text_for_embedding = f"No recorded stance summary for topic: {topic.replace('_', ' ')}."
            else:
                skipped_empty += 1
                continue

            summary_hash = stable_summary_hash(topic=topic, bioguide_id=bioguide_id, summary=summary)
            candidates_by_topic[topic][bioguide_id] = TopicCandidate(
                topic=topic,
                bioguide_id=bioguide_id,
                source_issue_file=issue_file.name,
                summary=summary,
                evidence=evidence,
                text_for_embedding=text_for_embedding,
                summary_hash=summary_hash,
            )

    return candidates_by_topic, members_by_topic, source_entries, skipped_empty


def process_topic(
    topic: str,
    members_in_scope: set[str],
    candidates_by_bioguide: dict[str, TopicCandidate],
    client: EmbeddingGemmaClient,
    output_dir: Path,
    batch_size: int,
    max_length: int,
    flush_every: int,
    overwrite: bool,
) -> dict[str, int]:
    output_path = output_dir / f"{topic}.json"
    existing_entries = load_existing_topic_entries(output_path)

    existing_by_bioguide: dict[str, dict[str, Any]] = {}
    for entry in existing_entries:
        bioguide_id = normalize_string(entry.get("bioguideId")).upper()
        if not bioguide_id or bioguide_id in existing_by_bioguide:
            continue
        existing_by_bioguide[bioguide_id] = dict(entry)

    removed_due_to_empty = 0
    reused_embedding = 0
    pending: list[TopicCandidate] = []

    for bioguide_id in sorted(members_in_scope):
        candidate = candidates_by_bioguide.get(bioguide_id)
        if candidate is None:
            if bioguide_id in existing_by_bioguide:
                del existing_by_bioguide[bioguide_id]
                removed_due_to_empty += 1
            continue

        existing = existing_by_bioguide.get(bioguide_id)
        existing_hash = normalize_string(existing.get("summaryHash")) if isinstance(existing, dict) else ""
        if (
            isinstance(existing, dict)
            and not overwrite
            and existing_hash == candidate.summary_hash
            and has_embedding(existing)
        ):
            existing["summary"] = candidate.summary
            existing["evidence"] = candidate.evidence
            existing["sourceIssueFile"] = candidate.source_issue_file
            existing["summaryHash"] = candidate.summary_hash
            existing["updatedAt"] = utc_now_iso()
            reused_embedding += 1
            continue

        pending.append(candidate)

    new_embeddings = 0
    with tqdm(total=len(pending), desc=f"{topic} embeddings", unit="summary") as pbar:
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            texts = [candidate.text_for_embedding for candidate in batch]
            vectors = client.embed_texts(texts, max_length=max_length, batch_size=batch_size)

            for candidate, vector in zip(batch, vectors):
                existing_by_bioguide[candidate.bioguide_id] = {
                    "bioguideId": candidate.bioguide_id,
                    "summary": candidate.summary,
                    "evidence": candidate.evidence,
                    "sourceIssueFile": candidate.source_issue_file,
                    "summaryHash": candidate.summary_hash,
                    "updatedAt": utc_now_iso(),
                    "embedding": vector,
                }
                new_embeddings += 1

            pbar.update(len(batch))

            if (new_embeddings % flush_every == 0) or (i + batch_size >= len(pending)):
                payload = sorted(
                    existing_by_bioguide.values(),
                    key=lambda entry: str(entry.get("bioguideId") or ""),
                )
                write_json_atomic(output_path, payload)

    payload = sorted(
        existing_by_bioguide.values(),
        key=lambda entry: str(entry.get("bioguideId") or ""),
    )
    write_json_atomic(output_path, payload)

    return {
        "scope_members": len(members_in_scope),
        "candidate_count": len(candidates_by_bioguide),
        "existing_count": len(existing_entries),
        "removed_due_to_empty": removed_due_to_empty,
        "reused_embedding": reused_embedding,
        "new_embeddings": new_embeddings,
        "final_count": len(payload),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed issue-profile summaries into per-topic JSON array files (resumable)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with *_issue_profile.json files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for per-topic embedding files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--bioguide-id",
        action="append",
        default=[],
        help="Process only this member ID. Can be passed multiple times.",
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        help="Process only this issue topic key. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Embed entries with empty summaries using a placeholder sentence.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_SAFE_LENGTH,
        help=f"Max tokenizer length passed to embedding model (default: {MAX_SAFE_LENGTH}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size per model call.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=64,
        help="Write topic files after this many new embeddings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-embed scoped entries even if summary hash is unchanged.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Optional model device override: "cuda", "cpu", or "hf_api".',
    )
    return parser.parse_args()


def main() -> None:
    configure_stdio_utf8()
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.flush_every < 1:
        raise ValueError("--flush-every must be >= 1")
    if args.max_length < 1:
        raise ValueError("--max-length must be >= 1")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    all_topics = get_issue_topics()
    requested_topics = [normalize_string(topic) for topic in args.topic if normalize_string(topic)]
    if requested_topics:
        invalid_topics = sorted({topic for topic in requested_topics if topic not in all_topics})
        if invalid_topics:
            raise ValueError(f"Unknown topic(s): {invalid_topics}. Valid topics: {all_topics}")
        topics = [topic for topic in all_topics if topic in set(requested_topics)]
    else:
        topics = all_topics

    requested_ids = {
        normalize_string(bioguide_id).upper()
        for bioguide_id in args.bioguide_id
        if normalize_string(bioguide_id)
    }
    profile_files = discover_issue_profile_files(args.input_dir, requested_ids=requested_ids)
    if not profile_files:
        if requested_ids:
            raise ValueError(f"No issue profile files found for requested bioguide IDs: {sorted(requested_ids)}")
        raise ValueError(f"No issue profile files found in: {args.input_dir}")

    print("=" * 72)
    print("Issue Profile Topic Embedding Export")
    print("=" * 72)
    print(f"Issue profile input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Members in scope: {len(profile_files)}")
    print(f"Topics in scope: {len(topics)}")

    candidates_by_topic, members_by_topic, source_entries, skipped_empty = collect_topic_candidates(
        profile_files=profile_files,
        topics=topics,
        include_empty=args.include_empty,
    )
    candidate_count = sum(len(topic_candidates) for topic_candidates in candidates_by_topic.values())
    print(f"Member-topic rows scanned: {source_entries}")
    print(f"Candidate embeddings: {candidate_count}")
    print(f"Skipped empty summaries: {skipped_empty}")

    print("Initializing embedding client...")
    client = EmbeddingGemmaClient(device=args.device)
    print(f"Mode: {client.mode} | Embedding dim: {client.embedding_dim} | Model: {MODEL_NAME}")
    effective_max_length = min(args.max_length, MAX_SAFE_LENGTH)
    if effective_max_length != args.max_length:
        print(
            f"Requested max length {args.max_length} exceeds safe cap {MAX_SAFE_LENGTH} "
            f"for model {MODEL_NAME}; using {effective_max_length}."
        )

    total_existing = 0
    total_removed_empty = 0
    total_reused = 0
    total_new = 0
    total_final = 0

    for topic in topics:
        stats = process_topic(
            topic=topic,
            members_in_scope=members_by_topic[topic],
            candidates_by_bioguide=candidates_by_topic[topic],
            client=client,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_length=effective_max_length,
            flush_every=args.flush_every,
            overwrite=args.overwrite,
        )
        total_existing += stats["existing_count"]
        total_removed_empty += stats["removed_due_to_empty"]
        total_reused += stats["reused_embedding"]
        total_new += stats["new_embeddings"]
        total_final += stats["final_count"]

    print("\n" + "=" * 72)
    print("Complete")
    print("=" * 72)
    print(f"Topics processed: {len(topics)}")
    print(f"Members in scope: {len(profile_files)}")
    print(f"Existing entries seen (sum across topic files): {total_existing}")
    print(f"Removed stale empty-summary entries: {total_removed_empty}")
    print(f"Existing embeddings reused: {total_reused}")
    print(f"New embeddings created: {total_new}")
    print(f"Final entries written (sum across topic files): {total_final}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
