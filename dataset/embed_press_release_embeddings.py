"""
Embed press releases from membersByBioguideId JSON into per-member JSON outputs.

Usage examples:
  uv run dataset/embed_press_release_embeddings.py
  uv run dataset/embed_press_release_embeddings.py --bioguide-id B001321
  uv run dataset/embed_press_release_embeddings.py --input data/press_releases_by_bioguide_test.json
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


DEFAULT_INPUT = PROJECT_ROOT / "data" / "press_releases_by_bioguide.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "press_release_embeddings"


@dataclass(frozen=True)
class EmbedCandidate:
    release_id: str
    url: str
    title: str
    date: str
    published_time: str
    text_for_embedding: str
    text_source: str
    body_char_count: int


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def stable_release_id(release: dict[str, Any]) -> str:
    url = normalize_string(release.get("url"))
    title = normalize_string(release.get("title"))
    published = normalize_string(release.get("publishedTime"))
    date = normalize_string(release.get("date"))
    key = "||".join([url, published, date, title])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def load_source_members(input_path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")

    members = payload.get("membersByBioguideId")
    if not isinstance(members, dict):
        raise ValueError("Input JSON must contain object key: membersByBioguideId")

    output: dict[str, dict[str, Any]] = {}
    for bioguide_id, member in members.items():
        if isinstance(member, dict):
            output[str(bioguide_id)] = member
    return output


def load_existing_member_output(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def create_candidate(release: dict[str, Any], max_body_chars: int) -> EmbedCandidate | None:
    release_id = stable_release_id(release)
    url = normalize_string(release.get("url"))
    title = normalize_string(release.get("title"))
    date = normalize_string(release.get("date"))
    published_time = normalize_string(release.get("publishedTime"))
    body_text = normalize_string(release.get("bodyText"))

    text_for_embedding = ""
    text_source = ""
    if body_text and len(body_text) <= max_body_chars:
        text_for_embedding = body_text
        text_source = "bodyText"
    elif title:
        text_for_embedding = title
        text_source = "title_fallback"
    else:
        return None

    return EmbedCandidate(
        release_id=release_id,
        url=url,
        title=title,
        date=date,
        published_time=published_time,
        text_for_embedding=text_for_embedding,
        text_source=text_source,
        body_char_count=len(body_text),
    )


def process_member(
    bioguide_id: str,
    member: dict[str, Any],
    client: EmbeddingGemmaClient,
    output_dir: Path,
    input_path: Path,
    max_body_chars: int,
    batch_size: int,
    max_length: int,
    flush_every: int,
    overwrite: bool,
) -> dict[str, int]:
    output_path = output_dir / f"{bioguide_id}.json"
    existing_payload = {} if overwrite else load_existing_member_output(output_path)
    existing_releases = existing_payload.get("pressReleases", []) if isinstance(existing_payload, dict) else []
    if not isinstance(existing_releases, list):
        existing_releases = []

    existing_ids = {
        str(item.get("releaseId"))
        for item in existing_releases
        if isinstance(item, dict) and item.get("releaseId")
    }

    all_releases = member.get("pressReleases", [])
    if not isinstance(all_releases, list):
        all_releases = []

    candidates: list[EmbedCandidate] = []
    skipped_empty = 0
    for release in all_releases:
        if not isinstance(release, dict):
            skipped_empty += 1
            continue
        candidate = create_candidate(release, max_body_chars=max_body_chars)
        if candidate is None:
            skipped_empty += 1
            continue
        if candidate.release_id in existing_ids:
            continue
        candidates.append(candidate)

    if overwrite:
        embedded_releases: list[dict[str, Any]] = []
    else:
        embedded_releases = existing_releases

    metadata = {
        "bioguideId": bioguide_id,
        "name": normalize_string(member.get("name")),
        "lastName": normalize_string(member.get("lastName")),
        "state": normalize_string(member.get("state")),
        "partyName": normalize_string(member.get("partyName")),
        "source": normalize_string(member.get("source")),
        "model": MODEL_NAME,
        "embeddingDimension": client.embedding_dim,
        "inputFile": str(input_path),
        "updatedAt": utc_now_iso(),
        "releaseCountInSource": len(all_releases),
    }

    payload: dict[str, Any] = {
        "metadata": metadata,
        "pressReleases": embedded_releases,
    }

    if not candidates:
        payload["metadata"]["embeddedCount"] = len(embedded_releases)
        payload["metadata"]["skippedEmptyCount"] = skipped_empty
        payload["metadata"]["newEmbeddingsAdded"] = 0
        write_json_atomic(output_path, payload)
        return {
            "source_count": len(all_releases),
            "existing_count": len(existing_releases),
            "skipped_empty": skipped_empty,
            "new_added": 0,
        }

    new_added = 0
    with tqdm(total=len(candidates), desc=f"{bioguide_id} embeddings", unit="release") as pbar:
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_texts = [c.text_for_embedding for c in batch]
            batch_embeddings = client.embed_texts(batch_texts, max_length=max_length, batch_size=batch_size)

            for candidate, vector in zip(batch, batch_embeddings):
                embedded_releases.append(
                    {
                        "releaseId": candidate.release_id,
                        "url": candidate.url,
                        "title": candidate.title,
                        "date": candidate.date,
                        "publishedTime": candidate.published_time,
                        "textSource": candidate.text_source,
                        "bodyCharCount": candidate.body_char_count,
                        "embedding": vector,
                    }
                )
                new_added += 1

            pbar.update(len(batch))

            if (new_added % flush_every == 0) or (i + batch_size >= len(candidates)):
                payload["metadata"]["updatedAt"] = utc_now_iso()
                payload["metadata"]["embeddedCount"] = len(embedded_releases)
                payload["metadata"]["skippedEmptyCount"] = skipped_empty
                payload["metadata"]["newEmbeddingsAdded"] = new_added
                write_json_atomic(output_path, payload)

    return {
        "source_count": len(all_releases),
        "existing_count": len(existing_releases),
        "skipped_empty": skipped_empty,
        "new_added": new_added,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed press releases into per-member JSON files (resumable)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSON with membersByBioguideId (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for per-member embedding JSON files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--bioguide-id",
        action="append",
        default=[],
        help="Process only this member ID. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-body-chars",
        type=int,
        default=24000,
        help="If bodyText exceeds this, fallback to title. If title empty, skip.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Max tokenizer length passed to embedding model.",
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
        help="Write output file after this many new embeddings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing output file and regenerate member embeddings from scratch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Optional model device override: "cuda", "cpu", or "hf_api".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.flush_every < 1:
        raise ValueError("--flush-every must be >= 1")
    if args.max_body_chars < 1:
        raise ValueError("--max-body-chars must be >= 1")
    if args.max_length < 1:
        raise ValueError("--max-length must be >= 1")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print("=" * 72)
    print("Press Release Embedding Export")
    print("=" * 72)
    print(f"Input: {args.input}")
    print(f"Output dir: {args.output_dir}")

    members = load_source_members(args.input)
    requested_ids = [str(x).strip() for x in args.bioguide_id if str(x).strip()]
    if requested_ids:
        members = {k: v for k, v in members.items() if k in set(requested_ids)}
        if not members:
            raise ValueError(f"No requested bioguide IDs found in input: {requested_ids}")

    print(f"Members to process: {len(members)}")
    print("Initializing embedding client...")
    client = EmbeddingGemmaClient(device=args.device)
    print(f"Mode: {client.mode} | Embedding dim: {client.embedding_dim}")

    total_source = 0
    total_existing = 0
    total_skipped_empty = 0
    total_new = 0

    member_items = sorted(members.items(), key=lambda item: item[0])
    for bioguide_id, member in tqdm(member_items, desc="Members", unit="member"):
        stats = process_member(
            bioguide_id=bioguide_id,
            member=member,
            client=client,
            output_dir=args.output_dir,
            input_path=args.input,
            max_body_chars=args.max_body_chars,
            batch_size=args.batch_size,
            max_length=args.max_length,
            flush_every=args.flush_every,
            overwrite=args.overwrite,
        )
        total_source += stats["source_count"]
        total_existing += stats["existing_count"]
        total_skipped_empty += stats["skipped_empty"]
        total_new += stats["new_added"]

    print("\n" + "=" * 72)
    print("Complete")
    print("=" * 72)
    print(f"Members processed: {len(member_items)}")
    print(f"Source releases seen: {total_source}")
    print(f"Existing embeddings reused: {total_existing}")
    print(f"Skipped (empty/unusable): {total_skipped_empty}")
    print(f"New embeddings created: {total_new}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
