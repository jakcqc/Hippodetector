from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EMBED_DIR_1 = DATA_DIR / "press_release_embeddings_1"
EMBED_DIR_2 = DATA_DIR / "press_release_embeddings_2"
OUTPUT_FILE = DATA_DIR / "press_release_embeddings_member_index.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_valid_embeddings(press_releases: Any) -> int:
    if not isinstance(press_releases, list):
        return 0
    total = 0
    for item in press_releases:
        if not isinstance(item, dict):
            continue
        emb = item.get("embedding")
        if isinstance(emb, list) and len(emb) > 0:
            total += 1
    return total


def build_index() -> dict[str, Any]:
    members: dict[str, dict[str, Any]] = {}
    source_dirs = [EMBED_DIR_1, EMBED_DIR_2]

    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for file_path in sorted(source_dir.glob("*.json")):
            member_id = file_path.stem.strip().upper()
            if not member_id:
                continue
            try:
                payload = load_json(file_path)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            valid_count = count_valid_embeddings(payload.get("pressReleases"))
            if valid_count <= 0:
                continue

            member_entry = members.get(member_id)
            if member_entry is None:
                member_entry = {
                    "bioguideId": member_id,
                    "name": str(metadata.get("name") or member_id),
                    "partyName": str(metadata.get("partyName") or "Unknown"),
                    "state": str(metadata.get("state") or "Unknown"),
                    "embeddingDimension": int(metadata.get("embeddingDimension") or 0),
                    "files": [],
                    "totalEmbeddingCount": 0,
                }
                members[member_id] = member_entry

            member_entry["embeddingDimension"] = int(metadata.get("embeddingDimension") or member_entry["embeddingDimension"])
            member_entry["name"] = str(metadata.get("name") or member_entry["name"])
            member_entry["partyName"] = str(metadata.get("partyName") or member_entry["partyName"])
            member_entry["state"] = str(metadata.get("state") or member_entry["state"])
            member_entry["files"].append(
                {
                    "path": str(file_path.relative_to(DATA_DIR)),
                    "sourceFolder": source_dir.name,
                    "embeddingCount": valid_count,
                    "mtimeNs": file_path.stat().st_mtime_ns,
                }
            )
            member_entry["totalEmbeddingCount"] += valid_count

    member_rows = sorted(members.values(), key=lambda row: (str(row.get("name") or ""), str(row.get("bioguideId") or "")))
    return {
        "generatedAtUtc": utc_now_iso(),
        "sourceFolders": [str(p.relative_to(PROJECT_ROOT)) for p in source_dirs if p.exists()],
        "membersWithEmbeddingsCount": len(member_rows),
        "members": member_rows,
    }


def main() -> None:
    index_payload = build_index()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(index_payload, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Members with embeddings: {index_payload['membersWithEmbeddingsCount']}")


if __name__ == "__main__":
    main()
