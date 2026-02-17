"""
Load member profile data into Qdrant vector database.

Usage:
    uv run RAG/load_embeddings.py --bioguide-id B001316
"""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Add parent directory to path to import embedding client
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from LLM.hf_embedding_gemma import EmbeddingGemmaClient

# Qdrant connection settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Collection names
BILLS_COLLECTION = "bills"
PRESS_RELEASES_COLLECTION = "press_releases"

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "data" / "profiles"
DATA_DIR = PROJECT_ROOT / "data"

# Precomputed embeddings zip files
PR_EMBEDDINGS_ZIPS = [
    DATA_DIR / "press_release_embeddings_1.zip",
    DATA_DIR / "press_release_embeddings_2.zip",
]


def strip_html(text: str) -> str:
    """Strip HTML tags from text."""
    if not text:
        return ""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    # Replace multiple whitespace with single space
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()


def load_member_profile(bioguide_id: str) -> Dict[str, Any]:
    """Load member profile from JSON file."""
    profile_path = PROFILES_DIR / f"{bioguide_id}.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Member profile not found: {profile_path}\n"
            f"Run: python dataset/build_member_profile.py --bioguide-id {bioguide_id}"
        )
    return json.loads(profile_path.read_text(encoding="utf-8"))


def join_vote_with_bill(vote: Dict[str, Any], bills: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Join a vote record with its corresponding bill details."""
    bill_id = vote.get("billId")
    if not bill_id:
        return None

    # Find matching bill
    for bill in bills:
        if bill.get("billId") == bill_id:
            return {
                # Bill details
                "billId": bill_id,
                "congress": bill.get("congress"),
                "type": bill.get("type"),
                "number": bill.get("number"),
                "title": bill.get("title", ""),
                "summary": bill.get("summary", ""),
                "subjects": bill.get("subjects", []),
                "cosponsors": bill.get("cosponsors", 0),
                "url": bill.get("url", ""),
                "latestAction": bill.get("latestAction", {}),
                # Vote details
                "memberVote": vote.get("memberVote"),
                "voteDate": vote.get("date"),
                "rollCall": vote.get("rollCall"),
                "session": vote.get("session"),
                "result": vote.get("result"),
            }
    return None


def create_bill_embedding_text(bill_vote: Dict[str, Any]) -> str:
    """
    Create text for bill embedding.

    Format: Title + Summary (HTML stripped) + Subjects + Vote position
    """
    parts = []

    # Title
    title = bill_vote.get("title", "")
    if title:
        parts.append(f"Title: {title}")

    # Summary (strip HTML)
    summary = strip_html(bill_vote.get("summary", ""))
    if summary:
        parts.append(f"Summary: {summary}")

    # Subjects
    subjects = bill_vote.get("subjects", [])
    if subjects:
        subjects_text = ", ".join(subjects)
        parts.append(f"Topics: {subjects_text}")

    # Vote position
    member_vote = bill_vote.get("memberVote")
    if member_vote:
        parts.append(f"Vote: {member_vote}")

    return "\n\n".join(parts)


def create_press_release_embedding_text(pr: Dict[str, Any]) -> str:
    """
    Create text for press release embedding.

    Format: Title + Body text
    """
    parts = []

    # Title
    title = pr.get("title", "")
    if title:
        parts.append(f"Title: {title}")

    # Body text
    body = pr.get("bodyText", "")
    if body:
        parts.append(f"Content: {body}")

    return "\n\n".join(parts)


def load_bills_into_qdrant(
    profile: Dict[str, Any],
    embedding_client: EmbeddingGemmaClient,
    qdrant_client: QdrantClient
) -> int:
    """Load bill embeddings into Qdrant."""
    metadata = profile.get("metadata", {})
    votes = profile.get("votes", [])
    bills = profile.get("bills", [])

    print(f"\nProcessing {len(votes)} votes with {len(bills)} bills...")

    # Join votes with bills
    bill_votes = []
    for vote in votes:
        joined = join_vote_with_bill(vote, bills)
        if joined:
            bill_votes.append(joined)

    print(f"Successfully joined {len(bill_votes)} bill-vote pairs")

    if not bill_votes:
        print("No bills to load")
        return 0

    # Generate embeddings
    print("Generating embeddings...")
    embedding_texts = [create_bill_embedding_text(bv) for bv in bill_votes]
    embeddings = embedding_client.embed_texts(embedding_texts)

    print(f"Generated {len(embeddings)} embeddings")

    # Create points for Qdrant
    points = []
    for bill_vote, embedding in zip(bill_votes, embeddings):
        # Create payload with all metadata
        payload = {
            # Bill details
            "billId": bill_vote.get("billId"),
            "congress": bill_vote.get("congress"),
            "type": bill_vote.get("type"),
            "number": bill_vote.get("number"),
            "title": bill_vote.get("title"),
            "summary": strip_html(bill_vote.get("summary", "")),
            "subjects": bill_vote.get("subjects"),
            "cosponsors": bill_vote.get("cosponsors"),
            "url": bill_vote.get("url"),
            "latestAction": bill_vote.get("latestAction"),
            # Vote details
            "memberVote": bill_vote.get("memberVote"),
            "voteDate": bill_vote.get("voteDate"),
            "rollCall": bill_vote.get("rollCall"),
            "session": bill_vote.get("session"),
            "result": bill_vote.get("result"),
            # Member details
            "bioguideId": metadata.get("bioguideId"),
            "memberName": metadata.get("name"),
            "party": metadata.get("party"),
            "state": metadata.get("state"),
        }

        point = PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload=payload
        )
        points.append(point)

    # Upload to Qdrant
    print(f"Uploading {len(points)} points to Qdrant...")
    qdrant_client.upsert(
        collection_name=BILLS_COLLECTION,
        points=points
    )

    print(f"✓ Loaded {len(points)} bill embeddings")
    return len(points)


def load_press_releases_into_qdrant(
    profile: Dict[str, Any],
    embedding_client: EmbeddingGemmaClient,
    qdrant_client: QdrantClient
) -> int:
    """Load press release embeddings into Qdrant."""
    metadata = profile.get("metadata", {})
    press_releases = profile.get("pressReleases", [])

    print(f"\nProcessing {len(press_releases)} press releases...")

    if not press_releases:
        print("No press releases to load")
        return 0

    # Generate embeddings
    print("Generating embeddings...")
    embedding_texts = [create_press_release_embedding_text(pr) for pr in press_releases]
    embeddings = embedding_client.embed_texts(embedding_texts)

    print(f"Generated {len(embeddings)} embeddings")

    # Create points for Qdrant
    points = []
    for pr, embedding in zip(press_releases, embeddings):
        # Create payload with all metadata
        payload = {
            # Press release details
            "id": pr.get("id"),
            "title": pr.get("title"),
            "date": pr.get("date"),
            "publishedTime": pr.get("publishedTime"),
            "url": pr.get("url"),
            "bodyText": pr.get("bodyText"),
            "topics": pr.get("topics", []),
            "relatedBills": pr.get("relatedBills", []),
            # Member details
            "bioguideId": metadata.get("bioguideId"),
            "memberName": metadata.get("name"),
            "party": metadata.get("party"),
            "state": metadata.get("state"),
        }

        point = PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload=payload
        )
        points.append(point)

    # Upload to Qdrant
    print(f"Uploading {len(points)} points to Qdrant...")
    qdrant_client.upsert(
        collection_name=PRESS_RELEASES_COLLECTION,
        points=points
    )

    print(f"✓ Loaded {len(points)} press release embeddings")
    return len(points)


def load_precomputed_pr_embeddings_into_qdrant(
    bioguide_id: str,
    qdrant_client: QdrantClient
) -> int:
    """Load pre-computed press release embeddings from zip files."""
    print(f"\nLoading pre-computed embeddings for {bioguide_id}...")

    # Find the member's embedding file in one of the zip archives
    pr_data = None
    for zip_path in PR_EMBEDDINGS_ZIPS:
        if not zip_path.exists():
            continue

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Try to find the bioguide file
                folder_name = zip_path.stem  # e.g., "press_release_embeddings_1"
                json_path = f"{folder_name}/{bioguide_id}.json"

                if json_path in zf.namelist():
                    print(f"Found embeddings in {zip_path.name}")
                    with zf.open(json_path) as f:
                        pr_data = json.load(f)
                    break
        except Exception as e:
            print(f"Warning: Could not read {zip_path.name}: {e}")
            continue

    if not pr_data:
        print(f"⚠️  No pre-computed embeddings found for {bioguide_id}")
        print("   Falling back to generating fresh embeddings...")
        return -1  # Signal to fall back

    # Extract metadata and press releases
    metadata = pr_data.get("metadata", {})
    press_releases = pr_data.get("pressReleases", [])

    print(f"Found {len(press_releases)} press releases with embeddings")

    if not press_releases:
        print("No press releases to load")
        return 0

    # Create points for Qdrant
    points = []
    for pr in press_releases:
        # Check if embedding exists
        embedding = pr.get("embedding")
        if not embedding:
            print(f"Warning: PR '{pr.get('title', 'Unknown')[:50]}...' missing embedding, skipping")
            continue

        # Create payload
        payload = {
            # Press release details
            "id": pr.get("releaseId", pr.get("id")),
            "title": pr.get("title", ""),
            "date": pr.get("date", ""),
            "publishedTime": pr.get("publishedTime", ""),
            "url": pr.get("url", ""),
            "bodyText": pr.get("textSource", pr.get("bodyText", "")),
            "topics": pr.get("topics", []),
            "relatedBills": pr.get("relatedBills", []),
            # Member details
            "bioguideId": bioguide_id,
            "memberName": metadata.get("name", ""),
            "party": metadata.get("party", ""),
            "state": metadata.get("state", ""),
        }

        point = PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload=payload
        )
        points.append(point)

    # Upload to Qdrant
    print(f"Uploading {len(points)} points to Qdrant...")
    qdrant_client.upsert(
        collection_name=PRESS_RELEASES_COLLECTION,
        points=points
    )

    print(f"✓ Loaded {len(points)} pre-computed press release embeddings")
    return len(points)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load member profile data into Qdrant vector database"
    )
    parser.add_argument(
        "--bioguide-id",
        type=str,
        required=True,
        help="Bioguide ID of the member (e.g., B001316 for Burlison)"
    )
    parser.add_argument(
        "--use-precomputed-pr",
        action="store_true",
        help="Use pre-computed PR embeddings from zip files (faster, saves LLM costs)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Loading Member Profile into Qdrant")
    print("=" * 60)

    # Load member profile
    print(f"\nLoading profile for {args.bioguide_id}...")
    profile = load_member_profile(args.bioguide_id)
    metadata = profile.get("metadata", {})
    data_collection = profile.get("dataCollection", {})

    print(f"Member: {metadata.get('name')}")
    print(f"Party: {metadata.get('party')}")
    print(f"State: {metadata.get('state')}")
    print(f"Votes: {data_collection.get('votesCount')}")
    print(f"Bills: {data_collection.get('billsCount')}")
    print(f"Press Releases: {data_collection.get('pressReleasesCount')}")

    # Initialize clients
    print("\nInitializing embedding model...")
    embedding_client = EmbeddingGemmaClient()
    print(f"Embedding dimension: {embedding_client.embedding_dim}")

    print("\nConnecting to Qdrant...")
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Load data
    bills_loaded = load_bills_into_qdrant(profile, embedding_client, qdrant_client)

    # Load press releases (precomputed or generate fresh)
    if args.use_precomputed_pr:
        print("\n" + "=" * 60)
        print("Using Pre-Computed Press Release Embeddings")
        print("=" * 60)
        pr_loaded = load_precomputed_pr_embeddings_into_qdrant(
            bioguide_id=args.bioguide_id,
            qdrant_client=qdrant_client
        )

        # Fall back to generating if precomputed not found
        if pr_loaded == -1:
            print("\nFalling back to generating fresh embeddings...")
            pr_loaded = load_press_releases_into_qdrant(profile, embedding_client, qdrant_client)
    else:
        pr_loaded = load_press_releases_into_qdrant(profile, embedding_client, qdrant_client)

    # Summary
    print("\n" + "=" * 60)
    print("Load Complete!")
    print("=" * 60)
    print(f"Bills loaded: {bills_loaded}")
    print(f"Press releases loaded: {pr_loaded}")
    print(f"Total embeddings: {bills_loaded + pr_loaded}")

    # Verify collections
    bills_collection = qdrant_client.get_collection(BILLS_COLLECTION)
    pr_collection = qdrant_client.get_collection(PRESS_RELEASES_COLLECTION)

    print("\nCollection status:")
    print(f"  {BILLS_COLLECTION}: {bills_collection.points_count} points")
    print(f"  {PRESS_RELEASES_COLLECTION}: {pr_collection.points_count} points")


if __name__ == "__main__":
    main()