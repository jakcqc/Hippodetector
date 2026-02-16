"""
Set up Qdrant collections for Hippodetector RAG system.

Usage:
    # Start Qdrant first: docker-compose up -d
    python RAG/setup_qdrant.py
"""

from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Qdrant connection settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Collection names
BILLS_COLLECTION = "bills"
PRESS_RELEASES_COLLECTION = "press_releases"

# Embedding dimension for google/embeddinggemma-300m
# The model has hidden_size=768 (standard BERT-style architecture)
EMBEDDING_DIM = 768


def get_qdrant_client(host: str = QDRANT_HOST, port: int = QDRANT_PORT) -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(host=host, port=port)


def create_bills_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Create collection for bill embeddings.

    Data structure: Joins votes + bills from member profile
    Embedding text: title + summary + subjects (from bills array)

    Metadata stored with each bill (aligned with member_profile_schema.json):
    From bills array:
    - billId: str (e.g., "119-hr-3424")
    - congress: int
    - type: str (e.g., "HR", "S")
    - number: str
    - title: str
    - summary: str (full text)
    - subjects: list[str] (topic tags)
    - cosponsors: int
    - url: str (Congress.gov URL)
    - latestAction: dict (actionDate, text)

    From votes array (joined by billId):
    - memberVote: str ("Yea", "Nay", "Present", "Not Voting")
    - voteDate: str (ISO timestamp)
    - rollCall: int
    - session: int
    - result: str (overall vote result)

    From metadata:
    - bioguideId: str (member who voted)
    - memberName: str
    - party: str
    - state: str
    """
    if recreate and client.collection_exists(BILLS_COLLECTION):
        print(f"Deleting existing collection: {BILLS_COLLECTION}")
        client.delete_collection(BILLS_COLLECTION)

    if not client.collection_exists(BILLS_COLLECTION):
        print(f"Creating collection: {BILLS_COLLECTION}")
        client.create_collection(
            collection_name=BILLS_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        print(f"✓ Created {BILLS_COLLECTION} collection")
    else:
        print(f"✓ Collection {BILLS_COLLECTION} already exists")


def create_press_releases_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Create collection for press release embeddings.

    Data structure: From pressReleases array in member profile
    Embedding text: title + bodyText

    Metadata stored with each press release (aligned with member_profile_schema.json):
    From pressReleases array:
    - id: str (unique press release identifier)
    - title: str
    - date: str (publication date)
    - publishedTime: str (ISO timestamp, optional)
    - url: str (member website URL)
    - bodyText: str (full plain text content)
    - topics: list[str] (extracted topics/themes)
    - relatedBills: list[str] (bill IDs mentioned, e.g., ["119-hr-3424"])

    From metadata:
    - bioguideId: str (member who issued it)
    - memberName: str
    - party: str
    - state: str
    """
    if recreate and client.collection_exists(PRESS_RELEASES_COLLECTION):
        print(f"Deleting existing collection: {PRESS_RELEASES_COLLECTION}")
        client.delete_collection(PRESS_RELEASES_COLLECTION)

    if not client.collection_exists(PRESS_RELEASES_COLLECTION):
        print(f"Creating collection: {PRESS_RELEASES_COLLECTION}")
        client.create_collection(
            collection_name=PRESS_RELEASES_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        print(f"✓ Created {PRESS_RELEASES_COLLECTION} collection")
    else:
        print(f"✓ Collection {PRESS_RELEASES_COLLECTION} already exists")


def test_connection(client: QdrantClient) -> bool:
    """Test Qdrant connection and print server info."""
    try:
        # Get collections list
        collections = client.get_collections()
        print("\n✓ Successfully connected to Qdrant")
        print(f"  Host: {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"  Collections: {len(collections.collections)}")

        for collection in collections.collections:
            info = client.get_collection(collection.name)
            print(f"    - {collection.name}: {info.points_count} points")

        return True
    except Exception as e:
        print(f"\n✗ Failed to connect to Qdrant: {e}")
        print("\nMake sure Qdrant is running:")
        print("  docker-compose up -d")
        return False


def main() -> None:
    """Main setup function."""
    print("=" * 60)
    print("Hippodetector RAG System - Qdrant Setup")
    print("=" * 60)

    # Connect to Qdrant
    print("\nConnecting to Qdrant...")
    client = get_qdrant_client()

    # Test connection
    if not test_connection(client):
        return

    # Create collections
    print("\nSetting up collections...")
    create_bills_collection(client, recreate=False)
    create_press_releases_collection(client, recreate=False)

    # Final status
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Generate embeddings for Burlison's data")
    print("  2. Load embeddings into Qdrant")
    print("  3. Test semantic search")


if __name__ == "__main__":
    main()
