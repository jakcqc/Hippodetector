"""
Semantic search over congressional voting records and press releases.

Usage:
    from RAG.search import search_all

    results = search_all(
        query="What is the member's stance on immigration?",
        bioguide_id="B001316",  # Optional: filter by member
        top_k=10
    )
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Add parent directory to path to import embedding client
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from LLM.hf_embedding_gemma import EmbeddingGemmaClient

# Qdrant connection settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Collection names
BILLS_COLLECTION = "bills"
PRESS_RELEASES_COLLECTION = "press_releases"


class BillSearchResult(BaseModel):
    """A single bill search result from Qdrant."""

    # Similarity score
    score: float = Field(description="Cosine similarity score (0.0-1.0)")

    # Bill details
    billId: str
    congress: int
    type: str
    number: str
    title: str
    summary: str
    subjects: List[str]
    url: str

    # Vote details
    memberVote: str
    voteDate: str
    rollCall: int
    session: int
    result: str

    # Member details
    bioguideId: str
    memberName: str
    party: str
    state: str


class PressReleaseSearchResult(BaseModel):
    """A single press release search result from Qdrant."""

    # Similarity score
    score: float = Field(description="Cosine similarity score (0.0-1.0)")

    # Press release details
    id: str
    title: str
    date: str
    url: str
    bodyText: str
    topics: List[str]
    relatedBills: List[str]

    # Member details
    bioguideId: str
    memberName: str
    party: str
    state: str


class SearchResults(BaseModel):
    """Combined search results from both collections."""

    query: str
    bills: List[BillSearchResult]
    pressReleases: List[PressReleaseSearchResult]
    totalResults: int


def get_qdrant_client(host: str = QDRANT_HOST, port: int = QDRANT_PORT) -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(host=host, port=port)


def search_bills(
    query: str,
    embedding_client: EmbeddingGemmaClient,
    qdrant_client: QdrantClient,
    top_k: int = 10,
    bioguide_id: Optional[str] = None,
    min_score: float = 0.0,
) -> List[BillSearchResult]:
    """
    Search bills collection for relevant voting records.

    Args:
        query: Natural language query
        embedding_client: Embedding model client
        qdrant_client: Qdrant database client
        top_k: Number of results to return
        bioguide_id: Optional filter by member bioguide ID
        min_score: Minimum similarity score (0.0-1.0)

    Returns:
        List of bill search results, sorted by relevance
    """
    # Generate query embedding
    query_vector = embedding_client.embed_text(query)

    # Build filter if bioguide_id specified
    search_filter = None
    if bioguide_id:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="bioguideId",
                    match=MatchValue(value=bioguide_id)
                )
            ]
        )

    # Search Qdrant
    search_results = qdrant_client.query_points(
        collection_name=BILLS_COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        score_threshold=min_score,
    ).points

    # Convert to structured results
    results = []
    for hit in search_results:
        result = BillSearchResult(
            score=hit.score,
            billId=hit.payload.get("billId", ""),
            congress=hit.payload.get("congress", 0),
            type=hit.payload.get("type", ""),
            number=hit.payload.get("number", ""),
            title=hit.payload.get("title", ""),
            summary=hit.payload.get("summary", ""),
            subjects=hit.payload.get("subjects", []),
            url=hit.payload.get("url", ""),
            memberVote=hit.payload.get("memberVote", ""),
            voteDate=hit.payload.get("voteDate", ""),
            rollCall=hit.payload.get("rollCall", 0),
            session=hit.payload.get("session", 0),
            result=hit.payload.get("result", ""),
            bioguideId=hit.payload.get("bioguideId", ""),
            memberName=hit.payload.get("memberName", ""),
            party=hit.payload.get("party", ""),
            state=hit.payload.get("state", ""),
        )
        results.append(result)

    return results


def search_press_releases(
    query: str,
    embedding_client: EmbeddingGemmaClient,
    qdrant_client: QdrantClient,
    top_k: int = 10,
    bioguide_id: Optional[str] = None,
    min_score: float = 0.0,
) -> List[PressReleaseSearchResult]:
    """
    Search press releases collection for relevant public statements.

    Args:
        query: Natural language query
        embedding_client: Embedding model client
        qdrant_client: Qdrant database client
        top_k: Number of results to return
        bioguide_id: Optional filter by member bioguide ID
        min_score: Minimum similarity score (0.0-1.0)

    Returns:
        List of press release search results, sorted by relevance
    """
    # Generate query embedding
    query_vector = embedding_client.embed_text(query)

    # Build filter if bioguide_id specified
    search_filter = None
    if bioguide_id:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="bioguideId",
                    match=MatchValue(value=bioguide_id)
                )
            ]
        )

    # Search Qdrant
    search_results = qdrant_client.query_points(
        collection_name=PRESS_RELEASES_COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        score_threshold=min_score,
    ).points

    # Convert to structured results
    results = []
    for hit in search_results:
        result = PressReleaseSearchResult(
            score=hit.score,
            id=hit.payload.get("id", ""),
            title=hit.payload.get("title", ""),
            date=hit.payload.get("date", ""),
            url=hit.payload.get("url", ""),
            bodyText=hit.payload.get("bodyText", ""),
            topics=hit.payload.get("topics", []),
            relatedBills=hit.payload.get("relatedBills", []),
            bioguideId=hit.payload.get("bioguideId", ""),
            memberName=hit.payload.get("memberName", ""),
            party=hit.payload.get("party", ""),
            state=hit.payload.get("state", ""),
        )
        results.append(result)

    return results


def search_all(
    query: str,
    bioguide_id: Optional[str] = None,
    top_k_bills: int = 10,
    top_k_press_releases: int = 5,
    min_score: float = 0.0,
    embedding_client: Optional[EmbeddingGemmaClient] = None,
    qdrant_client: Optional[QdrantClient] = None,
) -> SearchResults:
    """
    Search both bills and press releases for a query.

    Args:
        query: Natural language query
        bioguide_id: Optional filter by member bioguide ID
        top_k_bills: Number of bill results to return
        top_k_press_releases: Number of press release results to return
        min_score: Minimum similarity score (0.0-1.0)
        embedding_client: Optional embedding client (creates new if None)
        qdrant_client: Optional Qdrant client (creates new if None)

    Returns:
        SearchResults with bills and press releases
    """
    # Initialize clients if not provided
    if embedding_client is None:
        embedding_client = EmbeddingGemmaClient()
    if qdrant_client is None:
        qdrant_client = get_qdrant_client()

    # Search both collections
    bills = search_bills(
        query=query,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        top_k=top_k_bills,
        bioguide_id=bioguide_id,
        min_score=min_score,
    )

    press_releases = search_press_releases(
        query=query,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        top_k=top_k_press_releases,
        bioguide_id=bioguide_id,
        min_score=min_score,
    )

    return SearchResults(
        query=query,
        bills=bills,
        pressReleases=press_releases,
        totalResults=len(bills) + len(press_releases),
    )


def main() -> None:
    """Test semantic search with example queries."""
    import argparse

    parser = argparse.ArgumentParser(description="Test semantic search")
    parser.add_argument(
        "--query",
        type=str,
        default="What is the member's stance on federal regulations?",
        help="Search query"
    )
    parser.add_argument(
        "--bioguide-id",
        type=str,
        default="B001316",
        help="Bioguide ID to search (default: B001316 - Burlison)"
    )
    parser.add_argument(
        "--top-k-bills",
        type=int,
        default=5,
        help="Number of bill results"
    )
    parser.add_argument(
        "--top-k-pr",
        type=int,
        default=3,
        help="Number of press release results"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Semantic Search Test")
    print("=" * 80)
    print(f"\nQuery: {args.query}")
    print(f"Member: {args.bioguide_id}")
    print(f"Top bills: {args.top_k_bills}, Top press releases: {args.top_k_pr}")

    # Initialize clients
    print("\nInitializing embedding model...")
    embedding_client = EmbeddingGemmaClient()

    print("Connecting to Qdrant...")
    qdrant_client = get_qdrant_client()

    # Run search
    print("\nSearching...")
    results = search_all(
        query=args.query,
        bioguide_id=args.bioguide_id,
        top_k_bills=args.top_k_bills,
        top_k_press_releases=args.top_k_pr,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )

    # Display results
    print("\n" + "=" * 80)
    print(f"Results: {results.totalResults} total")
    print("=" * 80)

    print(f"\n📋 Bills ({len(results.bills)} results):")
    print("-" * 80)
    for i, bill in enumerate(results.bills, 1):
        print(f"\n{i}. {bill.title}")
        print(f"   Score: {bill.score:.4f}")
        print(f"   Bill ID: {bill.billId}")
        print(f"   Vote: {bill.memberVote} on {bill.voteDate}")
        print(f"   Subjects: {', '.join(bill.subjects[:3])}")
        if bill.summary:
            summary_preview = bill.summary[:150] + "..." if len(bill.summary) > 150 else bill.summary
            print(f"   Summary: {summary_preview}")

    print(f"\n\n📰 Press Releases ({len(results.pressReleases)} results):")
    print("-" * 80)
    for i, pr in enumerate(results.pressReleases, 1):
        print(f"\n{i}. {pr.title}")
        print(f"   Score: {pr.score:.4f}")
        print(f"   Date: {pr.date}")
        print(f"   URL: {pr.url}")
        if pr.bodyText:
            text_preview = pr.bodyText[:200] + "..." if len(pr.bodyText) > 200 else pr.bodyText
            print(f"   Preview: {text_preview}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
