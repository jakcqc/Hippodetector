"""
Structured schema for contradiction detection results.

Defines the output format for detected contradictions between
a member's stated positions and their voting record.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


ContradictionSeverity = Literal["direct", "moderate", "weak", "nuanced"]


class VoteEvidence(BaseModel):
    """Evidence from a congressional vote."""

    billId: str = Field(description="Bill identifier (e.g., '119-hr-3424')")
    title: str = Field(description="Bill title")
    summary: str = Field(description="Bill summary")
    subjects: List[str] = Field(description="Bill subject tags")
    memberVote: str = Field(description="How member voted: 'Yea', 'Nay', etc.")
    voteDate: str = Field(description="Vote date (ISO format)")
    billUrl: str = Field(description="Congress.gov URL for bill")
    congress: int = Field(description="Congress number")
    session: int = Field(description="Session number")
    rollCall: int = Field(description="Roll call number")


class StatementEvidence(BaseModel):
    """Evidence from a press release or public statement."""

    id: str = Field(description="Press release identifier")
    title: str = Field(description="Press release title")
    date: str = Field(description="Publication date")
    url: str = Field(description="URL to press release")
    excerpt: str = Field(description="Relevant excerpt from statement")
    extractedStance: str = Field(
        description="Extracted stance: 'supports', 'opposes', 'mixed', etc."
    )
    stanceSummary: Optional[str] = Field(
        default=None,
        description="Summary of the position stated in the press release"
    )


class Contradiction(BaseModel):
    """A detected contradiction between stated position and voting record."""

    contradictionId: str = Field(description="Unique identifier for this contradiction")
    issueCategory: str = Field(
        description="Issue category (e.g., 'health_care', 'immigration')"
    )
    issueName: str = Field(
        description="Human-readable issue name (e.g., 'Health Care')"
    )
    severity: ContradictionSeverity = Field(
        description="How strong the contradiction is"
    )

    # Evidence
    statement: StatementEvidence = Field(
        description="What they said publicly"
    )
    vote: VoteEvidence = Field(
        description="How they voted"
    )

    # Analysis
    explanation: str = Field(
        description="LLM-generated explanation of the contradiction"
    )
    confidenceScore: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this contradiction (0.0-1.0)"
    )

    # Context
    topicMatch: str = Field(
        description="How bill subjects match issue category"
    )
    detectedAt: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="When this contradiction was detected"
    )


class ContradictionReport(BaseModel):
    """Full report of contradictions for a query."""

    query: str = Field(description="Original user query")
    memberName: str = Field(description="Member name")
    bioguideId: str = Field(description="Member bioguide ID")
    party: str = Field(description="Member party")
    state: str = Field(description="Member state")

    contradictions: List[Contradiction] = Field(
        description="List of detected contradictions"
    )
    totalFound: int = Field(description="Total number of contradictions found")

    # Search metadata
    billsSearched: int = Field(description="Number of bills retrieved")
    statementsSearched: int = Field(description="Number of statements retrieved")
    executionTimeMs: float = Field(description="Query execution time in milliseconds")


# Example usage for documentation
EXAMPLE_CONTRADICTION = Contradiction(
    contradictionId="contr-2026-02-15-healthcare-001",
    issueCategory="health_care",
    issueName="Health Care",
    severity="direct",
    statement=StatementEvidence(
        id="pr-2026-01-15-burlison",
        title="Burlison Announces Hearing on Making Housing More Affordable",
        date="2026-01-15",
        url="https://burlison.house.gov/media/press-releases/...",
        excerpt="Years of Democratic leadership imposed burdensome federal regulations...",
        extractedStance="opposes",
        stanceSummary="Opposes federal regulations that increase costs"
    ),
    vote=VoteEvidence(
        billId="119-hr-3424",
        title="SPACE Act of 2025",
        summary="This bill directs GSA to collaborate with federal agencies...",
        subjects=["Government buildings, facilities, and property"],
        memberVote="Yea",
        voteDate="2025-09-08T18:56:00-04:00",
        billUrl="https://www.congress.gov/bill/119th-congress/house-bill/3424",
        congress=119,
        session=1,
        rollCall=240
    ),
    explanation=(
        "Representative Burlison publicly criticized federal regulations as burdensome "
        "in his January 2026 press release, yet voted 'Yea' on the SPACE Act which "
        "expands federal agency collaboration requirements. While the contexts differ "
        "(housing vs. federal space management), both involve increased federal oversight."
    ),
    confidenceScore=0.75,
    topicMatch="Both relate to federal regulations and government operations",
    detectedAt="2026-02-15T10:30:00Z"
)


EXAMPLE_REPORT = ContradictionReport(
    query="Find contradictions about federal regulations",
    memberName="Burlison, Eric",
    bioguideId="B001316",
    party="Republican",
    state="Missouri",
    contradictions=[EXAMPLE_CONTRADICTION],
    totalFound=1,
    billsSearched=25,
    statementsSearched=3,
    executionTimeMs=1542.3
)
