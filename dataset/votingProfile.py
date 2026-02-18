"""
Data models for politician voting records.

These models represent a minimal but meaningful structure for analyzing
how politicians voted and what the bills were about.
"""

from typing import List, Optional
from pydantic import BaseModel


class VoteRecord(BaseModel):
    """Single vote with bill context."""

    bill_id: str                    # e.g., "119-hr-3424"
    title: str                      # Bill title
    summary: Optional[str] = None   # Brief description of what bill does
    subjects: List[str] = []        # Topic tags (for categorization)
    vote: str                       # "Yea", "Nay", "Present", "Not Voting"
    date: str                       # Vote date (ISO format)


class VotingProfile(BaseModel):
    """Politician's complete voting record."""

    bioguide_id: str
    name: str
    congress: int
    total_votes: int
    votes: List[VoteRecord]
