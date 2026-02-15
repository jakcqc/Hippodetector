from typing import Optional, Literal
from pydantic import BaseModel


StanceStatus = Literal[
    "no_stance_recorded",
    "supports",
    "opposes",
    "mixed",
    "unknown"
]


class IssueStance(BaseModel):
    status: StanceStatus = "no_stance_recorded"
    summary: Optional[str] = None
    source_url: Optional[str] = None


class CandidateIssueProfile(BaseModel):
    abortion: IssueStance = IssueStance()
    budget_economy: IssueStance = IssueStance()
    civil_rights: IssueStance = IssueStance()
    corporations: IssueStance = IssueStance()
    crime: IssueStance = IssueStance()
    drugs: IssueStance = IssueStance()
    education: IssueStance = IssueStance()
    energy_oil: IssueStance = IssueStance()
    environment: IssueStance = IssueStance()
    families_children: IssueStance = IssueStance()
    foreign_policy: IssueStance = IssueStance()
    free_trade: IssueStance = IssueStance()
    government_reform: IssueStance = IssueStance()
    gun_control: IssueStance = IssueStance()
    health_care: IssueStance = IssueStance()
    homeland_security: IssueStance = IssueStance()
    immigration: IssueStance = IssueStance()
    jobs: IssueStance = IssueStance()
    principles_values: IssueStance = IssueStance()
    social_security: IssueStance = IssueStance()
    tax_reform: IssueStance = IssueStance()
    technology: IssueStance = IssueStance()
    war_peace: IssueStance = IssueStance()
    welfare_poverty: IssueStance = IssueStance()
