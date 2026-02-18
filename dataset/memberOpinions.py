from pydantic import BaseModel


class IssueSummary(BaseModel):
    # Short stance writeup for an issue.
    summary: str = ""
    # Number of evidence items supporting the summary.
    evidence: int = 0


class CandidateIssueProfile(BaseModel):
    abortion: IssueSummary = IssueSummary()
    budget_economy: IssueSummary = IssueSummary()
    civil_rights: IssueSummary = IssueSummary()
    corporations: IssueSummary = IssueSummary()
    crime: IssueSummary = IssueSummary()
    drugs: IssueSummary = IssueSummary()
    education: IssueSummary = IssueSummary()
    energy_oil: IssueSummary = IssueSummary()
    environment: IssueSummary = IssueSummary()
    families_children: IssueSummary = IssueSummary()
    foreign_policy: IssueSummary = IssueSummary()
    free_trade: IssueSummary = IssueSummary()
    government_reform: IssueSummary = IssueSummary()
    gun_control: IssueSummary = IssueSummary()
    health_care: IssueSummary = IssueSummary()
    homeland_security: IssueSummary = IssueSummary()
    immigration: IssueSummary = IssueSummary()
    jobs: IssueSummary = IssueSummary()
    principles_values: IssueSummary = IssueSummary()
    social_security: IssueSummary = IssueSummary()
    tax_reform: IssueSummary = IssueSummary()
    technology: IssueSummary = IssueSummary()
    war_peace: IssueSummary = IssueSummary()
    welfare_poverty: IssueSummary = IssueSummary()
