"""
Topic Matching: Map bill subjects to issue categories.

This module maps Congress.gov bill subject tags to the standardized
24 issue categories used in stance extraction, enabling comparison
between bills and press releases on the same topics.

Usage:
    from RAG.topic_matching import map_bill_subjects_to_issues

    subjects = ["Health care costs and insurance", "Medicare"]
    issues = map_bill_subjects_to_issues(subjects)
    # Returns: ["health_care"]
"""

from typing import List


# Bill subject to issue category mapping
# Maps common bill subjects to our predefined issue categories
SUBJECT_TO_ISSUE_MAPPING = {
    # Health
    "Health": "health_care",
    "Medicare": "health_care",
    "Medicaid": "health_care",
    "Health care costs and insurance": "health_care",
    "Mental health": "health_care",

    # Economy & Budget
    "Economics and public finance": "budget_economy",
    "Budget process": "budget_economy",
    "Government spending and payments": "budget_economy",
    "Inflation and prices": "budget_economy",

    # Tax
    "Taxation": "tax_reform",
    "Income tax": "tax_reform",
    "Tax administration and collection": "tax_reform",

    # Immigration
    "Immigration": "immigration",
    "Border security and unlawful immigration": "immigration",
    "Citizenship and naturalization": "immigration",

    # Crime & Law Enforcement
    "Crime and law enforcement": "crime",
    "Criminal investigation, prosecution, interrogation": "crime",
    "Law enforcement administration and funding": "crime",

    # Gun Control
    "Firearms and explosives": "gun_control",

    # Environment & Energy
    "Environmental protection": "environment",
    "Climate change and greenhouse gases": "environment",
    "Energy": "energy_oil",
    "Oil and gas": "energy_oil",
    "Alternative and renewable resources": "energy_oil",

    # Education
    "Education": "education",
    "Elementary and secondary education": "education",
    "Higher education": "education",

    # Foreign Policy
    "International affairs": "foreign_policy",
    "Diplomacy, foreign officials, Americans abroad": "foreign_policy",
    "Military assistance, sales, and agreements": "foreign_policy",

    # Trade
    "Foreign trade and international finance": "free_trade",
    "Trade agreements and negotiations": "free_trade",
    "Tariffs": "free_trade",

    # Social Security
    "Social security and elderly assistance": "social_security",

    # Welfare
    "Social welfare": "welfare_poverty",
    "Poverty and welfare assistance": "welfare_poverty",

    # Jobs & Labor
    "Labor and employment": "jobs",
    "Unemployment": "jobs",
    "Employment and training programs": "jobs",

    # Civil Rights
    "Civil rights and liberties, minority issues": "civil_rights",
    "Racial and ethnic relations": "civil_rights",
    "Sex, gender, sexual orientation discrimination": "civil_rights",

    # Government Reform
    "Government operations and politics": "government_reform",
    "Administrative law and regulatory procedures": "government_reform",
    "Congressional oversight": "government_reform",

    # Homeland Security
    "Emergency management": "homeland_security",
    "Terrorism": "homeland_security",
    "National security": "homeland_security",

    # Families & Children
    "Families": "families_children",
    "Child safety and welfare": "families_children",

    # Abortion (often under Health or Civil Rights in bills)
    "Abortion": "abortion",

    # Corporations
    "Business records": "corporations",
    "Corporate finance and management": "corporations",
}


def map_bill_subjects_to_issues(subjects: List[str]) -> List[str]:
    """
    Map bill subjects to issue categories.

    Args:
        subjects: List of bill subject strings from Congress.gov

    Returns:
        List of matched issue categories (deduplicated)

    Examples:
        >>> map_bill_subjects_to_issues(["Health care costs and insurance"])
        ['health_care']

        >>> map_bill_subjects_to_issues(["Medicare", "Social security and elderly assistance"])
        ['health_care', 'social_security']
    """
    matched_issues = set()

    for subject in subjects:
        # Direct match
        if subject in SUBJECT_TO_ISSUE_MAPPING:
            matched_issues.add(SUBJECT_TO_ISSUE_MAPPING[subject])
        else:
            # Fuzzy match - check if subject contains key phrase
            for subject_key, issue in SUBJECT_TO_ISSUE_MAPPING.items():
                if subject_key.lower() in subject.lower():
                    matched_issues.add(issue)
                    break

    return list(matched_issues)


def main():
    """Test topic matching with example bill subjects."""
    print("=" * 80)
    print("Bill Subject → Issue Category Mapping Test")
    print("=" * 80)

    test_subjects = [
        "Administrative law and regulatory procedures",
        "Health care costs and insurance",
        "Immigration",
        "Taxation"
    ]

    print(f"\nBill subjects: {test_subjects}")
    mapped_issues = map_bill_subjects_to_issues(test_subjects)
    print(f"Mapped to issues: {mapped_issues}")

    # Test fuzzy matching
    print("\n" + "=" * 80)
    print("Fuzzy Matching Test")
    print("=" * 80)

    fuzzy_subjects = [
        "Federal regulation of healthcare providers",  # Should match health_care
        "Border security measures",  # Should match immigration
        "Tax credits for small businesses"  # Should match tax_reform
    ]

    print(f"\nBill subjects: {fuzzy_subjects}")
    fuzzy_mapped = map_bill_subjects_to_issues(fuzzy_subjects)
    print(f"Mapped to issues: {fuzzy_mapped}")


if __name__ == "__main__":
    main()
