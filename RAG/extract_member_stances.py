"""
Extract stances from all press releases for a member.

Usage:
    uv run python RAG/extract_member_stances.py --bioguide-id B001316
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RAG.extract_stances import extract_stance_from_text

PROFILES_DIR = Path(__file__).resolve().parents[1] / "data" / "profiles"


def main():
    parser = argparse.ArgumentParser(description="Extract stances from member press releases")
    parser.add_argument(
        "--bioguide-id",
        type=str,
        required=True,
        help="Member bioguide ID (e.g., B001316)"
    )
    parser.add_argument(
        "--issues",
        type=str,
        nargs="+",
        default=["government_reform", "budget_economy", "health_care", "immigration"],
        help="Issues to extract stances for"
    )

    args = parser.parse_args()

    # Load member profile
    profile_path = PROFILES_DIR / f"{args.bioguide_id}.json"
    if not profile_path.exists():
        print(f"Error: Member profile not found: {profile_path}")
        return

    with open(profile_path, 'r') as f:
        profile = json.load(f)

    member_name = profile['metadata']['name']
    press_releases = profile['pressReleases']

    print("=" * 80)
    print(f"EXTRACTING STANCES FOR {member_name}")
    print("=" * 80)
    print(f"Member: {member_name} ({args.bioguide_id})")
    print(f"Press Releases: {len(press_releases)}")
    print(f"Issues to test: {', '.join(args.issues)}")
    print()

    # Extract stances for each press release
    all_findings = []

    for i, pr in enumerate(press_releases, 1):
        print(f"\n{'=' * 80}")
        print(f"Press Release {i}/{len(press_releases)}")
        print(f"Title: {pr['title']}")
        print(f"Date: {pr['date']}")
        print("=" * 80)

        full_text = f"{pr['title']}\n\n{pr['bodyText']}"
        pr_stances = {}

        for issue in args.issues:
            print(f"\nExtracting stance on '{issue}'...")

            try:
                stance = extract_stance_from_text(
                    text=full_text,
                    issue_category=issue,
                    source_url=pr['url']
                )

                if stance.status != "no_stance_recorded":
                    print(f"  ✓ Status: {stance.status}")
                    if stance.summary:
                        summary_preview = stance.summary[:100] + "..." if len(stance.summary) > 100 else stance.summary
                        print(f"  Summary: {summary_preview}")
                    pr_stances[issue] = stance.dict()
                else:
                    print(f"  - No clear stance")

            except Exception as e:
                print(f"  ✗ Error: {e}")

        if pr_stances:
            all_findings.append({
                "press_release": {
                    "title": pr['title'],
                    "date": pr['date'],
                    "url": pr['url']
                },
                "stances": pr_stances
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Press releases with stances: {len(all_findings)}/{len(press_releases)}")

    total_stances = sum(len(f['stances']) for f in all_findings)
    print(f"Total stances extracted: {total_stances}")

    # Save to file
    output_file = PROFILES_DIR / f"{args.bioguide_id}_stances.json"
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": profile['metadata'],
            "stances": all_findings
        }, f, indent=2)

    print(f"\n✓ Saved to: {output_file}")


if __name__ == "__main__":
    main()
