"""
Extract stances from all press releases for one or more members.

Usage:
    # Single member
    uv run python RAG/extract_member_stances.py --bioguide-id B001316

    # Multiple members from TOML file
    uv run python RAG/extract_member_stances.py --config members.toml

    # See extract_stances.example.toml for configuration format
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tomllib  # Python 3.11+ built-in (project requires >= 3.12)

from RAG.extract_stances import extract_stance_from_text

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STANCES_OUTPUT_DIR = DATA_DIR / "stances"
PRESS_RELEASES_FILE = DATA_DIR / "press_releases_by_bioguide.json"
CONGRESS_MEMBERS_FILE = DATA_DIR / "congress_members.json"


def load_press_releases_data() -> Dict[str, Any]:
    """Load all press releases from consolidated JSON file."""
    if not PRESS_RELEASES_FILE.exists():
        raise FileNotFoundError(f"Press releases file not found: {PRESS_RELEASES_FILE}")

    with open(PRESS_RELEASES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("membersByBioguideId", {})


def load_member_name(bioguide_id: str) -> str:
    """Load member name from congress_members.json."""
    if not CONGRESS_MEMBERS_FILE.exists():
        return f"Member {bioguide_id}"

    with open(CONGRESS_MEMBERS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    members = data.get("members", [])
    for member in members:
        if member.get("bioguideId") == bioguide_id:
            return member.get("name", f"Member {bioguide_id}")

    return f"Member {bioguide_id}"


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    return config


def extract_stances_for_member(
    bioguide_id: str,
    press_releases_data: Dict[str, Any],
    issues: List[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """Extract stances from all press releases for a single member."""

    # Get member's press releases
    member_data = press_releases_data.get(bioguide_id)
    if not member_data:
        if verbose:
            print(f"Warning: No press releases found for {bioguide_id}")
        return None

    press_releases = member_data.get("pressReleases", [])
    if not press_releases:
        if verbose:
            print(f"Warning: Member {bioguide_id} has no press releases")
        return None

    member_name = load_member_name(bioguide_id)

    if verbose:
        print("=" * 80)
        print(f"EXTRACTING STANCES FOR {member_name}")
        print("=" * 80)
        print(f"Member: {member_name} ({bioguide_id})")
        print(f"Press Releases: {len(press_releases)}")
        print(f"Issues to extract: {', '.join(issues)}")
        print()

    # Extract stances for each press release
    all_findings = []

    for i, pr in enumerate(press_releases, 1):
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Press Release {i}/{len(press_releases)}")
            print(f"Title: {pr.get('title', 'Untitled')}")
            print(f"Date: {pr.get('date', 'Unknown')}")
            print("=" * 80)

        full_text = f"{pr.get('title', '')}\n\n{pr.get('bodyText', '')}"
        pr_stances = {}

        for issue in issues:
            if verbose:
                print(f"\nExtracting stance on '{issue}'...")

            try:
                stance = extract_stance_from_text(
                    text=full_text,
                    issue_category=issue,
                    source_url=pr.get('url', '')
                )

                if stance.status != "no_stance_recorded":
                    if verbose:
                        print(f"  ✓ Status: {stance.status}")
                        if stance.summary:
                            summary_preview = stance.summary[:100] + "..." if len(stance.summary) > 100 else stance.summary
                            print(f"  Summary: {summary_preview}")
                    pr_stances[issue] = stance.dict()
                else:
                    if verbose:
                        print(f"  - No clear stance")

            except Exception as e:
                if verbose:
                    print(f"  ✗ Error: {e}")

        if pr_stances:
            all_findings.append({
                "press_release": {
                    "title": pr.get('title', ''),
                    "date": pr.get('date', ''),
                    "url": pr.get('url', '')
                },
                "stances": pr_stances
            })

    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Press releases with stances: {len(all_findings)}/{len(press_releases)}")

        total_stances = sum(len(f['stances']) for f in all_findings)
        print(f"Total stances extracted: {total_stances}")

    return {
        "bioguideId": bioguide_id,
        "name": member_name,
        "stances": all_findings
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract stances from member press releases",
        epilog="See extract_stances.example.toml for batch processing configuration"
    )

    # Input mode: single member or batch from config
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--bioguide-id",
        type=str,
        help="Single member bioguide ID (e.g., B001316)"
    )
    input_group.add_argument(
        "--config",
        type=Path,
        help="TOML configuration file for batch processing"
    )

    # Optional overrides
    parser.add_argument(
        "--issues",
        type=str,
        nargs="+",
        help="Issues to extract (overrides config file)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (overrides config file)"
    )

    args = parser.parse_args()

    # Load press releases data once
    print("Loading press releases data...")
    press_releases_data = load_press_releases_data()
    print(f"Loaded press releases for {len(press_releases_data)} members\n")

    # Determine members and configuration
    if args.bioguide_id:
        # Single member mode
        members = [args.bioguide_id]
        issues = args.issues or ["government_reform", "budget_economy", "health_care", "immigration"]
        output_dir = args.output_dir or STANCES_OUTPUT_DIR
    else:
        # Batch mode from config file
        config = load_config_file(args.config)

        # Parse members list (can be simple array or detailed objects)
        members_config = config.get("members", [])
        if not members_config:
            print("Error: No members specified in config file")
            return

        # Extract bioguide IDs (handle both formats)
        members = []
        for member in members_config:
            if isinstance(member, str):
                members.append(member)
            elif isinstance(member, dict):
                members.append(member.get("bioguide_id"))
            else:
                print(f"Warning: Invalid member format: {member}")

        # Get issues and output directory from config or args
        issues = args.issues or config.get("issues", ["government_reform", "budget_economy", "health_care", "immigration"])
        output_dir = args.output_dir or Path(config.get("output_dir", str(STANCES_OUTPUT_DIR)))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each member
    results = []
    for i, bioguide_id in enumerate(members, 1):
        print(f"\n{'#' * 80}")
        print(f"PROCESSING MEMBER {i}/{len(members)}: {bioguide_id}")
        print(f"{'#' * 80}\n")

        try:
            result = extract_stances_for_member(
                bioguide_id=bioguide_id,
                press_releases_data=press_releases_data,
                issues=issues,
                verbose=True
            )

            if result:
                # Save individual file
                output_file = output_dir / f"{bioguide_id}_stances.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                print(f"\n✓ Saved to: {output_file}")
                results.append(result)
            else:
                print(f"\n✗ No stances extracted for {bioguide_id}")

        except Exception as e:
            print(f"\n✗ Error processing {bioguide_id}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary for batch mode
    if len(members) > 1:
        print("\n" + "=" * 80)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Members processed: {len(results)}/{len(members)}")
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
