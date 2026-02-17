"""
End-to-end contradiction detection pipeline for politicians.

This script runs the complete pipeline:
1. Fetch voting records
2. Fetch bill details
3. Build member profile
4. Load embeddings into Qdrant
5. Extract stances from press releases (using RAG search)
6. Detect contradictions

Usage:
    # Single politician by bioguide ID
    python run_contradiction_pipeline.py --bioguide-ids B001316

    # Multiple politicians
    python run_contradiction_pipeline.py --bioguide-ids B001316,O000172,P000197

    # From file (one bioguide ID per line)
    python run_contradiction_pipeline.py --from-file politicians.txt

    # Skip steps already completed
    python run_contradiction_pipeline.py --bioguide-ids B001316 --skip-voting --skip-bills
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MEMBERS_DIR = DATA_DIR / "members"


class PipelineRunner:
    """Runs the contradiction detection pipeline for politicians."""

    def __init__(
        self,
        bioguide_ids: List[str],
        skip_voting: bool = False,
        skip_bills: bool = False,
        skip_profile: bool = False,
        skip_embeddings: bool = False,
        max_votes: Optional[int] = None,
    ):
        self.bioguide_ids = bioguide_ids
        self.skip_voting = skip_voting
        self.skip_bills = skip_bills
        self.skip_profile = skip_profile
        self.skip_embeddings = skip_embeddings
        self.max_votes = max_votes

    def run(self):
        """Run the pipeline for all politicians."""
        print("=" * 80)
        print("CONTRADICTION DETECTION PIPELINE")
        print("=" * 80)
        print(f"Politicians: {len(self.bioguide_ids)}")
        print(f"  {', '.join(self.bioguide_ids)}")
        print()

        results = []
        for i, bioguide_id in enumerate(self.bioguide_ids, 1):
            print("\n" + "=" * 80)
            print(f"POLITICIAN {i}/{len(self.bioguide_ids)}: {bioguide_id}")
            print("=" * 80)

            result = self.run_for_politician(bioguide_id)
            results.append(result)

            if not result["success"]:
                print(f"\n⚠️  Pipeline failed for {bioguide_id}: {result.get('error')}")

        # Summary
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print(f"✓ Successful: {len(successful)}/{len(results)}")
        if failed:
            print(f"✗ Failed: {len(failed)}/{len(results)}")
            for r in failed:
                print(f"  - {r['bioguide_id']}: {r.get('error', 'Unknown error')}")

        return results

    def run_for_politician(self, bioguide_id: str) -> dict:
        """Run the complete pipeline for one politician."""
        result = {
            "bioguide_id": bioguide_id,
            "success": False,
            "steps_completed": [],
            "error": None,
        }

        try:
            # Step 1: Fetch voting records
            if not self.skip_voting:
                print("\n📊 Step 1: Fetching voting records...")
                if not self._fetch_voting_records(bioguide_id):
                    result["error"] = "Failed to fetch voting records"
                    return result
                result["steps_completed"].append("voting_records")
            else:
                print("\n📊 Step 1: Skipping voting records (already exists)")

            # Step 2: Fetch bill details
            if not self.skip_bills:
                print("\n📄 Step 2: Fetching bill details...")
                if not self._fetch_bill_details(bioguide_id):
                    result["error"] = "Failed to fetch bill details"
                    return result
                result["steps_completed"].append("bill_details")
            else:
                print("\n📄 Step 2: Skipping bill details (already exists)")

            # Step 3: Build member profile
            if not self.skip_profile:
                print("\n👤 Step 3: Building member profile...")
                if not self._build_member_profile(bioguide_id):
                    result["error"] = "Failed to build member profile"
                    return result
                result["steps_completed"].append("member_profile")
            else:
                print("\n👤 Step 3: Skipping member profile (already exists)")

            # Step 4: Load embeddings into Qdrant
            if not self.skip_embeddings:
                print("\n🔢 Step 4: Loading embeddings into Qdrant...")
                if not self._load_embeddings(bioguide_id):
                    result["error"] = "Failed to load embeddings"
                    return result
                result["steps_completed"].append("embeddings")
            else:
                print("\n🔢 Step 4: Skipping embeddings (already loaded)")

            # Step 5: Run contradiction detection
            print("\n🔍 Step 5: Detecting contradictions...")
            contradictions = self._detect_contradictions(bioguide_id)
            result["steps_completed"].append("contradiction_detection")
            result["contradictions"] = contradictions

            result["success"] = True
            print(f"\n✅ Pipeline completed successfully for {bioguide_id}")
            print(f"   Found {len(contradictions)} potential contradictions")

        except Exception as e:
            result["error"] = str(e)
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _fetch_voting_records(self, bioguide_id: str) -> bool:
        """Step 1: Fetch voting records."""
        cmd = [
            "uv", "run", "python", "dataset/voting_record.py",
            "--bioguide-id", bioguide_id
        ]
        if self.max_votes:
            cmd.extend(["--max-votes", str(self.max_votes)])

        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode == 0

    def _fetch_bill_details(self, bioguide_id: str) -> bool:
        """Step 2: Fetch bill details."""
        cmd = [
            "uv", "run", "python", "dataset/fetch_bill_details.py",
            "--bioguide-id", bioguide_id
        ]

        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode == 0

    def _build_member_profile(self, bioguide_id: str) -> bool:
        """Step 3: Build member profile."""
        cmd = [
            "uv", "run", "python", "dataset/build_member_profile.py",
            "--bioguide-id", bioguide_id
        ]

        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode == 0

    def _load_embeddings(self, bioguide_id: str) -> bool:
        """Step 4: Load embeddings into Qdrant."""
        cmd = [
            "uv", "run", "python", "RAG/load_embeddings.py",
            "--bioguide-id", bioguide_id,
            "--use-precomputed-pr"  # Use pre-computed PR embeddings (faster, no LLM cost)
        ]

        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode == 0

    def _detect_contradictions(self, bioguide_id: str) -> list:
        """Step 5: Detect contradictions using RAG pipeline."""
        # TODO: Implement contradiction detection
        # For now, this is a placeholder that runs semantic search and stance extraction

        print("  Running semantic search for relevant bills and press releases...")
        print("  Extracting stances from press releases...")
        print("  Comparing votes vs stances...")
        print("  ⚠️  Note: Full contradiction detection not yet implemented")
        print("  ⚠️  This step currently returns placeholder data")

        # Placeholder return
        return []


def load_bioguide_ids_from_file(file_path: Path) -> List[str]:
    """Load bioguide IDs from a text file (one per line)."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ids = []
    for line in file_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)

    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Run complete contradiction detection pipeline for politicians"
    )

    # Input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--bioguide-ids",
        type=str,
        help="Comma-separated list of bioguide IDs (e.g., B001316,O000172)"
    )
    input_group.add_argument(
        "--from-file",
        type=Path,
        help="Path to file with bioguide IDs (one per line)"
    )

    # Pipeline options
    parser.add_argument(
        "--skip-voting",
        action="store_true",
        help="Skip fetching voting records (if already exists)"
    )
    parser.add_argument(
        "--skip-bills",
        action="store_true",
        help="Skip fetching bill details (if already exists)"
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip building member profile (if already exists)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip loading embeddings (if already loaded)"
    )
    parser.add_argument(
        "--max-votes",
        type=int,
        help="Limit number of votes to fetch (for testing)"
    )

    args = parser.parse_args()

    # Get bioguide IDs
    if args.bioguide_ids:
        bioguide_ids = [bid.strip() for bid in args.bioguide_ids.split(",")]
    else:
        bioguide_ids = load_bioguide_ids_from_file(args.from_file)

    if not bioguide_ids:
        print("Error: No bioguide IDs provided")
        sys.exit(1)

    # Run pipeline
    runner = PipelineRunner(
        bioguide_ids=bioguide_ids,
        skip_voting=args.skip_voting,
        skip_bills=args.skip_bills,
        skip_profile=args.skip_profile,
        skip_embeddings=args.skip_embeddings,
        max_votes=args.max_votes,
    )

    results = runner.run()

    # Exit with error code if any failed
    if any(not r["success"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
