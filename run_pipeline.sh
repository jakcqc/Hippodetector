#!/bin/bash
#
# Wrapper script for running the contradiction detection pipeline
#
# Usage:
#   ./run_pipeline.sh --help                        # Show help
#   ./run_pipeline.sh B001316                       # Run for single politician
#   ./run_pipeline.sh B001316,A000055               # Run for multiple politicians
#   ./run_pipeline.sh --file sample_politicians.txt # Run from file
#   ./run_pipeline.sh --sample                      # Run sample 20 politicians
#
# Options:
#   --skip-all          Skip all data collection steps (voting, bills, profile, embeddings)
#   --skip-voting       Skip fetching voting records
#   --skip-bills        Skip fetching bill details
#   --skip-profile      Skip building member profile
#   --skip-embeddings   Skip loading embeddings
#   --max-votes N       Limit number of votes to fetch (for testing)
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BIOGUIDE_IDS=""
FROM_FILE=""
SKIP_VOTING=""
SKIP_BILLS=""
SKIP_PROFILE=""
SKIP_EMBEDDINGS=""
MAX_VOTES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Contradiction Detection Pipeline Runner"
            echo ""
            echo "Usage:"
            echo "  $0 B001316                           # Single politician"
            echo "  $0 B001316,A000055                   # Multiple politicians"
            echo "  $0 --file sample_politicians.txt     # From file"
            echo "  $0 --sample                          # Run sample 20 politicians"
            echo ""
            echo "Options:"
            echo "  --skip-all          Skip all data collection steps"
            echo "  --skip-voting       Skip fetching voting records"
            echo "  --skip-bills        Skip fetching bill details"
            echo "  --skip-profile      Skip building member profile"
            echo "  --skip-embeddings   Skip loading embeddings"
            echo "  --max-votes N       Limit votes to fetch (testing)"
            echo ""
            echo "Examples:"
            echo "  # Run full pipeline for Burlison"
            echo "  $0 B001316"
            echo ""
            echo "  # Run for existing data (skip data collection)"
            echo "  $0 B001316 --skip-all"
            echo ""
            echo "  # Run sample politicians with existing data"
            echo "  $0 --sample --skip-all"
            exit 0
            ;;
        --sample)
            FROM_FILE="sample_politicians.txt"
            shift
            ;;
        --file|-f)
            FROM_FILE="$2"
            shift 2
            ;;
        --skip-all)
            SKIP_VOTING="--skip-voting"
            SKIP_BILLS="--skip-bills"
            SKIP_PROFILE="--skip-profile"
            SKIP_EMBEDDINGS="--skip-embeddings"
            shift
            ;;
        --skip-voting)
            SKIP_VOTING="--skip-voting"
            shift
            ;;
        --skip-bills)
            SKIP_BILLS="--skip-bills"
            shift
            ;;
        --skip-profile)
            SKIP_PROFILE="--skip-profile"
            shift
            ;;
        --skip-embeddings)
            SKIP_EMBEDDINGS="--skip-embeddings"
            shift
            ;;
        --max-votes)
            MAX_VOTES="--max-votes $2"
            shift 2
            ;;
        *)
            # Assume it's a bioguide ID or comma-separated list
            BIOGUIDE_IDS="$1"
            shift
            ;;
    esac
done

# Validate input
if [ -z "$BIOGUIDE_IDS" ] && [ -z "$FROM_FILE" ]; then
    echo -e "${RED}Error: No bioguide IDs or file specified${NC}"
    echo "Usage: $0 --help"
    exit 1
fi

# Build command
CMD="uv run python run_contradiction_pipeline.py"

if [ -n "$BIOGUIDE_IDS" ]; then
    CMD="$CMD --bioguide-ids $BIOGUIDE_IDS"
elif [ -n "$FROM_FILE" ]; then
    if [ ! -f "$FROM_FILE" ]; then
        echo -e "${RED}Error: File not found: $FROM_FILE${NC}"
        exit 1
    fi
    CMD="$CMD --from-file $FROM_FILE"
fi

# Add skip flags
CMD="$CMD $SKIP_VOTING $SKIP_BILLS $SKIP_PROFILE $SKIP_EMBEDDINGS $MAX_VOTES"

# Show command being run
echo -e "${GREEN}Running:${NC} $CMD"
echo ""

# Run the pipeline
eval $CMD

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Pipeline completed successfully${NC}"
else
    echo ""
    echo -e "${RED}✗ Pipeline failed${NC}"
    exit 1
fi
