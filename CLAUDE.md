# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hippodetector is a congressional hypocrisy detection system that identifies contradictions between U.S. House members' voting records and their public statements. It uses RAG (Retrieval-Augmented Generation) with semantic search to match topics and compare positions.

**Core Architecture:** Per-member JSON databases (`data/members/{bioguideId}.json`) containing aggregated votes, bill details, and press releases → embedded into Qdrant vector DB → semantic search + LLM-based stance extraction → contradiction detection.

## Essential Commands

### Pipeline Execution (Primary Interface)

```bash
# Complete pipeline for single member (most common usage)
./run_pipeline.sh B001316

# Process 20 sample politicians from sample_politicians.txt
./run_pipeline.sh --sample

# Data collection only (skip Step 5: contradiction detection)
./run_pipeline.sh --sample --skip-contradictions

# Resume from specific step (if earlier steps completed)
./run_pipeline.sh B001316 --skip-voting --skip-bills --skip-profile

# Python interface (for programmatic usage)
uv run python run_contradiction_pipeline.py --bioguide-ids B001316,O000172
uv run python run_contradiction_pipeline.py --from-file politicians.txt
```

### Vector Database (Qdrant)

```bash
# Start Qdrant (required for RAG operations)
docker-compose up -d

# Initialize collections (one-time setup)
uv run python RAG/setup_qdrant.py

# Check status
curl http://localhost:6333/healthz
curl http://localhost:6333/collections
```

### Streamlit UI

```bash
streamlit run server/app.py
# Dashboard: http://localhost:6333/dashboard (Qdrant)
```

### Manual Data Collection (for debugging)

```bash
# Fetch individual components
uv run python dataset/voting_record.py --bioguide-id O000172 --congress 119
uv run python dataset/fetch_bill_details.py --from-votes data/votes/O000172.json
uv run python dataset/build_member_profile.py --bioguide-id O000172

# Load embeddings manually
uv run python RAG/load_embeddings.py --bioguide-id B001316 --use-precomputed-pr
```

## Architecture & Data Flow

### 5-Step Pipeline

The `run_contradiction_pipeline.py` script orchestrates the complete workflow:

1. **Voting Records** (`dataset/voting_record.py`) → `data/votes/{bioguideId}.json`
   - Fetches all votes from Congress.gov API
   - Includes vote position (Yea/Nay/Not Voting), date, rollcall number

2. **Bill Details** (`dataset/fetch_bill_details.py`) → `data/bills_cache/{billId}.json`
   - For each voted bill, fetches comprehensive details (3 API calls per bill)
   - Includes: title, summary (multiple versions), subjects, cosponsors
   - Uses file-based caching to avoid redundant API calls

3. **Member Profile** (`dataset/build_member_profile.py`) → `data/profiles/{bioguideId}.json`
   - Joins votes + bills + press releases into single profile
   - This is the canonical data structure used by all downstream operations
   - Schema: `{metadata, votes: [], bills: [], pressReleases: []}`

4. **Load Embeddings** (`RAG/load_embeddings.py`) → Qdrant collections
   - Generates 768-dim embeddings using google/embeddinggemma-300m
   - **Optimization:** Uses pre-computed PR embeddings from zip files (438 House members)
   - Creates two collections: `bills` (one point per vote) and `press_releases`

5. **Detect Contradictions** (TODO - currently placeholder)
   - Semantic search → stance extraction → topic matching → contradiction scoring
   - Will output structured ContradictionReport (see RAG/contradiction_schema.py)

### Data Storage Patterns

**Per-Member Files (Preferred):**
- `data/profiles/{bioguideId}.json` - Complete profile (canonical source)
- `data/votes/{bioguideId}.json` - Individual voting records
- Scales better than monolithic files for 438 House members

**Caching Strategy:**
- `data/bills_cache/{congress}-{billType}-{billNumber}.json` - Shared bill cache
- Files checked before API calls to avoid redundant requests
- Congress.gov rate limit: 5,000 requests/hour (important for batch processing)

**Pre-computed Embeddings:**
- `data/press_release_embeddings_1.zip` (Members A-L)
- `data/press_release_embeddings_2.zip` (Members M-Z)
- Saves ~15-20 minutes per member + LLM API costs
- Automatically used by pipeline with `--use-precomputed-pr` flag

### RAG Architecture Details

**Post-Retrieval Stance Extraction (Design Choice):**
1. Semantic search retrieves top-k bills and press releases
2. LLM extracts structured stances from retrieved press releases
3. Topic matching maps bill subjects → issue categories (24 categories from memberOpinions.py)
4. Contradiction detection compares vote positions vs stated stances

**Why post-retrieval?** Preserves semantic search flexibility, only processes retrieved items (cost-efficient), enables structured output without upfront categorization of all press releases.

## Directory Structure & Key Files

### Data Collection (`dataset/`)
- `voting_record.py` - Congress.gov API client for votes (with progress bars)
- `fetch_bill_details.py` - Bill details fetcher (comprehensive: summaries + subjects)
- `build_member_profile.py` - Profile aggregator (joins votes + bills + PRs)
- `pressReleaseScraper.py` - House.gov website scraper
- `memberOpinions.py` - Issue stance schema (24 categories: health_care, immigration, etc.)
- `billDataGrabber.py` - Legacy bill fetcher (superseded by fetch_bill_details.py)

### RAG System (`RAG/`)
- `setup_qdrant.py` - Initialize vector DB collections
- `load_embeddings.py` - Generate & load embeddings (supports pre-computed)
- `topic_matching.py` - Bill subjects → issue categories mapping
- `extract_stances.py` - LLM-based stance extraction (supports Archia/Claude + Gemini)
- `extract_member_stances.py` - Batch stance extraction for all PRs
- `contradiction_schema.py` - Output schemas (VoteEvidence, StatementEvidence, Contradiction, ContradictionReport)
- `search.py` - Semantic search (TODO)

### LLM & Embeddings (`LLM/`)
- `hf_embedding_gemma.py` - Embedding generation (google/embeddinggemma-300m)
- `archia_model_types.py` - Model type definitions
- Smart fallback: GPU → HF Inference API → CPU (with batching)

### UI (`server/`)
- `app.py` - Streamlit main app
- `pages/1_Voted_Bills.py` - Streamlit page

### Documentation (`docs/`)
- `architecture.md` - System architecture, design decisions, data flow (READ THIS for deep understanding)
- `Internal/member_profile_schema.json` - JSON schema for member profiles
- `Internal/member_profile_example.json` - Example member profile

## Development Patterns

### Environment Variables

Required in `.env` (see `.env.example`):
```bash
CONGRESS_API_KEY=...        # Congress.gov API access
ARCHIA_API_KEY=...          # LLM provider (optional)
GOOGLE_API_KEY=...          # Gemini API (optional)
HF_TOKEN=...                # HuggingFace API (for embedding fallback)
```

### Package Management (uv)

This project uses **uv** for dependency management (faster than pip):
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv
uv venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
# or
uv sync
```

### Python Version

Requires Python 3.12 (see `.python-version` and `pyproject.toml`)

### API Rate Limits

- **Congress.gov:** 5,000 requests/hour
- **Bill details:** 3 API calls per bill (details, summaries, subjects)
- **Batch processing:** Always use caching to avoid redundant requests
- **Embeddings:** Use `--use-precomputed-pr` flag to avoid regenerating press release embeddings

### House Members Only

The pipeline uses pre-computed embeddings for **438 House members**. Senate members will fail during embedding load. The pipeline validates bioguide IDs against available members in zip files and filters out invalid IDs automatically.

### Test Case

**Eric Burlison (B001316)** is the primary test case:
- 432 votes with positions
- 273 bills with comprehensive details
- 10 press releases
- Full profile: `data/profiles/B001316.json`
- Use this for validation when testing new features

## Important Implementation Notes

### Schema Relationships

```
memberOpinions.py (Input Stance Schema)
├── IssueStance: {status: "supports"|"opposes"|"mixed", summary: str, source_url: str}
└── CandidateIssueProfile: {health_care: IssueStance, immigration: IssueStance, ...}

contradiction_schema.py (Output Schema)
├── VoteEvidence: bill metadata + vote position
├── StatementEvidence: press release + extracted stance
├── Contradiction: {statement, vote, explanation, severity, confidenceScore}
└── ContradictionReport: {query, member metadata, contradictions: []}
```

### Contradiction Severity Levels

- **direct:** Vote directly contradicts stated position (e.g., says "oppose X", votes Yes on X)
- **moderate:** Contradiction in related context (e.g., says "reduce regulations", votes for regulatory bill)
- **weak:** Minor inconsistency or timing difference
- **nuanced:** Complex, debatable, or requires context

### Confidence Scoring Factors

1. Topic match strength (bill subjects ↔ issue category)
2. Clarity of extracted stance (supports vs mixed)
3. Recency (older statements = lower confidence)
4. LLM's own confidence in extraction

## RAG System Workflow

When contradiction detection is implemented, the flow will be:

1. **User Query** → Generate embedding
2. **Semantic Search** → Retrieve top-k bills and press releases from Qdrant
3. **Stance Extraction** → LLM extracts structured stances from retrieved press releases
4. **Topic Matching** → Map bill subjects to issue categories using fuzzy matching
5. **Contradiction Detection** → Compare vote positions vs stated stances
6. **Severity Scoring** → Classify contradiction type and assign confidence score
7. **Output Generation** → Return ContradictionReport with citations

## Common Workflows

### Adding a New House Member

```bash
# Get bioguide ID from https://bioguide.congress.gov
./run_pipeline.sh <BIOGUIDE_ID>

# Or add to sample_politicians.txt and run batch
./run_pipeline.sh --sample
```

### Debugging Pipeline Failures

```bash
# Run individual steps with verbose output
uv run python dataset/voting_record.py --bioguide-id B001316 --max-votes 50
uv run python dataset/fetch_bill_details.py --from-votes data/votes/B001316.json
uv run python dataset/build_member_profile.py --bioguide-id B001316
uv run python RAG/load_embeddings.py --bioguide-id B001316 --use-precomputed-pr

# Check generated files
cat data/profiles/B001316.json | jq '.metadata'
cat data/profiles/B001316.json | jq '.votes | length'
cat data/profiles/B001316.json | jq '.bills | length'
```

### Regenerating Embeddings

```bash
# Delete existing embeddings
rm -rf data/qdrant_storage/

# Reinitialize
docker-compose down && docker-compose up -d
uv run python RAG/setup_qdrant.py

# Load fresh embeddings (without pre-computed PRs)
uv run python RAG/load_embeddings.py --bioguide-id B001316
```

## Future Development Areas

Current status (from docs/Internal/todo.md and AGENTS.md):
- ✅ Data collection pipeline (Steps 1-4)
- ✅ RAG infrastructure (Qdrant, embeddings, pre-computed optimization)
- ✅ Topic matching and stance extraction
- ⏳ Semantic search implementation (RAG/search.py)
- ⏳ Contradiction detection logic (RAG/detect_contradictions.py)
- ⏳ Streamlit UI integration

When implementing new features, always validate with Burlison test case first before scaling to all 435 House members.
