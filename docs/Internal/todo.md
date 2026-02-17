# Hippodetector Development Checklist

## High Level Objectives

### RAG System
- [x] Build per-member database (votes + bills + press releases)
- [x] Set up Qdrant vector DB with Docker Compose
- [x] Create embedding generation script
- [x] Load embeddings into Qdrant (369 bills + 10 PRs loaded for Burlison)
- [x] Build automated pipeline script (run_contradiction_pipeline.py + run_pipeline.sh)
  - [x] Steps 1-4: Data collection and embedding loading (can run with `--skip-contradictions`)
  - [ ] Step 5: Contradiction detection (TODO - can be run standalone later)
- [ ] Add natural language query interface to Streamlit

### Scale to All Members
- [ ] Add parallelization to pipeline
- [ ] Implement incremental update logic
- [ ] Generate profiles for all 435 House members

---

## Low Level Implementation

### Data Structure & Schema
- [x] Define per-member database schema and JSON structure

### Pipeline Development
- [x] Find Burlison's bioguide ID
- [x] Build member profile aggregation script
- [x] Implement voting record fetcher for specific bioguide ID (with progress bars)
- [x] Implement bill details fetcher (comprehensive data: summaries & subjects)
- [x] Link existing press release data
- [x] Build end-to-end automated pipeline (run_contradiction_pipeline.py)
  - [x] Step 1: Fetch voting records
  - [x] Step 2: Fetch bill details
  - [x] Step 3: Build member profile
  - [x] Step 4: Load embeddings into Qdrant
  - [x] Skip flags for partial execution (`--skip-voting`, `--skip-bills`, `--skip-embeddings`, `--skip-contradictions`)
  - [x] Pre-computed PR embeddings support (438 House members)
  - [x] Bash wrapper (run_pipeline.sh) with convenience features
  - [ ] Step 5: Contradiction detection (TODO)

### Burlison Test Case
- [x] Fetch Burlison's voting record (432 votes)
- [x] Extract unique bills from Burlison's votes (273 bills)
- [x] Fetch bill details for Burlison's bills (comprehensive data)
- [x] Generate complete Burlison profile JSON (data/members/B001316.json)

### RAG System Setup
- [x] Create docker-compose.yml for Qdrant
- [x] Define Qdrant collection schemas (bills and press releases)
- [x] Test Qdrant connection and basic operations
- [x] Build embedding generation script (RAG/load_embeddings.py)
- [x] Load embeddings into Qdrant (369 bill-vote pairs + 10 PRs using CPU batching)
- [x] Implement semantic search function (RAG/search.py)
  - Query embedding generation
  - Qdrant similarity search
  - Return top-k bills and press releases
  - Tested with "federal regulations" query - returns relevant bills + press releases
- [x] Build post-retrieval stance extraction (RAG/extract_stances.py)
  - Use dataset/memberOpinions.py IssueStance schema
  - Extract structured stances from retrieved press releases
  - Map bill subjects to issue categories
  - Dual LLM provider support (Archia/Claude or Gemini)
  - Tested with Gemini - correctly extracts stances with structured output
- [ ] Build contradiction detector (RAG/detect_contradictions.py)
  - Compare extracted stances vs vote positions
  - Match topics: bill subjects ↔ issue categories
  - Generate structured contradiction output with sources
- [ ] Test full RAG pipeline with sample queries on Burlison data
- [ ] Add natural language query interface to Streamlit UI

**Note**: Using post-processing approach (Option 3) - extract stances AFTER retrieval for flexibility and cost-efficiency
