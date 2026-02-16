# Agent Instructions for Hippodetector

## Project Overview
Hippodetector detects hypocrisy in U.S. Congress members by comparing their voting records against their public statements (press releases). Uses RAG (Retrieval-Augmented Generation) to semantically match topics and identify contradictions.

## Architecture

### Data Pipeline
- **Per-member databases**: Separate JSON files in `data/members/{bioguideId}.json`
- **Structure**: Each profile contains metadata, voting record, bill details, and press releases
- **Sources**: Congress.gov API (votes/bills), House.gov website scraping (press releases)

### RAG System
- **Vector DB**: Embeddings of bills (with vote direction) and press releases
- **Query Pattern**: Natural language → retrieve relevant votes/statements → extract structured stances → compare stances vs votes → LLM explains contradictions
- **Stance Extraction**: Post-retrieval extraction using `dataset/memberOpinions.py` schema (24 issue categories)
- **Output**: Structured contradictions with source citations (IssueStance model)

### Tech Stack
- Python 3.12 (uv package manager)
- Streamlit (UI)
- Congress.gov API
- Qdrant (vector DB via Docker)
- Embedding model (see `LLM/` directory)

## Key Files

### Data Collection
- `dataset/voting_record.py` - Fetch member voting records (with progress bars)
- `dataset/fetch_bill_details.py` - Fetch comprehensive bill details (summaries & subjects)
- `dataset/build_member_profile.py` - Aggregate votes, bills, and press releases into per-member profiles
- `dataset/pressReleaseScraper.py` - Scrape press releases from House.gov
- `dataset/fetch_congress_members.py` - Get member metadata
- `dataset/billDataGrabber.py` - Legacy bill fetcher (superseded by fetch_bill_details.py)

### Analysis
- `LLM/archia_hello_world.py` - Model testing
- `LLM/hf_embedding_gemma.py` - Embedding generation
- `LLM/archia_model_types.py` - Model type definitions

### Interface
- `server/app.py` - Streamlit app (currently shows press releases only)

### Data
- `data/congress_members.json` - All member metadata
- `data/congress_bills_voted_last_5_years.json` - All bills with votes (421K bills, 20K with votes)
- `data/votes_{bioguideId}.json` - Individual voting records
- `data/bills_cache/` - Cached bill details (comprehensive: summaries & subjects)
- `data/members/{bioguideId}.json` - Complete member profiles (votes + bills + press releases)
- `data/burlison_press_releases.json` - Press release data (Burlison test case)

### Documentation
- **`docs/architecture.md`** - System architecture, design decisions, data flow
- `docs/Internal/member_profile_schema.json` - JSON schema definition for per-member databases
- `docs/Internal/member_profile_example.json` - Example member profile with sample data
- `docs/Internal/todo.md` - Current development checklist
- `RAG/README.md` - RAG component setup and usage

## Current State
- ✅ Press release scraper working
- ✅ Bill metadata collected (5-year window)
- ✅ Voting record fetcher with progress bars
- ✅ Bill details fetcher with comprehensive data (summaries & subjects, 3 API calls per bill)
- ✅ Member profile aggregation script (combines votes + bills + press releases)
- ✅ Streamlit UI for browsing press releases
- ✅ Per-member database schema defined
- ✅ **Burlison test case complete** (432 votes, 273 bills, 10 press releases)
- ✅ **RAG infrastructure complete**:
  - Docker Compose setup for Qdrant
  - Embedding generation with smart fallback (GPU → HF API → CPU)
  - 369 bill-vote pairs + 10 press releases loaded into Qdrant
- ⏳ RAG pipeline (next: semantic search, stance extraction, contradiction detection)

## Test Case
**Eric Burlison (B001316)** - Complete test case for the data collection pipeline
- ✅ 432 votes with positions
- ✅ 273 bills with comprehensive details (titles, summaries, subjects)
- ✅ 10 press releases
- ✅ Full profile saved to `data/members/B001316.json`
- Next: RAG analysis with this data

## Development Guidelines
1. **Data collection**: Always cache API responses to avoid redundant calls
2. **API rate limits**: Congress.gov API has 5,000 requests/hour limit (bill details need 3 calls each)
3. **File structure**: Use separate per-member files (`data/members/{bioguideId}.json`) not one monolithic file
4. **Scalability**: Design for all 435 House members, but validate with Burlison first
5. **RAG first**: The goal is contradiction detection via retrieval + LLM reasoning, not just keyword matching

## Next Agent Tasks
See `docs/Internal/todo.md` for current checklist (organized by High Level / Low Level)
