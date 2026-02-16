# Hippodetector Development Checklist

## High Level Objectives

### RAG System
- [x] Build per-member database (votes + bills + press releases)
- [x] Set up Qdrant vector DB with Docker Compose
- [x] Create embedding generation script
- [x] Load embeddings into Qdrant (369 bills + 10 PRs loaded for Burlison)
- [ ] Build RAG pipeline with retrieval + stance extraction + contradiction detection
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
- [ ] Build post-retrieval stance extraction (RAG/extract_stances.py)
  - Use dataset/memberOpinions.py IssueStance schema
  - Extract structured stances from retrieved press releases
  - Map bill subjects to issue categories
- [ ] Build contradiction detector (RAG/detect_contradictions.py)
  - Compare extracted stances vs vote positions
  - Match topics: bill subjects ↔ issue categories
  - Generate structured contradiction output with sources
- [ ] Test full RAG pipeline with sample queries on Burlison data
- [ ] Add natural language query interface to Streamlit UI

**Note**: Using post-processing approach (Option 3) - extract stances AFTER retrieval for flexibility and cost-efficiency
