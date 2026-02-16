# Hippodetector Development Checklist

## High Level Objectives

### RAG System
- [ ] Build per-member database (votes + bills + press releases)
- [ ] Set up Qdrant vector DB with Docker Compose
- [ ] Generate embeddings and load into Qdrant
- [ ] Build RAG pipeline with retrieval + LLM reasoning
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
