# RAG System for Hippodetector

This directory contains the Retrieval-Augmented Generation (RAG) system for detecting contradictions between congressional members' voting records and public statements.

## Architecture

### Components

1. **Vector Database (Qdrant)**
   - Stores embeddings for bills and press releases
   - Enables semantic search across voting records and statements
   - Running in Docker container (port 6333)

2. **Embedding Model (google/embeddinggemma-300m)**
   - Generates 768-dimensional embeddings
   - Uses mean pooling with normalization
   - See `LLM/hf_embedding_gemma.py` for implementation

3. **Collections**
   - **bills**: Bill embeddings with vote context
     - Metadata: billId, title, summary, memberVote, voteDate, bioguideId, subjects
   - **press_releases**: Press release embeddings
     - Metadata: releaseId, title, content, date, bioguideId, url

### Design Decision: Post-Retrieval Stance Extraction

**Approach**: We use **Option 3 (Post-Processing)** for structured opinion extraction.

**Methodology**:
1. Store raw press releases as embeddings (preserves full context)
2. Use RAG to retrieve relevant bills + press releases via semantic search
3. Extract structured stances (using `dataset/memberOpinions.py` schema) ONLY from retrieved items
4. Compare extracted stances vs vote positions on matching topics
5. Return structured contradictions with source citations

**Why This Approach?**
- ✅ **Flexible**: Semantic search finds any contradiction, not limited to predefined categories
- ✅ **Cost-effective**: Only run LLM extraction on retrieved items (~5-10), not all PRs
- ✅ **Preserves context**: Full text available for nuanced analysis
- ✅ **Structured output**: Clear contradiction explanations using IssueStance model
- ✅ **Iterative**: Can improve extraction without rebuilding embeddings

**Alternative Approaches Considered**:
- **Option 1 (Raw only)**: No structured extraction - rejected due to vague outputs
- **Option 2 (Preprocess all)**: Extract stances before embedding - rejected due to upfront cost and lost flexibility

## Setup

### 1. Start Qdrant

```bash
# From project root
docker-compose up -d
```

### 2. Install Dependencies

```bash
# Install qdrant-client (already in pyproject.toml)
uv sync
```

### 3. Initialize Collections

```bash
python RAG/setup_qdrant.py
```

This will:
- Connect to Qdrant
- Create the `bills` and `press_releases` collections
- Verify the setup

### 4. Generate and Load Embeddings

```bash
# Generate embeddings for Burlison's data and load into Qdrant
uv run RAG/load_embeddings.py --bioguide-id B001316
```

This will:
- Load `data/members/B001316.json`
- Join votes with bills (to get vote positions)
- Generate embeddings using google/embeddinggemma-300m
- Load 273 bill embeddings into Qdrant
- Load 10 press release embeddings into Qdrant

## Workflow

### Data Flow

```
Member Profile (JSON)
    ↓
Extract bills + press releases
    ↓
Generate embeddings (embeddinggemma-300m)
    ↓
Load into Qdrant with metadata
    ↓
Semantic search enabled
```

### Query Flow

```
User query (natural language)
    ↓
Generate query embedding
    ↓
Search Qdrant (cosine similarity)
    ↓
Retrieve relevant bills + press releases
    ↓
Extract stances from retrieved PRs (using memberOpinions.py schema)
    ↓
Match topics: bill subjects ↔ issue categories
    ↓
Compare: extracted stance vs vote position
    ↓
LLM reasoning: explain contradiction
    ↓
Return structured contradiction with sources
```

## File Structure

```
RAG/
├── README.md                  # This file
├── setup_qdrant.py           # Initialize Qdrant collections
├── load_embeddings.py        # Generate and load embeddings
├── contradiction_schema.py   # Output schema definitions (Contradiction, ContradictionReport)
├── search.py                 # Semantic search functions (TODO)
├── extract_stances.py        # Post-retrieval stance extraction (TODO)
└── detect_contradictions.py  # Compare stances vs votes (TODO)
```

## Data Schemas

### Input Schema
- **`dataset/memberOpinions.py`**: Defines `IssueStance` and `CandidateIssueProfile`
  - 24 issue categories (healthcare, immigration, etc.)
  - Stance status: supports, opposes, mixed, unknown

### Output Schema
- **`RAG/contradiction_schema.py`**: Defines structured contradiction output
  - `Contradiction`: Single detected contradiction with evidence
  - `VoteEvidence`: Bill details + vote position
  - `StatementEvidence`: Press release excerpt + extracted stance
  - `ContradictionReport`: Full query result with metadata

## Next Steps

1. ✅ Docker Compose setup
2. ✅ Qdrant collection schemas
3. ✅ Connection testing
4. ✅ Embedding generation script
5. ⏳ Load embeddings into Qdrant (ready to test)
6. ⏳ Implement semantic search
7. ⏳ Build LLM contradiction detector
8. ⏳ Add to Streamlit UI

## Testing

```bash
# Check Qdrant is running
curl http://localhost:6333/healthz

# View collections
curl http://localhost:6333/collections

# View Qdrant dashboard
open http://localhost:6333/dashboard
```

## Configuration

- **Qdrant Host**: localhost
- **Qdrant Port**: 6333 (REST), 6334 (gRPC)
- **Embedding Dimension**: 768
- **Distance Metric**: Cosine similarity
- **Storage**: `data/qdrant_storage/` (gitignored)
