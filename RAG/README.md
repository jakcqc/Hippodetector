# RAG System

The RAG (Retrieval-Augmented Generation) component handles semantic search over congressional voting records and public statements.

**For system architecture and design decisions, see [docs/architecture.md](../docs/architecture.md)**

## Quick Start

### 1. Start Qdrant Vector Database

```bash
# From project root
docker-compose up -d
```

### 2. Initialize Collections

```bash
python RAG/setup_qdrant.py
```

Creates `bills` and `press_releases` collections in Qdrant.

### 3. Load Member Data

```bash
# Example: Load Burlison's data
uv run RAG/load_embeddings.py --bioguide-id B001316
```

This will:
- Load member profile from `data/members/B001316.json`
- Generate embeddings using google/embeddinggemma-300m
- Smart fallback: GPU → HF Inference API → CPU (with batching)
- Load embeddings into Qdrant (369 bills + 10 press releases for Burlison)

**Note**: CPU mode takes ~15-20 minutes but works on limited hardware.

## Components

### Vector Database (Qdrant)
- Stores 768-dim embeddings
- Cosine similarity search
- Running on port 6333
- Dashboard: http://localhost:6333/dashboard

### Collections

**bills**
- Metadata: billId, title, summary, memberVote, voteDate, subjects, etc.
- One point per bill-vote pair

**press_releases**
- Metadata: releaseId, title, content, date, url
- One point per press release

### Embedding Model
- Model: google/embeddinggemma-300m
- Dimension: 768
- Implementation: `LLM/hf_embedding_gemma.py`

## Files

```
RAG/
├── setup_qdrant.py           # Initialize collections
├── load_embeddings.py        # Generate & load embeddings
├── contradiction_schema.py   # Output schemas
├── search.py                 # Semantic search (TODO)
├── extract_stances.py        # Stance extraction (TODO)
└── detect_contradictions.py  # Contradiction detection (TODO)
```

## Schemas

See [contradiction_schema.py](contradiction_schema.py) for output types:
- `VoteEvidence` - Bill + vote data
- `StatementEvidence` - Press release + stance
- `Contradiction` - Detected contradiction
- `ContradictionReport` - Full query result

See [dataset/memberOpinions.py](../dataset/memberOpinions.py) for input stance schema.

## Configuration

- **Qdrant Host**: localhost
- **Qdrant Port**: 6333 (REST), 6334 (gRPC)
- **Storage**: `data/qdrant_storage/` (gitignored)

## Testing

```bash
# Check Qdrant status
curl http://localhost:6333/healthz

# View collections
curl http://localhost:6333/collections

# Check point counts
curl http://localhost:6333/collections/bills | grep points_count
curl http://localhost:6333/collections/press_releases | grep points_count
```

## Next Steps

1. ✅ Setup & data loading
2. ⏳ Implement semantic search
3. ⏳ Build contradiction detector
4. ⏳ Add to Streamlit UI
