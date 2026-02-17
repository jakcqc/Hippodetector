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

### 3. Run Complete Pipeline (Recommended)

The easiest way to set up everything:

```bash
# Run for a single politician
./run_pipeline.sh B001316

# Or run 20 sample politicians
./run_pipeline.sh --sample

# Run data collection only (skip contradiction detection)
./run_pipeline.sh --sample --skip-contradictions
```

This automatically:
1. Fetches voting records
2. Fetches bill details
3. Builds member profile
4. Loads embeddings into Qdrant (using pre-computed PR embeddings when available)
5. Detects contradictions (coming soon - can be skipped with `--skip-contradictions`)

### 3. Manual Data Loading (Alternative)

```bash
# Load with pre-computed embeddings (fast, no LLM cost)
uv run python RAG/load_embeddings.py --bioguide-id B001316 --use-precomputed-pr

# Or generate fresh embeddings (slower, uses LLM)
uv run python RAG/load_embeddings.py --bioguide-id B001316
```

**Pre-computed Embeddings:**
- Available for 438 House members in `data/press_release_embeddings_*.zip`
- Saves ~15-20 minutes per member
- No LLM API costs
- Automatically used by pipeline script

**Fresh Embedding Generation:**
- Loads member profile from `data/members/{bioguide_id}.json`
- Generates embeddings using google/embeddinggemma-300m
- Smart fallback: GPU → HF Inference API → CPU (with batching)
- CPU mode: ~15-20 minutes but works on limited hardware

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
├── load_embeddings.py        # Generate & load embeddings (supports pre-computed)
├── topic_matching.py         # Map bill subjects to issue categories
├── extract_stances.py        # LLM-based stance extraction (dual provider support)
├── extract_member_stances.py # Batch stance extraction for all PRs
├── contradiction_schema.py   # Output schemas
├── search.py                 # Semantic search (TODO)
└── detect_contradictions.py  # Contradiction detection (TODO)
```

**Pipeline Scripts:**
```
run_contradiction_pipeline.py  # End-to-end Python pipeline
run_pipeline.sh                # Bash wrapper with convenience features
sample_politicians.txt         # 20 diverse House members for testing
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
2. ✅ Pre-computed embeddings optimization (438 House members)
3. ✅ Automated pipeline with progress tracking
4. ✅ Topic matching (bill subjects → issue categories)
5. ✅ Stance extraction with LLM (Archia/Claude + Gemini support)
6. ⏳ Implement semantic search (RAG/search.py)
7. ⏳ Build contradiction detector (RAG/detect_contradictions.py)
8. ⏳ Add to Streamlit UI

## Additional Features

- **House Member Validation**: Pipeline automatically validates bioguide IDs against 438 available members
- **Progress Bar**: Visual progress tracking for batch processing
- **Skip Flags**: Resume pipeline from any step (`--skip-voting`, `--skip-bills`, `--skip-embeddings`, `--skip-contradictions`)
- **Sample Data**: `sample_politicians.txt` with 20 diverse representatives for testing
