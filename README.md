# Hippodetector
Ever wondered about how hypocritical a local representative is? Now you can find out!

Hippodetector helps you analyze politician voting records from Congress and detect inconsistencies between their votes and public statements.

## Quick Start

### Run Complete Pipeline

The easiest way to analyze politicians is using the automated pipeline:

```bash
# Run for a single politician (e.g., Eric Burlison)
./run_pipeline.sh B001316

# Run for 20 sample politicians
./run_pipeline.sh --sample

# Run from a custom file
./run_pipeline.sh --file my_politicians.txt

# Run data collection only (skip contradiction detection)
./run_pipeline.sh --sample --skip-contradictions

# See all options
./run_pipeline.sh --help
```

The pipeline automatically:
1. Fetches voting records from Congress API
2. Fetches bill details
3. Builds member profile (votes + bills + press releases)
4. Loads embeddings into Qdrant vector database
5. Detects contradictions (coming soon - can be skipped with `--skip-contradictions`)

### Manual Steps

```bash
# Get a politician's recent voting record
uv run python dataset/voting_record.py --bioguide-id O000172 --congress 119 --max-votes 50

# Or use the Python script directly
uv run python run_contradiction_pipeline.py --bioguide-ids B001316

# Run data collection only (skip contradiction detection)
uv run python run_contradiction_pipeline.py --bioguide-ids B001316 --skip-contradictions
```

## Data Files

### Pre-computed Embeddings (Included)

The project includes pre-computed press release embeddings for **438 House members**:
- `data/press_release_embeddings_1.zip` - Members A-L
- `data/press_release_embeddings_2.zip` - Members M-Z
- Total: ~533MB compressed
- Using these saves time and LLM API costs

The pipeline automatically uses these pre-computed embeddings when available.

### Sample Politicians File

Use `sample_politicians.txt` as a starting point - contains 20 diverse House members:
```bash
./run_pipeline.sh --sample
```

### Generated Data

The pipeline generates:
- `data/votes/{BIOGUIDE_ID}.json` - Voting records for each member
- `data/profiles/{BIOGUIDE_ID}.json` - Complete member profile (votes, bills, press releases)
- Example: `data/profiles/B001316.json` for Eric Burlison

### Streamlit Data Requirements

The Streamlit app expects data files in `data/` with these exact names:

- Press releases input:
  - `data/press_releases_by_bioguide.json`
  - Expected shape: top-level `membersByBioguideId` map (same format used by `server/app.py`)

- Voted bills compact input:
  - `data/congress_bills_voted_compact_last_1_year.json`
  - Used by `server/pages/1_Voted_Bills.py`

If your team stores data elsewhere, copy/sync the exported files into those exact paths before starting Streamlit.

### Build the compact voted-bills file

If you already have:
- `data/congress_bill_summaries_last_1_years.json`

generate the compact file with:

```bash
python dataset/build_voted_bills_compact.py
```

This writes:
- `data/congress_bills_voted_compact_last_1_year.json`

### Run Streamlit

```bash
streamlit run server/app.py
```

## Documentation

Documentation:
- [System Architecture](docs/architecture.md) - How Hippodetector works
- [RAG System](RAG/README.md) - Vector database and semantic search setup

---

# Project Setup (uv)

This project uses **uv** for fast Python dependency management.

## 1. Install uv

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Or:

```bash
pip install uv
```

---

## 2. Create Virtual Environment

From the project root:

```bash
uv venv
```

This creates a `.venv/` directory.

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies or sync

```bash
uv pip install -r requirements.txt
```

```bash
uv sync
```

---
