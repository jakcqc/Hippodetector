# Hippodetector
Ever wondered about how hypocritical a local representative is? Now you can find out!

Hippodetector helps you analyze politician voting records from Congress and detect inconsistencies between their votes and public statements.

## Quick Start

```bash
# Get a politician's recent voting record
uv run dataset/voting_record.py --bioguide-id O000172 --congress 119 --max-votes 50
```

## Required Data File Locations

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
