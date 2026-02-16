# Hippodetector Documentation

Welcome to the Hippodetector documentation! This project detects hypocrisy in U.S. Congress members by comparing their voting records against their public statements (press releases) using RAG (Retrieval-Augmented Generation).

## Documentation Index

### Architecture & Design
- **[Project Overview](../AGENTS.md)** - High-level architecture and design decisions
- **[Member Profile Schema](Internal/member_profile_schema.json)** - JSON schema for per-member databases
- **[Schema Example](Internal/member_profile_example.json)** - Sample member profile with data
- **[Development Checklist](Internal/todo.md)** - Current development roadmap

### API Documentation
- **[Voting Record API](voting_record_api.md)** - Complete guide to fetching and using voting records

### Getting Started
- [Quick Start Guide](#quick-start)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Running the Streamlit UI](#streamlit-ui)

## How It Works

Hippodetector uses a **RAG (Retrieval-Augmented Generation)** system to detect contradictions between what Congress members say and how they vote:

1. **Data Collection**: Fetch voting records, bill details, and press releases for each member
2. **Per-Member Databases**: Store complete profiles in `data/members/{bioguideId}.json`
3. **Vector Embeddings**: Generate embeddings for bills (with vote direction) and press releases
4. **Semantic Retrieval**: Query the vector database using natural language (e.g., "What climate bills did they vote against?")
5. **LLM Reasoning**: Compare retrieved votes vs. statements to identify contradictions
6. **Contextual Output**: Generate explanations like "Voted Nay on HR 2891 (climate bill) but issued 3 press releases championing environmental leadership"

### Current Development Phase

We're using **Eric Burlison** as a test case to validate the full pipeline before scaling to all 435 House members.

## Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to project
cd Hippodetector

# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Configure API keys
cp .env.example .env
# Edit .env and add your Congress API key
```

### 2. Get Congress API Key

1. Sign up at https://api.congress.gov/sign-up/
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
3. Add your API key to `.env`:
   ```bash
   CONGRESS_API=your_api_key_here
   ```

### 3. Start Qdrant (Vector Database)

```bash
# Start Qdrant vector database
docker compose up -d
```

### 4. Run the Streamlit UI

```bash
# Launch the press release viewer
uv run streamlit run server/app.py
```

The UI will open at `http://localhost:8501` showing press releases by member.

### 5. Fetch Data (Optional)

```bash
# Step 1: Fetch voting records
uv run dataset/voting_record.py --bioguide-id B001316 --congress 119 --max-votes 500

# Step 2: Scrape press releases
uv run dataset/pressReleaseScraper.py --bioguide-ids B001316

# Step 3: Fetch bill details for the votes
uv run dataset/fetch_bill_details.py --from-votes data/votes_B001316.json

# Step 4: Build complete member profile
uv run dataset/build_member_profile.py --bioguide-id B001316
```

## Installation

### System Requirements

- **Python**: 3.12 or higher
- **Package Manager**: uv
- **Docker**: For running Qdrant vector database
- **Operating System**: Linux, macOS, or Windows (WSL recommended)

### Install uv

#### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows (PowerShell)
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Or via pip:
```bash
pip install uv
```

### Project Setup

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync
```

## Basic Usage

### Fetching Voting Records

```bash
# Basic usage with Bioguide ID
uv run dataset/voting_record.py --bioguide-id <BIOGUIDE_ID> --congress 119 --max-votes 50

# Example: Get AOC's voting record
uv run dataset/voting_record.py --bioguide-id O000172 --congress 119 --max-votes 50
```

### Output

The script creates a JSON file in `data/votes_<bioguide_id>.json` with:
- Member information (name, party, state)
- Voting summary statistics
- Detailed list of all votes with:
  - Bill information
  - Vote position (Yea/Nay/etc.)
  - Vote result
  - Date and roll call number

### Example Output

```json
{
  "fetchedAtUtc": "2026-02-14T19:18:33.799864+00:00",
  "member": {
    "bioguideId": "O000172",
    "name": "Alexandria Ocasio-Cortez",
    "party": "Democratic",
    "state": "New York",
    "chamber": "House of Representatives"
  },
  "congress": 119,
  "voteSummary": {
    "Yea": 18,
    "Nay": 20,
    "Not Voting": 1
  },
  "totalVotes": 50,
  "votes": [...]
}
```

**Note:** Data files (*.json, *.csv) in the `data/` directory are ignored by git and must be generated locally by running the scripts.

## Streamlit UI

Launch the web interface to browse press releases:

```bash
uv run streamlit run server/app.py
```

### Current Features
- Browse press releases by member
- Filter by text/title search
- View member profiles (name, state, party, status)
- Formatted HTML display of press release content
- Dark/light theme with custom "Where are the Hippos?" branding

### Upcoming Features
- Member voting record visualization
- Bills voted on with summaries
- Contradiction detection and highlighting
- Natural language query interface (RAG-powered)
- Side-by-side comparison: votes vs. rhetoric

## Project Structure

```
Hippodetector/
├── dataset/                    # Data collection scripts
│   ├── fetch_congress_members.py    # Fetch member metadata
│   ├── voting_record.py             # Fetch voting records
│   ├── billDataGrabber.py           # Fetch bill details from Congress.gov (legacy)
│   ├── fetch_bill_details.py        # Fetch full bill details for specific bills
│   ├── pressReleaseScraper.py       # Scrape press releases from House.gov
│   └── build_member_profile.py      # Aggregate all data into member profile
├── data/                       # Generated data (gitignored)
│   ├── congress_members.json        # All member metadata
│   ├── congress_bills_voted_last_5_years.json  # Bill metadata (421K bills)
│   ├── votes_*.json                 # Individual voting records
│   ├── *_press_releases.json        # Press release data
│   └── members/                     # Per-member databases (future)
│       └── {bioguideId}.json        # Complete member profile
├── LLM/                        # Embedding & model code
│   ├── archia_hello_world.py        # Model testing
│   ├── hf_embedding_gemma.py        # Embedding generation
│   └── archia_model_types.py        # Model type definitions
├── server/                     # Streamlit UI
│   ├── app.py                       # Main Streamlit app
│   └── HippoD.png                   # App logo
├── docs/                       # Documentation
│   ├── README.md                    # This file
│   ├── voting_record_api.md         # API reference
│   ├── examples.md                  # Practical examples
│   ├── CHEATSHEET.md                # Quick reference
│   └── Internal/                    # Internal documentation
│       ├── member_profile_schema.json    # Database schema
│       ├── member_profile_example.json   # Schema example
│       └── todo.md                       # Development checklist
├── AGENTS.md                   # Project overview for AI agents
├── docker-compose.yml          # Qdrant vector database setup
├── .env.example                # Environment variables template
├── pyproject.toml              # Project dependencies (uv)
└── README.md                   # Main project README
```

**Note:** Files in `data/` are generated by running scripts and are gitignored (not committed).

## Common Bioguide IDs

Here are some commonly searched politicians:

### Test Case
- **Eric Burlison** (MO-7): `B001316` ← *Primary test case for RAG system*

### House Members (Current)
- Alexandria Ocasio-Cortez (NY-14): `O000172`
- Nancy Pelosi (CA-11): `P000197`
- Kevin McCarthy (CA-20): `M001165`
- Hakeem Jeffries (NY-08): `J000294`
- Marjorie Taylor Greene (GA-14): `G000596`
- Matt Gaetz (FL-01): `G000578`

### Finding More IDs

1. **Generate the member list**:
   ```bash
   uv run dataset/fetch_congress_members.py
   ```

2. **Search the list**:
   ```bash
   cat data/congress_members.json | grep -i "member_name" -A 3
   ```

3. **Online resources**:
   - https://bioguide.congress.gov/
   - https://www.congress.gov/members

## Development Roadmap

### Current Phase: Building the RAG System
1. ✅ Define per-member database schema
2. ⏳ Build data aggregation pipeline
3. ⏳ Generate Burlison test case profile
4. ⏳ Implement vector DB and embeddings
5. ⏳ Build RAG query interface
6. ⏳ Add contradiction detection to Streamlit

See [docs/Internal/todo.md](Internal/todo.md) for the complete checklist.

## Next Steps for Users

1. **[Read the Member Profile Schema](Internal/member_profile_schema.json)** to understand the data structure
2. **[Review the Project Overview](../AGENTS.md)** for architecture details
3. **Launch the Streamlit UI** to browse existing press release data
4. **Explore the Data**: Check the JSON files in `data/` directory
5. **Follow Development**: Check [todo.md](Internal/todo.md) for progress updates

## Resources

- **Congress.gov API**: https://api.congress.gov/
- **API Documentation**: https://github.com/LibraryOfCongress/api.congress.gov
- **Bioguide Directory**: https://bioguide.congress.gov/
- **uv Documentation**: https://docs.astral.sh/uv/

## Contributing

To contribute to Hippodetector:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

See LICENSE file for details.
