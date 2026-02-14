# Hippodetector Documentation

Welcome to the Hippodetector documentation! This project helps you analyze politician voting records to detect hypocrisy by comparing their votes with their public statements.

## Documentation Index

### API Documentation
- **[Voting Record API](voting_record_api.md)** - Complete guide to fetching and using voting records

### Getting Started
- [Quick Start Guide](#quick-start)
- [Installation](#installation)
- [Basic Usage](#basic-usage)

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

### 3. Fetch Voting Records

```bash
# Get recent votes for a politician
uv run dataset/voting_record.py --bioguide-id O000172 --congress 119 --max-votes 50
```

## Installation

### System Requirements

- **Python**: 3.12 or higher
- **Package Manager**: uv
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

## Project Structure

```
Hippodetector/
├── dataset/
│   ├── fetch_congress_members.py          # Fetch Congress member list (prerequisite)
│   └── voting_record.py        # Fetch member voting records (main script)
├── data/
│   ├── .gitkeep               # Preserves directory in git
│   ├── congress_members.json   # List of all Congress members (generated)
│   └── votes_*.json           # Voting records by member (generated)
├── LLM/
│   └── archia_*.py            # LLM integration
├── docs/
│   ├── README.md              # This file
│   ├── voting_record_api.md   # Complete API reference
│   ├── examples.md            # Practical examples
│   └── CHEATSHEET.md          # Quick reference
├── .env.example               # Environment variables template
├── .gitignore                 # Ignores data/*.json and data/*.csv
├── pyproject.toml            # Project dependencies
└── README.md                 # Main project README
```

**Note:** Files marked as "(generated)" are created by running the scripts and are not committed to git.

## Common Bioguide IDs

Here are some commonly searched politicians:

### House Members (Current)
- Alexandria Ocasio-Cortez: `O000172`
- Nancy Pelosi: `P000197`
- Kevin McCarthy: `M001165`
- Hakeem Jeffries: `J000294`
- Marjorie Taylor Greene: `G000596`
- Matt Gaetz: `G000578`

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

## Next Steps

1. **[Read the Voting Record API Documentation](voting_record_api.md)** for detailed usage
2. **Explore the Data**: Check the JSON files in `data/` directory
3. **Build Analysis Tools**: Use the voting data to detect hypocrisy
4. **Integrate with LLMs**: Use the LLM tools to analyze voting patterns

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
