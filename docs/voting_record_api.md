# Voting Record API Documentation

## Overview

The `voting_record.py` script fetches a politician's voting record from the official Congress.gov API. It retrieves detailed information about each vote a House member has cast, including the bill number, date, their vote position, and the vote result.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Data Structure](#data-structure)
- [Examples](#examples)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Get 50 recent votes for a House member
uv run dataset/voting_record.py --bioguide-id P000197 --congress 119 --max-votes 50
```

## Installation

### Prerequisites

- Python 3.12+
- uv package manager
- Congress.gov API key

### Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Get Congress API Key**:
   - Sign up at: https://api.congress.gov/sign-up/
   - Add your key to `.env` file:
     ```
     CONGRESS_API=your_api_key_here
     ```

## Usage

### Basic Commands

#### Get votes by Bioguide ID (Recommended)

```bash
uv run dataset/voting_record.py --bioguide-id O000172 --congress 119 --max-votes 50
```

#### Get votes by member name (slower)

```bash
uv run dataset/voting_record.py --name "Nancy Pelosi" --congress 119 --max-votes 50
```

#### Get all votes for a congress (very slow)

```bash
uv run dataset/voting_record.py --bioguide-id P000197 --congress 119
```

### Command-Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--bioguide-id` | string | Yes* | Member's Bioguide ID (e.g., `P000197`) |
| `--name` | string | Yes* | Member's name (searches for match) |
| `--congress` | integer | No | Congress number (e.g., `119`). Defaults to member's current congress |
| `--chamber` | string | No | `house` or `senate` (default: `house`). Only House supported currently |
| `--max-votes` | integer | No | Maximum number of recent votes to fetch. Recommended: 50-100 for speed |
| `--output` | string | No | Custom output path (default: `data/votes_<bioguide_id>.json`) |

\* Either `--bioguide-id` or `--name` is required

### Finding Bioguide IDs

You can find Bioguide IDs in several ways:

1. **Use the member list**:
   ```bash
   cat data/congress_members.json | grep -A 2 "Pelosi"
   ```

2. **Search on Congress.gov**:
   - Visit https://www.congress.gov/members
   - Find the member and look at the URL

3. **Common Examples**:
   - Nancy Pelosi: `P000197`
   - Alexandria Ocasio-Cortez: `O000172`
   - Kevin McCarthy: `M001165`
   - Hakeem Jeffries: `J000294`

## API Reference

### Congress.gov API Endpoints Used

1. **Member Info**: `/v3/member/{bioguideId}`
   - Gets basic member information

2. **House Roll Call Votes**: `/v3/house-vote/{congress}`
   - Lists all House roll call votes for a congress

3. **Member Votes**: `/v3/house-vote/{congress}/{session}/{rollCall}/members`
   - Gets how each member voted on a specific roll call

### Rate Limits

- **5,000 requests per hour** (Congress.gov API)
- Script automatically paginates through results
- Fetching 50 votes typically makes ~52 API calls

## Data Structure

### Output JSON Format

```json
{
  "fetchedAtUtc": "2026-02-14T19:18:33.799864+00:00",
  "member": {
    "bioguideId": "O000172",
    "name": "Alexandria Ocasio-Cortez",
    "state": "New York",
    "party": "Democratic",
    "chamber": "House of Representatives"
  },
  "congress": 119,
  "voteSummary": {
    "Yea": 18,
    "Nay": 20,
    "Aye": 2,
    "No": 8,
    "Present": 0,
    "Not Voting": 1,
    "Other": 1
  },
  "totalVotes": 50,
  "votes": [
    {
      "congress": 119,
      "session": 1,
      "rollCall": 240,
      "date": "2025-09-08T18:56:00-04:00",
      "legislation": {
        "type": "HR",
        "number": "3424",
        "url": "https://www.congress.gov/bill/119/house-bill/3424"
      },
      "question": "On Passage",
      "result": "Passed",
      "voteType": "2/3 Yea-And-Nay",
      "memberVote": "Yea",
      "party": "D",
      "state": "NY"
    }
  ]
}
```

### Vote Types

| Value | Description |
|-------|-------------|
| `Yea` | Voted in favor |
| `Nay` | Voted against |
| `Aye` | Voted in favor (voice vote) |
| `No` | Voted against (voice vote) |
| `Present` | Present but abstained |
| `Not Voting` | Did not vote |
| `Other` | Special cases (e.g., Speaker election) |

### Legislation Types

| Type | Description |
|------|-------------|
| `HR` | House Bill |
| `HRES` | House Resolution |
| `HJRES` | House Joint Resolution |
| `HCONRES` | House Concurrent Resolution |
| `SJRES` | Senate Joint Resolution |
| `HAMDT` | House Amendment |

## Examples

### Example 1: Recent Voting Record

```bash
# Get AOC's 50 most recent votes
uv run dataset/voting_record.py --bioguide-id O000172 --congress 119 --max-votes 50
```

**Output:**
```
Fetching voting record for O000172...
Filtering by Congress: 119
  Fetching House votes for Congress 119...
    Fetched 432 votes so far...
  Limited to checking the 50 most recent roll call votes

  Checking 50 roll call votes for member O000172...
    Processed 25/50 votes...
    Processed 50/50 votes...

============================================================
Member: Alexandria Ocasio-Cortez
Party: Democratic
State: New York
============================================================

Voting Summary:
  Total Votes: 50
  Yea: 18 (36.0%)
  Nay: 20 (40.0%)
  ...

Detailed voting record saved to: data/votes_O000172.json
```

### Example 2: Full Congress Voting Record

```bash
# Get all votes for the 119th Congress (slower)
uv run dataset/voting_record.py --bioguide-id P000197 --congress 119
```

**Note:** This will fetch ALL roll call votes (~400+) and can take 5-10 minutes.

### Example 3: Custom Output Path

```bash
# Save to a specific location
uv run dataset/voting_record.py --bioguide-id O000172 --congress 119 --max-votes 50 --output analysis/aoc_votes.json
```

## Limitations

### Current Limitations

1. **House Only**: Senate votes are not yet available in the Congress.gov API (beta)
2. **Recent Congresses**: Data available from 118th Congress (2023) onwards
3. **Performance**: Fetching all votes for a member requires checking each roll call individually
4. **Name Search**: Searching by name may be unreliable; Bioguide ID is recommended

### Recommended Practices

- Use `--max-votes` to limit API calls and improve speed
- Use Bioguide IDs instead of names when possible
- Start with 50-100 votes for testing, then scale up if needed
- Cache results to avoid re-fetching the same data

## Troubleshooting

### Error: Missing Congress API key

```
RuntimeError: Missing Congress API key
```

**Solution:** Add your API key to `.env`:
```bash
echo "CONGRESS_API=your_key_here" >> .env
```

### Error: Could not find member with name

```
Error: Could not find member with name 'John Doe'
```

**Solution:** Use the member's Bioguide ID instead, or check the exact name format in `data/congress_members.json`

### Slow Performance

**Issue:** Script is taking too long to fetch votes.

**Solution:** Use `--max-votes` to limit the number of votes:
```bash
uv run dataset/voting_record.py --bioguide-id P000197 --max-votes 50
```

### HTTP Error 429: Rate Limited

**Issue:** Too many API requests.

**Solution:**
- Wait a few minutes before retrying
- Reduce `--max-votes` value
- The Congress.gov API allows 5,000 requests/hour

## Advanced Usage

### Python API

You can also import and use the script programmatically:

```python
from dataset.voting_record import fetch_member_votes, ensure_api_key
from pathlib import Path

# Load API key
from dataset.voting_record import load_env_file
load_env_file(Path(".env"))
api_key = ensure_api_key()

# Fetch votes
votes, member_info = fetch_member_votes(
    bioguide_id="O000172",
    api_key=api_key,
    congress=119,
    chamber="house",
    max_votes=50
)

# Process votes
for vote in votes:
    print(f"{vote['date']}: {vote['memberVote']} on {vote['legislation']['type']} {vote['legislation']['number']}")
```

## Resources

- **Congress.gov API Documentation**: https://github.com/LibraryOfCongress/api.congress.gov
- **API Sign-up**: https://api.congress.gov/sign-up/
- **Bioguide Directory**: https://bioguide.congress.gov/
- **Congress.gov**: https://www.congress.gov/

## Support

For issues or questions:
1. Check this documentation
2. Review the GitHub issues at the project repository
3. Check the Congress.gov API documentation
