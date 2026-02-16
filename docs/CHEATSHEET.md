# Hippodetector Quick Reference

## Essential Commands

### Fetch Voting Records

```bash
# Quick fetch (50 recent votes)
uv run dataset/voting_record.py --bioguide-id <ID> --congress 119 --max-votes 50

# Full congress
uv run dataset/voting_record.py --bioguide-id <ID> --congress 119

# Custom output
uv run dataset/voting_record.py --bioguide-id <ID> --congress 119 --max-votes 50 --output my_file.json
```

## Common Bioguide IDs

| Name | Party | Bioguide ID |
|------|-------|-------------|
| Alexandria Ocasio-Cortez | D | `O000172` |
| Nancy Pelosi | D | `P000197` |
| Hakeem Jeffries | D | `J000294` |
| Kevin McCarthy | R | `M001165` |
| Marjorie Taylor Greene | R | `G000596` |
| Matt Gaetz | R | `G000578` |

## Quick Data Queries (jq)

```bash
# View vote summary
cat data/votes_<ID>.json | jq '.voteSummary'

# Count total votes
cat data/votes_<ID>.json | jq '.totalVotes'

# Find specific bill
cat data/votes_<ID>.json | jq '.votes[] | select(.legislation.number == "3424")'

# List all "Nay" votes
cat data/votes_<ID>.json | jq '.votes[] | select(.memberVote == "Nay")'

# Get member info
cat data/votes_<ID>.json | jq '.member'
```

## Python Snippets

### Load voting data
```python
import json
with open('data/votes_O000172.json') as f:
    data = json.load(f)
```

### Count votes by type
```python
summary = data['voteSummary']
for vote_type, count in summary.items():
    print(f"{vote_type}: {count}")
```

### Filter votes
```python
nay_votes = [v for v in data['votes'] if v['memberVote'] == 'Nay']
```

## File Locations

- **Member list**: `data/congress_members.json`
- **Voting records**: `data/votes_<bioguide_id>.json`
- **Environment**: `.env`
- **Documentation**: `docs/`

## Common Issues

**Missing API key:**
```bash
cp .env.example .env
# Then edit .env and add your API key
```

**Member not found:**
- Use Bioguide ID instead of name
- Check `data/congress_members.json`

**Slow performance:**
- Add `--max-votes 50` to limit API calls

## Links

- Get API Key: https://api.congress.gov/sign-up/
- Find Bioguide IDs: https://bioguide.congress.gov/
- Congress.gov: https://www.congress.gov/

## Example Workflow

```bash
# 1. Get member list (one-time setup)
uv run dataset/fetch_congress_members.py

# 2. Fetch votes
uv run dataset/voting_record.py --bioguide-id O000172 --congress 119 --max-votes 50

# 3. View summary
cat data/votes_O000172.json | jq '.voteSummary'

# 4. Find specific votes
cat data/votes_O000172.json | jq '.votes[] | select(.memberVote == "Nay")'

# 5. Analyze in Python
python3 -c "
import json
with open('data/votes_O000172.json') as f:
    data = json.load(f)
print(f'Total votes: {data[\"totalVotes\"]}')
print(f'Nays: {data[\"voteSummary\"][\"Nay\"]}')
"
```
