# Usage Examples

This document provides practical examples for using the Hippodetector voting record tools.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Analyzing Voting Patterns](#analyzing-voting-patterns)
- [Python Integration](#python-integration)
- [Data Analysis](#data-analysis)

## Basic Examples

### Example 1: Get Recent Votes

Fetch the 50 most recent votes for Alexandria Ocasio-Cortez:

```bash
uv run dataset/voting_record.py \
  --bioguide-id O000172 \
  --congress 119 \
  --max-votes 50
```

**Expected output:**
```
Fetching voting record for O000172...
  Fetching House votes for Congress 119...
    Fetched 432 votes so far...
  Limited to checking the 50 most recent roll call votes

  Checking 50 roll call votes for member O000172...
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
  Not Voting: 1 (2.0%)

Detailed voting record saved to: data/votes_O000172.json
```

### Example 2: Compare Two Politicians

Fetch voting records for two members to compare:

```bash
# Get Nancy Pelosi's votes
uv run dataset/voting_record.py --bioguide-id P000197 --congress 119 --max-votes 100

# Get Kevin McCarthy's votes
uv run dataset/voting_record.py --bioguide-id M001165 --congress 119 --max-votes 100
```

### Example 3: Full Congressional Session

Get all votes from a complete congressional session (takes longer):

```bash
uv run dataset/voting_record.py \
  --bioguide-id O000172 \
  --congress 119
```

**Note:** This will fetch ~400+ votes and may take 5-10 minutes.

### Example 4: Custom Output Location

Save votes to a specific directory:

```bash
uv run dataset/voting_record.py \
  --bioguide-id J000294 \
  --congress 119 \
  --max-votes 50 \
  --output analysis/jeffries_votes.json
```

## Analyzing Voting Patterns

### Find Votes on Specific Bills

Use `jq` to filter votes for specific legislation:

```bash
# Find all votes on HR bills
cat data/votes_O000172.json | jq '.votes[] | select(.legislation.type == "HR")'

# Find votes on a specific bill
cat data/votes_O000172.json | jq '.votes[] | select(.legislation.number == "3424")'
```

### Count Vote Types

```bash
# Count "Nay" votes
cat data/votes_O000172.json | jq '[.votes[] | select(.memberVote == "Nay")] | length'

# Show voting percentages
cat data/votes_O000172.json | jq '.voteSummary'
```

### Find Votes by Date Range

```bash
# Get votes from September 2025
cat data/votes_O000172.json | jq '.votes[] | select(.date | startswith("2025-09"))'
```

## Python Integration

### Example: Load and Analyze Votes

```python
import json
from pathlib import Path

# Load voting record
with open('data/votes_O000172.json', 'r') as f:
    data = json.load(f)

# Print member info
member = data['member']
print(f"Analyzing {member['name']} ({member['party']}, {member['state']})")

# Calculate voting statistics
total = data['totalVotes']
summary = data['voteSummary']

print(f"\nVoting Statistics:")
print(f"  Total Votes: {total}")
for vote_type, count in summary.items():
    if count > 0:
        percentage = (count / total) * 100
        print(f"  {vote_type}: {count} ({percentage:.1f}%)")

# Find all "Nay" votes
nay_votes = [v for v in data['votes'] if v['memberVote'] == 'Nay']
print(f"\nFound {len(nay_votes)} 'Nay' votes")

# Show first 5
for vote in nay_votes[:5]:
    leg = vote['legislation']
    print(f"  - {leg['type']} {leg['number']} on {vote['date'][:10]}")
```

### Example: Compare Two Members

```python
import json

def load_votes(bioguide_id):
    with open(f'data/votes_{bioguide_id}.json', 'r') as f:
        return json.load(f)

# Load two members
member1 = load_votes('O000172')  # AOC
member2 = load_votes('P000197')  # Pelosi

# Compare voting patterns
print("Voting Pattern Comparison")
print("=" * 50)

def print_stats(data):
    member = data['member']
    summary = data['voteSummary']
    total = data['totalVotes']

    print(f"\n{member['name']} ({member['party']}):")
    print(f"  Yea: {summary['Yea']} ({summary['Yea']/total*100:.1f}%)")
    print(f"  Nay: {summary['Nay']} ({summary['Nay']/total*100:.1f}%)")
    print(f"  Not Voting: {summary['Not Voting']} ({summary['Not Voting']/total*100:.1f}%)")

print_stats(member1)
print_stats(member2)
```

### Example: Find Common Votes

```python
import json

def load_votes(bioguide_id):
    with open(f'data/votes_{bioguide_id}.json', 'r') as f:
        return json.load(f)

member1 = load_votes('O000172')
member2 = load_votes('P000197')

# Create lookup by roll call
votes1 = {v['rollCall']: v for v in member1['votes']}
votes2 = {v['rollCall']: v for v in member2['votes']}

# Find common roll calls
common_rolls = set(votes1.keys()) & set(votes2.keys())

print(f"Found {len(common_rolls)} common votes")

# Count agreements
agreements = 0
disagreements = 0

for roll in common_rolls:
    if votes1[roll]['memberVote'] == votes2[roll]['memberVote']:
        agreements += 1
    else:
        disagreements += 1

print(f"Agreements: {agreements} ({agreements/len(common_rolls)*100:.1f}%)")
print(f"Disagreements: {disagreements} ({disagreements/len(common_rolls)*100:.1f}%)")
```

## Data Analysis

### Export to CSV

Convert JSON voting data to CSV for spreadsheet analysis:

```python
import json
import csv

# Load votes
with open('data/votes_O000172.json', 'r') as f:
    data = json.load(f)

# Write to CSV
with open('analysis/votes.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    # Header
    writer.writerow([
        'Date', 'Congress', 'Roll Call', 'Bill Type', 'Bill Number',
        'Member Vote', 'Result', 'Party', 'State'
    ])

    # Data
    for vote in data['votes']:
        leg = vote['legislation']
        writer.writerow([
            vote['date'],
            vote['congress'],
            vote['rollCall'],
            leg['type'],
            leg['number'],
            vote['memberVote'],
            vote['result'],
            vote['party'],
            vote['state']
        ])

print("Exported to analysis/votes.csv")
```

### Party-Line Voting Analysis

Check how often a member votes with their party:

```python
import json

with open('data/votes_O000172.json', 'r') as f:
    data = json.load(f)

# This is simplified - in reality you'd need to fetch
# voting data for all party members to determine party line
member_party = data['member']['party']
votes = data['votes']

print(f"Analyzing {data['member']['name']}")
print(f"Party: {member_party}")
print(f"Total votes analyzed: {len(votes)}")

# Count vote types
vote_counts = {}
for vote in votes:
    vote_type = vote['memberVote']
    vote_counts[vote_type] = vote_counts.get(vote_type, 0) + 1

print("\nVote Distribution:")
for vote_type, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(votes)) * 100
    print(f"  {vote_type}: {count} ({percentage:.1f}%)")
```

### Generate Report

Create a markdown report of voting activity:

```python
import json
from datetime import datetime

with open('data/votes_O000172.json', 'r') as f:
    data = json.load(f)

member = data['member']
summary = data['voteSummary']

# Generate markdown report
report = f"""# Voting Record Report: {member['name']}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Member Information

- **Name:** {member['name']}
- **Party:** {member['party']}
- **State:** {member['state']}
- **Chamber:** {member['chamber']}

## Voting Summary

Total Votes Analyzed: {data['totalVotes']}

| Vote Type | Count | Percentage |
|-----------|-------|------------|
"""

for vote_type, count in summary.items():
    if count > 0:
        percentage = (count / data['totalVotes']) * 100
        report += f"| {vote_type} | {count} | {percentage:.1f}% |\n"

report += f"""
## Recent Votes

"""

# Add first 10 votes
for i, vote in enumerate(data['votes'][:10], 1):
    leg = vote['legislation']
    report += f"{i}. **{leg['type']} {leg['number']}** - {vote['date'][:10]}\n"
    report += f"   - Vote: **{vote['memberVote']}**\n"
    report += f"   - Result: {vote['result']}\n"
    report += f"   - [View on Congress.gov]({leg['url']})\n\n"

# Save report
with open(f"analysis/report_{member['bioguideId']}.md", 'w') as f:
    f.write(report)

print(f"Report saved to analysis/report_{member['bioguideId']}.md")
```

## Advanced Examples

### Batch Processing Multiple Members

Process voting records for multiple politicians:

```bash
#!/bin/bash

# List of Bioguide IDs
MEMBERS=(
    "O000172"  # AOC
    "P000197"  # Pelosi
    "M001165"  # McCarthy
    "J000294"  # Jeffries
)

# Fetch votes for each
for member in "${MEMBERS[@]}"; do
    echo "Fetching votes for $member..."
    uv run dataset/voting_record.py \
        --bioguide-id "$member" \
        --congress 119 \
        --max-votes 50
    echo "Done with $member"
    echo "---"
done

echo "All members processed!"
```

### Automated Daily Updates

Set up a cron job to fetch latest votes daily:

```bash
# crontab entry (runs daily at 9 AM)
0 9 * * * cd /path/to/Hippodetector && /path/to/uv run dataset/voting_record.py --bioguide-id O000172 --congress 119 --max-votes 10
```

## Tips and Best Practices

1. **Start Small**: Use `--max-votes 50` for testing, then increase
2. **Cache Results**: Save JSON files and reuse them to avoid redundant API calls
3. **Batch Processing**: Wait between requests when processing multiple members
4. **Error Handling**: Check for API errors and handle rate limiting
5. **Data Validation**: Verify the fetched data before analysis

## Resources

- [Voting Record API Documentation](voting_record_api.md)
- [Python JSON Documentation](https://docs.python.org/3/library/json.html)
- [jq Manual](https://stedolan.github.io/jq/manual/)
