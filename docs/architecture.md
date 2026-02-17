# RAG System Design Document

## Overview

This document describes the design of the Hippodetector RAG system for detecting contradictions between congressional members' voting records and public statements.

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION                                               │
│                                                                  │
│  Member Profile (data/members/B001316.json)                     │
│  ├── metadata: name, party, state                               │
│  ├── votes: [{billId, memberVote, voteDate, ...}]              │
│  ├── bills: [{billId, title, summary, subjects, ...}]          │
│  └── pressReleases: [{id, title, bodyText, date, ...}]         │
│                                                                  │
│  ↓ join votes + bills by billId                                 │
│  ↓ create embedding texts                                       │
│                                                                  │
│  Embeddings Generated (google/embeddinggemma-300m)              │
│  ├── Bill embeddings: title + summary + subjects + vote         │
│  └── PR embeddings: title + bodyText                            │
│                                                                  │
│  ↓ load into Qdrant                                             │
│                                                                  │
│  Vector Database (Qdrant)                                       │
│  ├── Collection: bills (273 points)                             │
│  └── Collection: press_releases (10 points)                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2. QUERY PROCESSING                                             │
│                                                                  │
│  User Query: "Find contradictions about healthcare"             │
│                                                                  │
│  ↓ generate embedding                                           │
│                                                                  │
│  RAG/search.py                                                  │
│  ├── Search bills collection (top-k by cosine similarity)       │
│  ├── Search press_releases collection (top-k)                   │
│  └── Return: List[VoteEvidence], List[RawPressRelease]          │
│                                                                  │
│  Retrieved Items:                                               │
│  ├── 5-10 relevant bills with vote positions                    │
│  └── 3-5 relevant press releases                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 3. STANCE EXTRACTION (Post-Processing)                          │
│                                                                  │
│  RAG/extract_stances.py                                         │
│                                                                  │
│  For each retrieved press release:                              │
│  ├── Use LLM to extract structured stance                       │
│  ├── Input: PR title + bodyText                                 │
│  ├── Prompt: "Extract stance on [issue] from this statement"    │
│  └── Output: IssueStance (dataset/memberOpinions.py)            │
│      ├── status: "supports" | "opposes" | "mixed"               │
│      ├── summary: brief description                             │
│      └── source_url: PR url                                     │
│                                                                  │
│  Extracted Stances:                                             │
│  ├── healthcare: {status: "opposes", summary: "..."}            │
│  ├── immigration: {status: "supports", summary: "..."}          │
│  └── ...                                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 4. TOPIC MATCHING                                               │
│                                                                  │
│  Map bill subjects to issue categories:                         │
│                                                                  │
│  Bill Subjects          →  Issue Category                       │
│  ─────────────────────────────────────────                      │
│  "Healthcare"           →  health_care                           │
│  "Medicare"             →  health_care                           │
│  "Budget deficits"      →  budget_economy                        │
│  "Immigration"          →  immigration                           │
│  ...                                                             │
│                                                                  │
│  Matched Bills:                                                 │
│  ├── 119-hr-3424 (Healthcare) → health_care stance              │
│  └── 119-hr-5348 (Immigration) → immigration stance             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 5. CONTRADICTION DETECTION                                      │
│                                                                  │
│  RAG/detect_contradictions.py                                   │
│                                                                  │
│  For each (bill, stance) pair on same topic:                    │
│                                                                  │
│  ├── Compare vote position vs stated stance                     │
│  │   ├── Vote: "Yea" on healthcare bill                         │
│  │   └── Stance: "opposes" federal healthcare regulations       │
│  │                                                               │
│  ├── Determine contradiction severity                           │
│  │   ├── "direct": clear opposition                             │
│  │   ├── "moderate": contextual difference                      │
│  │   ├── "weak": minor inconsistency                            │
│  │   └── "nuanced": complex/debatable                           │
│  │                                                               │
│  ├── Use LLM to generate explanation                            │
│  │   ├── Input: bill summary + vote + PR excerpt + stance       │
│  │   └── Output: natural language explanation                   │
│  │                                                               │
│  └── Build Contradiction object (contradiction_schema.py)       │
│      ├── statement: StatementEvidence                           │
│      ├── vote: VoteEvidence                                     │
│      ├── explanation: str                                       │
│      ├── severity: ContradictionSeverity                        │
│      └── confidenceScore: float                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 6. OUTPUT GENERATION                                            │
│                                                                  │
│  ContradictionReport (contradiction_schema.py)                  │
│  ├── query: "Find contradictions about healthcare"              │
│  ├── memberName: "Burlison, Eric"                               │
│  ├── bioguideId: "B001316"                                      │
│  ├── party: "Republican"                                        │
│  ├── contradictions: [                                          │
│  │     {                                                         │
│  │       issueCategory: "health_care",                          │
│  │       severity: "moderate",                                  │
│  │       statement: {...},                                      │
│  │       vote: {...},                                           │
│  │       explanation: "...",                                    │
│  │       confidenceScore: 0.75                                  │
│  │     }                                                         │
│  │   ]                                                           │
│  ├── totalFound: 1                                              │
│  └── executionTimeMs: 1542.3                                    │
│                                                                  │
│  ↓ return to Streamlit UI                                       │
│                                                                  │
│  User sees:                                                     │
│  ├── Contradiction summary                                      │
│  ├── Source citations (bill + PR links)                         │
│  ├── Confidence score                                           │
│  └── Detailed explanation                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Post-Retrieval Stance Extraction (Option 3)

**Why?**
- Preserves flexibility of semantic search
- Only processes retrieved items (cost-efficient)
- Enables structured output without upfront categorization

**Trade-offs:**
- Can't pre-filter by issue category
- More complex query-time logic
- Dependent on retrieval quality

### 2. Topic Matching Strategy

**Bill Subjects → Issue Categories:**
- Use fuzzy matching + keyword mapping
- Multiple subjects can map to one issue
- One subject can map to multiple issues (handled in contradiction detection)

**Example Mappings:**
```python
SUBJECT_TO_ISSUE_MAP = {
    "health care": ["health_care"],
    "medicare": ["health_care", "social_security"],
    "medicaid": ["health_care", "welfare_poverty"],
    "immigration": ["immigration"],
    "border security": ["immigration", "homeland_security"],
    ...
}
```

### 3. Contradiction Severity Scoring

**Criteria:**
1. **Direct**: Vote directly contradicts stated position
   - Example: Says "oppose X", votes Yes on X
2. **Moderate**: Contradiction in related context
   - Example: Says "reduce regulations", votes for regulatory bill
3. **Weak**: Minor inconsistency or timing difference
   - Example: Old statement, recent changed vote
4. **Nuanced**: Complex, debatable, or requires context
   - Example: Supports concept but opposes specific implementation

### 4. Confidence Scoring

**Factors:**
1. Topic match strength (bill subjects ↔ issue category)
2. Clarity of extracted stance (supports vs mixed)
3. Recency (older statements = lower confidence)
4. LLM's own confidence in extraction

## Schema Relationships

```
memberOpinions.py (Input)
├── IssueStance
│   ├── status: StanceStatus
│   ├── summary: str
│   └── source_url: str
└── CandidateIssueProfile
    ├── health_care: IssueStance
    ├── immigration: IssueStance
    └── ... (24 categories)

contradiction_schema.py (Output)
├── VoteEvidence (from Qdrant bills collection)
│   ├── billId, title, summary
│   ├── memberVote, voteDate
│   └── subjects
├── StatementEvidence (from Qdrant press_releases + LLM extraction)
│   ├── id, title, date, url
│   ├── excerpt
│   ├── extractedStance (from IssueStance.status)
│   └── stanceSummary (from IssueStance.summary)
├── Contradiction
│   ├── statement: StatementEvidence
│   ├── vote: VoteEvidence
│   ├── explanation: str (LLM-generated)
│   ├── severity: ContradictionSeverity
│   └── confidenceScore: float
└── ContradictionReport
    ├── query: str
    ├── member metadata
    ├── contradictions: List[Contradiction]
    └── search metadata
```

## Automated Pipeline

The project includes an end-to-end automated pipeline for processing politicians:

### Pipeline Script: `run_contradiction_pipeline.py`

**Execution Steps:**
1. **Fetch Voting Records** (`dataset/voting_record.py`)
   - Retrieves all votes from Congress API
   - Stores in `data/votes/{bioguide_id}.json`

2. **Fetch Bill Details** (`dataset/fetch_bill_details.py`)
   - For each voted bill, fetches full details from Congress API
   - Includes: title, summary, subjects, cosponsors
   - Stores in `data/bills/{bioguide_id}.json`

3. **Build Member Profile** (`dataset/build_member_profile.py`)
   - Joins votes + bills + press releases
   - Creates unified profile: `data/members/{bioguide_id}.json`
   - Profile structure: metadata, votes[], bills[], pressReleases[]

4. **Load Embeddings** (`RAG/load_embeddings.py`)
   - Generates 768-dim embeddings using `google/embeddinggemma-300m`
   - **Optimization**: Uses pre-computed PR embeddings for 438 House members
   - Loads into Qdrant collections: `bills`, `press_releases`

5. **Detect Contradictions** (TODO)
   - Will implement full contradiction detection logic
   - Can be skipped with `--skip-contradictions` to run Steps 1-4 only

### Bash Wrapper: `run_pipeline.sh`

Convenient command-line interface:

```bash
# Single politician
./run_pipeline.sh B001316

# Sample 20 politicians
./run_pipeline.sh --sample

# From custom file
./run_pipeline.sh --file my_politicians.txt

# Data collection only (skip contradiction detection)
./run_pipeline.sh --sample --skip-contradictions

# Skip data collection (use existing data)
./run_pipeline.sh B001316 --skip-all
```

**Features:**
- Progress bar for multi-politician processing (tqdm)
- House member validation (filters non-House members)
- Color-coded output
- Skip flags for partial pipeline execution (`--skip-voting`, `--skip-bills`, `--skip-profile`, `--skip-embeddings`, `--skip-contradictions`)

### Pre-computed Embeddings

**Location:**
- `data/press_release_embeddings_1.zip` (Members A-L)
- `data/press_release_embeddings_2.zip` (Members M-Z)

**Contents:**
- 438 House members with pre-computed PR embeddings
- ~533MB compressed
- Saves LLM API costs and processing time

**Usage:**
```bash
# Pipeline automatically uses pre-computed embeddings
./run_pipeline.sh --bioguide-ids B001316

# Or manually with flag
uv run python RAG/load_embeddings.py --bioguide-id B001316 --use-precomputed-pr
```

## Implementation Roadmap

1. ✅ **Data Ingestion**: Load embeddings into Qdrant
2. ✅ **Pipeline Automation**: End-to-end pipeline with progress tracking
3. ✅ **Pre-computed Embeddings**: Optimized PR embedding loading
4. ✅ **Topic Matching**: Map bill subjects to issue categories (RAG/topic_matching.py)
5. ✅ **Stance Extraction**: LLM extraction using IssueStance schema (RAG/extract_stances.py)
6. ⏳ **Search Function**: Implement semantic search (RAG/search.py)
7. ⏳ **Contradiction Detection**: Compare and score (RAG/detect_contradictions.py)
8. ⏳ **Streamlit UI**: Display ContradictionReport to users

## Future Enhancements

- **Multi-member queries**: Compare positions across members
- **Temporal analysis**: Track position changes over time
- **Bill co-sponsorship**: Include co-sponsored bills as additional evidence
- **Social media integration**: Analyze tweets/statements beyond press releases
- **Explainability**: Add citation highlighting and reasoning chains
