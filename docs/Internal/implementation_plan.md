# Political Claims vs. Voting Record Analysis — Implementation Plan

## Project Overview

**Objective:** Analyse a politician's public claims/opinions against their voting record to detect alignment or contradiction.

**Premise:** A politician's beliefs are embedded in how they vote. Bill content may not always be agreeable, so metrics are needed to determine vote rationale, followed by post-processing to assess whether public statements align with voting patterns.

**Timeline:** 4 days | **Team:** 5 members

---

## Assumptions (Confirm/Adjust)

- **Target:** Single U.S. Congress politician
- **Data:** ProPublica Congress API (votes), official press releases / social media (statements)
- **Stack:** Python, LLM API for classification, pandas for analysis
- **Output:** Proof-of-concept notebook + summary report + presentation

---

## Team Roles

| Role | ID | Focus |
|---|---|---|
| **Lead A** — Data Engineer | M1 | Votes & bills pipeline |
| **Lead B** — Data Engineer | M2 | Statements pipeline |
| **Lead C** — LLM Specialist | M3 | All LLM prompt design, classification, explanation generation |
| **Lead D** — Analytics | M4 | Alignment model, scoring logic, disagreement detection |
| **Lead E** — Analytics/Viz | M5 | Visualization, integration, report structure |

> M3 is one of the 1–2 members with LLM API experience. The other (if applicable) pairs with M3 or serves as M4/M5.

---

## Day 1 — Data Acquisition + Topic/Stance Extraction

### Morning: Votes & Bill Data (M1, 3–4 hrs)

1. **Set up ProPublica Congress API** — obtain free API key.
2. **Pull voting record** for target politician — endpoint: `members/{member-id}/votes.json`. Extract: bill ID, vote (yes/no/abstain), date.
3. **Pull bill summaries** via Congress.gov API / GovTrack. Store: bill ID, title, official summary, subjects/tags.
4. **Normalize into structured dataframe:** `bill_id | title | summary | subjects | vote | date`

### Morning: Statements Corpus (M2, 3–4 hrs)

5. **Identify statement sources** (pick 1–2 max):
   - Official website press releases (scrape or RSS)
   - Twitter/X via export or API
   - Floor speeches via Congressional Record API
6. **Extract and store:** `statement_id | date | source | raw_text`
7. **Quality pass:** Deduplicate, remove boilerplate, confirm date coverage overlaps with voting data.

### Morning: Taxonomy + Prompt Design (M3, 3–4 hrs)

8. **Define topic taxonomy** (15–25 categories) from CRS bill subject tags, consolidated into workable clusters.
   - Examples: healthcare, immigration, defense, taxation, environment, education, gun policy, trade, judiciary, infrastructure.
9. **Write bill classification prompt** — input: bill summary → output: primary/secondary topic + confidence.
10. **Write statement stance extraction prompt** — input: claim text → output: topic, stance, strength, evidence quote.

### Morning: Scoring Design + Repo Setup (M4, M5)

11. **M4:** Define alignment score formula. Draft scoring logic pseudocode.
12. **M5:** Set up project repo, shared data schema, output folder structure.

### Afternoon: Classification + Prototyping

| Member | Task |
|---|---|
| M1 | Normalize votes dataframe. Cache all raw API responses. |
| M2 | Clean statements corpus. Deduplicate, remove boilerplate. |
| M3 | Run bill classification (CRS tags first pass → LLM second pass for gaps). Validate ~20 samples. |
| M4 | Review taxonomy with M3 — ensure it's scorable. Begin alignment module skeleton. |
| M5 | Research viz libraries. Prototype heatmap + timeline chart with dummy data. |

### Day 1 Deliverable

- `votes_bills.parquet` — votes with topic labels
- `statements.parquet` — cleaned statements corpus
- Taxonomy finalized
- Scoring module scaffolded
- Viz prototypes with dummy data

### Critical Handoff

M1 → M3 (bill summaries for classification), M2 → M3 (statements for Day 2 processing).

---

## Day 2 — Stance Extraction + Scoring + Visualization

### Morning

| Member | Task |
|---|---|
| M1 | Build end-to-end pipeline runner (parameterized by member ID). Add error handling. |
| M2 | Assist M3 — manual validation of statement stance extraction (~20+ samples). |
| M3 | Run stance extraction on full statements corpus. Iterate prompt if validation < 80%. |
| M4 | Implement alignment scoring on real bill data (from Day 1 output). |
| M5 | Connect real data to heatmap + timeline prototypes. |

### Afternoon

| Member | Task |
|---|---|
| M1 | Write data refresh / caching logic. Document pipeline. |
| M2 | Build disagreement register structure. Populate with M4's flagged cases. |
| M3 | Generate LLM explanations for top contradictions (context: bill + vote + stated position). |
| M4 | Compute per-topic scores. Flag low-N topics, omnibus bills. Identify top contradictions → pass to M2 & M3. |
| M5 | Build contradiction spotlight viz. Summary stats dashboard. |

### Day 2 Deliverable

- Reusable pipeline complete
- Disagreement register populated
- All LLM outputs finalized
- Alignment scores + contradiction list finalized
- All visualizations rendering with real data

### Critical Handoff

M4 → M3 (contradiction cases for explanation), M4 → M5 (scores for viz), M3 → M2 (stance outputs for register).

---

## Day 3 — Testing & Validation

| Member | Task |
|---|---|
| **M1** | End-to-end rerun from fresh API pull. Fix pipeline breaks. Test with a second politician if time permits. |
| **M2** | Manual audit of top 10 contradictions against source material. Document any misclassifications. |
| **M3** | Refine LLM prompts based on M2's audit. Sensitivity test: vary confidence thresholds. |
| **M4** | Sensitivity testing: vary alignment thresholds, topic granularity. Confirm robustness. Write methodology section. |
| **M5** | Polish visualizations. Begin assembling presentation deck skeleton. Draft limitations section. |

### Day 3 Deliverable

- All outputs validated
- Audited results with confidence notes
- Presentation deck skeleton ready

---

## Day 4 — Presentation

### Slide Ownership

| Member | Section |
|---|---|
| **M1** | Data sources & pipeline architecture |
| **M2** | Contradiction deep-dives (2–3 case studies) |
| **M3** | LLM methodology — classification approach, validation results, confidence |
| **M4** | Alignment model, scoring results, sensitivity analysis |
| **M5** | Visualizations, integration, overall narrative flow |

### Schedule

- **Morning:** Draft slides independently.
- **Midday:** Integrate + cross-review.
- **Afternoon:** Rehearse, refine, finalize.

### Suggested Deck Structure

1. Problem statement & approach
2. Data sources & pipeline architecture
3. Methodology (taxonomy, classification, scoring)
4. Key findings (heatmap, timeline, contradiction spotlight)
5. Case study deep-dives
6. Limitations & confidence discussion
7. Extensions / next steps

---

## Visualizations (Target: 3–4 Charts)

| Chart | Purpose |
|---|---|
| **Topic alignment heatmap** | Topics on y-axis, alignment score as color gradient |
| **Timeline view** | Votes + statements plotted chronologically per topic — shows position shifts |
| **Contradiction spotlight** | Top 5 misalignments with supporting evidence (bill excerpt + statement excerpt) |
| **Summary statistics** | Overall alignment score, most/least consistent topics, statement-to-vote ratio |

---

## Coordination Protocol

- **Daily EOD sync** (30 min) — blockers, handoffs, next-day priorities.
- **Shared repo** with agreed data schema by Day 1 morning.
- **Data contract:** All intermediate outputs as `.parquet` with documented column schemas.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| API rate limits delay Day 1 | Medium | High | Cache aggressively; GovTrack as fallback |
| LLM misclassifies topics/stances | Medium | High | Manual validation sample; confidence thresholds; cap at 2 prompt revisions |
| Too few statements on key topics | Medium | Medium | Lower topic granularity; merge related categories |
| Omnibus bills pollute signal | High | Medium | Flag and run analysis with/without; report both |
| Visualization scope creep | Medium | Medium | Commit to 3–4 charts max; polish on Day 3 |
| Pipeline breaks on rerun (Day 3) | Low | High | Cache all raw API responses on Day 1 |
| Scope creep into multi-politician comparison | Low | High | Defer — design pipeline to be parameterizable but execute for one |

---

## Key Decisions Required Before Starting

1. **Target politician** — name / member ID.
2. **Statement sources** — which 1–2 channels to collect.
3. **Time horizon** — which legislative session(s) to cover.
4. **LLM provider** — OpenAI / Anthropic / other + model selection.
5. **Alignment threshold** — what score constitutes "misalignment" (suggested: < 0.5).

---
---

# How-To Reference Guide

This section provides detailed implementation guidance for every complex task in the plan. Each how-to is tagged with the responsible team member and the day it is executed.

---

## How-To: Pulling Voting Records via API (M1, Day 1)

### Data Sources (in priority order)

1. **ProPublica Congress API** — free, structured, well-documented.
2. **Congress.gov API** — official, more comprehensive, slightly harder to paginate.
3. **GovTrack API** — good fallback, includes bill metadata.

### ProPublica Workflow

**Step 1: Get the member ID.**

Look up the politician at `https://api.propublica.org/congress/v1/members.json` or use their Bioguide ID (findable on congress.gov).

**Step 2: Pull vote positions.**

```
GET https://api.propublica.org/congress/v1/members/{member-id}/votes.json
Headers: X-API-Key: {your-key}
```

Response returns paginated results (20 per page). Loop through all pages:

```python
import requests, json, os

API_KEY = os.environ["PROPUBLICA_KEY"]
MEMBER_ID = "S001191"  # example
BASE_URL = f"https://api.propublica.org/congress/v1/members/{MEMBER_ID}/votes.json"

all_votes = []
offset = 0

while True:
    resp = requests.get(
        BASE_URL,
        headers={"X-API-Key": API_KEY},
        params={"offset": offset}
    )
    data = resp.json()

    # Cache raw response immediately
    with open(f"data/raw/votes/page_{offset}.json", "w") as f:
        json.dump(data, f)

    votes = data["results"][0]["votes"]
    if not votes:
        break

    all_votes.extend(votes)
    offset += 20
```

**Step 3: Extract relevant fields per vote.**

```python
records = []
for v in all_votes:
    records.append({
        "bill_id": v.get("bill", {}).get("bill_id"),       # e.g. "hr1234-118"
        "bill_title": v.get("bill", {}).get("title"),
        "vote_position": v.get("position"),                  # Yes / No / Not Voting
        "vote_date": v.get("date"),
        "vote_description": v.get("description"),
        "roll_call": v.get("roll_call"),
        "congress": v.get("congress"),
        "session": v.get("session"),
    })
```

**Step 4: Pull bill summaries separately.**

ProPublica vote data includes bill titles but not full summaries. Use Congress.gov API:

```
GET https://api.congress.gov/v2/bill/118/hr/1234?api_key=XXX
```

The `summaries` field contains CRS-authored summaries. Cache each bill response individually (`data/raw/bills/HR1234.json`).

**Step 5: Merge into a single dataframe.**

```python
import pandas as pd

votes_df = pd.DataFrame(records)
bills_df = pd.DataFrame(bill_summaries)  # from Congress.gov
merged = votes_df.merge(bills_df, on="bill_id", how="left")
merged.to_parquet("data/processed/votes_bills.parquet")
```

### Output Schema

```
bill_id | bill_title | summary | crs_subjects | vote_position | vote_date | congress | session
```

### Pitfalls

- **Nominations & procedural votes** have no `bill` object — filter these out or tag as `procedural`.
- **Pagination** — ProPublica returns max 20 per request; Congress.gov returns max 250. Always loop until empty.
- **Rate limits** — ProPublica: 5,000 requests/day. Congress.gov: 1,000/hour. Pace requests with `time.sleep(0.5)`.

---

## How-To: Collecting the Statements Corpus (M2, Day 1)

### Source Priority (pick 1–2)

| Source | Pros | Cons |
|---|---|---|
| **Official press releases** (politician's .gov website) | Formal positions, topically rich | May need scraping; inconsistent formatting |
| **Floor speeches** (Congressional Record API) | On-the-record, linked to legislative action | Verbose; may require chunking |
| **Twitter/X** | High volume, real-time positions | Noisy; sarcasm/retweets complicate stance extraction |
| **Campaign website** | Clear policy stances | Static; may not reflect evolving positions |

### Press Release Collection

Most congressional websites host press releases at predictable URLs:

```
https://www.{lastname}.senate.gov/newsroom/press-releases
https://{lastname}.house.gov/media/press-releases
```

**Scraping approach (Python + BeautifulSoup):**

```python
import requests
from bs4 import BeautifulSoup
import json

BASE_URL = "https://www.example.senate.gov/newsroom/press-releases"

def scrape_press_releases(base_url, max_pages=10):
    releases = []
    for page in range(1, max_pages + 1):
        resp = requests.get(f"{base_url}?page={page}")
        soup = BeautifulSoup(resp.text, "html.parser")

        # Structure varies per site — inspect HTML first
        for item in soup.select(".press-release-item"):  # adjust selector
            title = item.select_one("h3").text.strip()
            date = item.select_one(".date").text.strip()
            link = item.select_one("a")["href"]

            # Fetch full text
            detail = requests.get(link)
            detail_soup = BeautifulSoup(detail.text, "html.parser")
            body = detail_soup.select_one(".press-release-body").text.strip()

            releases.append({
                "statement_id": f"pr_{page}_{len(releases)}",
                "date": date,
                "source": "press_release",
                "title": title,
                "url": link,
                "raw_text": body
            })

            # Cache raw HTML
            with open(f"data/raw/statements/press_releases/{title[:50]}.html", "w") as f:
                f.write(detail.text)

    return releases
```

### Floor Speeches via Congressional Record

```
GET https://api.congress.gov/v2/congressional-record?api_key=XXX
```

Filter by member name or use the GPO's Bound Congressional Record. These are lengthy — plan to chunk them during stance extraction (Day 2).

### Cleaning Pipeline

1. **Remove boilerplate** — headers, footers, "Contact: office@..." blocks, "###" markers.
2. **Deduplicate** — same release posted across multiple pages.
3. **Date normalization** — parse all dates to `YYYY-MM-DD`.
4. **Coverage check** — confirm statement dates overlap with voting record date range. Flag gaps.

### Output Schema

```
statement_id | date | source | title | url | raw_text
```

### Pitfalls

- **Website redesigns** break scrapers — inspect the HTML structure before coding; cache pages immediately.
- **Co-signed statements** — multiple politicians on one release. Tag these and decide whether to include.
- **Social media noise** — if using Twitter/X, filter to original tweets only (no retweets, no replies to constituents).

---

## How-To: Caching Raw API Responses (M1, Day 1)

**What it means:** Save every API response to local files (JSON) immediately upon retrieval, before any processing.

**Why it matters:**

1. **API rate limits** — ProPublica and Congress.gov impose request caps. If the pipeline fails midway through Day 2 or Day 3, cached files allow rerunning offline without re-fetching hundreds of requests.
2. **Reproducibility** — Day 3 requires an end-to-end rerun for validation. If the API returns slightly different data (e.g., a vote gets corrected, a bill summary updates), results change. Cached data ensures consistent inputs across runs.
3. **Speed** — API calls are slow (network latency, pagination). Reading from local `.json` files is near-instant. This matters when M3 and M4 are iterating on classification and scoring logic.

**Directory structure:**

```
data/
  raw/
    votes/
      page_0.json         # Raw API response, untouched
      page_20.json
    bills/
      HR1234.json          # One file per bill
      S567.json
    statements/
      press_releases/
        raw_page_1.html
  processed/
    votes_bills.parquet    # Cleaned, merged, ready for analysis
    statements.parquet
```

**Pattern:** `fetch → save raw → process from raw`. Never process directly from the API response without saving it first. M1's pipeline runner on Day 2 should check for cached files before making any API call — fetch only what's missing.

**Cache-aware fetch pattern:**

```python
import os, json, requests

def fetch_with_cache(url, cache_path, headers=None):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)

    return data
```

---

## How-To: Creating the Topic Taxonomy (M3, Day 1 — M4 Review)

### Step 1: Start from CRS Subject Tags (Structured, Free)

Every bill in Congress.gov has a **Congressional Research Service (CRS) policy area** tag — a curated taxonomy of ~32 top-level subjects. Pull these programmatically:

```
GET https://api.congress.gov/v2/bill?congress=118&api_key=XXX
```

Each bill returns a `policyArea` field (e.g., "Health", "Armed Forces and National Security", "Taxation").

**Problem:** 32 categories is too granular for a 4-day project with potentially sparse data. Some categories will have 0–1 votes.

### Step 2: Consolidate into 12–18 Workable Clusters

Group CRS tags by policy affinity:

| Consolidated Topic | CRS Tags Merged |
|---|---|
| Healthcare | Health, Medicare/Medicaid |
| Defense & Security | Armed Forces, Homeland Security, Intelligence |
| Economy & Taxation | Taxation, Finance, Economics, Commerce |
| Immigration | Immigration |
| Environment & Energy | Environmental Protection, Energy, Public Lands |
| Education | Education, Higher Education |
| Criminal Justice | Crime and Law Enforcement, Judiciary |
| Social Policy | Social Welfare, Civil Rights, Labor |
| Infrastructure | Transportation, Housing, Water Resources |
| Foreign Affairs | International Affairs, Trade, Sanctions |
| Government Operations | Government Operations, Congressional Oversight |
| Gun Policy | Firearms (often tagged under Crime — separate due to political salience) |

**Why consolidate:** You need enough votes *and* statements per topic to compute a meaningful alignment score. Fewer than 3 data points per topic = unreliable signal.

### Step 3: Validate & Finalize

1. **Pull all unique CRS `policyArea` values** from the target politician's voted-on bills.
2. **Check distribution** — how many bills per tag? Merge any tag with < 3 bills into a parent cluster.
3. **Check political salience** — split any merged cluster where sub-topics have opposing political valence (e.g., "Crime" should separate gun policy from sentencing reform if the politician has distinct stances on each).
4. **M4 reviews** — confirm every topic is scorable (i.e., a yes/no vote can be interpreted as support/opposition on that topic).

### Step 4: Store as Configuration

Save as a reusable JSON config that all modules reference:

```json
{
  "taxonomy_version": "1.0",
  "topics": {
    "healthcare": {
      "crs_tags": ["Health", "Medicare/Medicaid"],
      "description": "Healthcare policy, insurance, public health"
    },
    "defense_security": {
      "crs_tags": ["Armed Forces and National Security", "Homeland Security", "Intelligence"],
      "description": "Military, national security, intelligence"
    }
  }
}
```

### Edge Case: Multi-Topic Bills

Some bills span multiple topics (e.g., an infrastructure bill with tax provisions and environmental regulations). Handle via the LLM second pass:

- Assign `primary_topic` (what the bill is mainly about) and `secondary_topic`.
- Alignment scoring uses `primary_topic` by default.
- Sensitivity testing on Day 3 can check whether including secondary topics changes results.

The taxonomy doesn't need to be perfect — it needs to be **consistent** across bills and statements so comparisons are valid.

---

## How-To: Bill Classification LLM Prompt (M3, Day 1)

### Two-Pass Approach

**Pass 1 (free, no LLM):** Use existing CRS `policyArea` tags. Map each to the consolidated taxonomy using the JSON config. This covers ~70–80% of bills.

**Pass 2 (LLM):** For bills with missing, vague, or multi-topic CRS tags, send the bill summary to the LLM.

### Prompt Template

```
You are a legislative analyst. Classify the following bill into the provided topic taxonomy.

TAXONOMY:
{insert taxonomy JSON — topic names + descriptions}

BILL:
Title: {bill_title}
Summary: {bill_summary}

Respond in JSON only, no other text:
{
  "primary_topic": "<topic_key from taxonomy>",
  "secondary_topic": "<topic_key or null>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining classification>"
}
```

### Implementation Notes

- **Temperature:** Set to 0 for deterministic, reproducible classification.
- **Batch processing:** Send bills in batches of 5–10 per API call if the model supports it, to reduce latency and cost.
- **Confidence threshold:** If `confidence < 0.6`, flag for manual review by M4.
- **Fallback:** If the LLM returns an invalid topic key, map to `"other"` and flag.

### Validation (Day 1 Afternoon)

Manually check ~20 classified bills:

- Does the `primary_topic` match your intuitive reading of the bill?
- Are multi-topic bills assigned the correct *primary* topic?
- Are confidence scores calibrated? (i.e., low confidence on genuinely ambiguous bills, high on clear-cut ones)

If accuracy < 80%, revise the prompt — typically the taxonomy descriptions need more specificity.

---

## How-To: Statement Segmentation into Claims (M3, Day 2)

### Why Segment?

A single press release or speech may contain positions on 3–5 different topics. If you score the entire statement as one unit, you lose topic-level granularity. Segmentation splits compound statements into individual **claims** — each mapped to one topic and one stance.

### Prompt Template

```
You are a political analyst. The following is a public statement by a politician.
Break it into individual political claims or positions. Each claim should express
ONE opinion on ONE topic.

STATEMENT:
{raw_text}

Respond in JSON only, no other text:
[
  {
    "claim_id": 1,
    "claim_text": "<extracted or paraphrased claim>",
    "original_excerpt": "<verbatim quote from the statement supporting this claim>"
  },
  ...
]

Rules:
- Only extract claims that express a political position, opinion, or policy stance.
- Skip procedural language, greetings, and biographical content.
- If the statement contains no political claims, return an empty array [].
```

### Example Input/Output

**Input (press release excerpt):**
> "Today I voted against HR 4521 because it fails to protect our veterans' healthcare benefits. I also remain committed to securing our southern border and will continue to oppose any amnesty provisions."

**Output:**
```json
[
  {
    "claim_id": 1,
    "claim_text": "Opposes HR 4521 due to insufficient veteran healthcare protections",
    "original_excerpt": "I voted against HR 4521 because it fails to protect our veterans' healthcare benefits"
  },
  {
    "claim_id": 2,
    "claim_text": "Supports border security and opposes amnesty provisions",
    "original_excerpt": "I remain committed to securing our southern border and will continue to oppose any amnesty provisions"
  }
]
```

### Pitfalls

- **Over-segmentation** — splitting one coherent position into multiple fragments. The prompt should specify "ONE opinion on ONE topic" to prevent this.
- **Long speeches** — if a floor speech exceeds the model's context window, pre-chunk by paragraph before sending.
- **Boilerplate claims** — "I'm fighting for hardworking families" is too vague to score. The prompt's instruction to skip non-position language helps, but flag claims with no specific policy content during validation.

---

## How-To: Statement Stance Extraction (M3, Day 2 — M2 Validates)

### Purpose

For each segmented claim, determine: what topic is it about, and does the politician support or oppose it?

### Prompt Template

```
You are a political analyst. For the following claim by a politician, determine
the topic and the politician's stance.

TAXONOMY:
{insert taxonomy JSON — topic names + descriptions}

CLAIM:
{claim_text}

ORIGINAL EXCERPT:
{original_excerpt}

Respond in JSON only, no other text:
{
  "topic": "<topic_key from taxonomy>",
  "stance": "supports" | "opposes" | "neutral",
  "strength": "strong" | "moderate" | "weak",
  "reasoning": "<one sentence explaining stance classification>"
}

Rules:
- "supports" = the politician is in favour of action/policy on this topic.
- "opposes" = the politician is against action/policy on this topic.
- "neutral" = the politician mentions the topic without a clear directional stance.
- "strong" = explicit, unequivocal language ("I will always fight for...", "I firmly oppose...").
- "moderate" = clear position but hedged ("I generally support...", "I have concerns about...").
- "weak" = implied position, vague language.
```

### Mapping Stance to Voting Direction

This is the critical link between statements and votes. The mapping depends on what the **bill does**, not just the topic:

| Bill Action | Vote = Yes | Vote = No |
|---|---|---|
| Bill **expands** healthcare | Supports healthcare | Opposes healthcare expansion |
| Bill **restricts** immigration | Supports restriction | Opposes restriction |
| Bill **increases** defense spending | Supports defense spending | Opposes defense spending |

M3 must encode the bill's **directional intent** during bill classification (Step 9 on Day 1). Add a field:

```json
{
  "primary_topic": "healthcare",
  "bill_direction": "expands",
  "confidence": 0.85
}
```

This allows M4's scoring logic to correctly interpret a "Yes" vote as support or opposition depending on what the bill actually does.

### Validation Protocol (M2, Day 2 Morning)

1. Take a random sample of 20–25 stance extractions.
2. For each, read the original statement excerpt and independently assess:
   - Is the topic correct?
   - Is the stance (supports/opposes/neutral) correct?
   - Is the strength reasonable?
3. Calculate accuracy: `correct / total`. Target: ≥ 80%.
4. If < 80%, categorize errors (wrong topic? wrong stance? wrong strength?) and feed back to M3 for prompt revision.
5. **Cap at 2 prompt revisions** to stay on schedule.

---

## How-To: Flagging Omnibus & Procedural Bills (M4, Day 2)

### Why Flag?

**Omnibus bills** bundle multiple unrelated provisions. A politician may vote "Yes" on an omnibus despite opposing specific provisions within it — or vote "No" despite supporting most of it. Using these votes as clean topic signals introduces noise.

**Procedural votes** (cloture, motion to table, motion to recommit) reflect legislative strategy, not policy positions.

### Detection Rules

**Omnibus flag — apply if ANY of these are true:**

1. Bill title contains: "Omnibus", "Consolidated Appropriations", "Continuing Resolution", "Minibus".
2. Bill has 3+ CRS subject tags spanning unrelated policy areas.
3. Bill summary exceeds 2,000 words (proxy for complexity).
4. LLM classification returns `confidence < 0.5` on primary topic assignment.

**Procedural flag — apply if ANY of these are true:**

1. ProPublica vote `question` field contains: "On the Cloture Motion", "On the Motion to Table", "On the Motion to Recommit", "On Agreeing to the Resolution".
2. No `bill` object in the vote response (pure procedural action).
3. Vote description references parliamentary procedure rather than policy substance.

### Implementation

Add two boolean columns to the votes dataframe:

```python
votes_df["is_omnibus"] = False
votes_df["is_procedural"] = False

# Omnibus detection
omnibus_keywords = ["omnibus", "consolidated appropriations", "continuing resolution", "minibus"]
votes_df.loc[
    votes_df["bill_title"].str.lower().str.contains("|".join(omnibus_keywords), na=False),
    "is_omnibus"
] = True

# Procedural detection
procedural_keywords = ["cloture", "motion to table", "motion to recommit", "agreeing to the resolution"]
votes_df.loc[
    votes_df["vote_description"].str.lower().str.contains("|".join(procedural_keywords), na=False),
    "is_procedural"
] = True
```

### Impact on Scoring

- Compute alignment scores **three ways**: (a) all votes, (b) excluding procedural, (c) excluding both omnibus and procedural.
- Report all three. If results diverge significantly, that itself is a finding worth highlighting.

---

## How-To: Alignment Scoring (M4, Day 2)

### Conceptual Framework

For each topic, compare two independent signals:

- **Voting signal:** How often does the politician vote in favour of bills on this topic?
- **Statement signal:** How often does the politician express support for this topic in public statements?

If the signals agree, the politician is consistent. If they diverge, there is a potential contradiction.

### Step 1: Compute Voting Support Rate Per Topic

```python
def compute_vote_support_rate(votes_df, topic):
    topic_votes = votes_df[
        (votes_df["primary_topic"] == topic) &
        (~votes_df["is_procedural"])
    ]

    if len(topic_votes) < 3:
        return {"rate": None, "n": len(topic_votes), "flag": "insufficient_data"}

    # A "Yes" on an "expands" bill = support
    # A "No" on a "restricts" bill = also support (opposing restriction)
    support_votes = topic_votes[
        ((topic_votes["vote_position"] == "Yes") & (topic_votes["bill_direction"] == "expands")) |
        ((topic_votes["vote_position"] == "No") & (topic_votes["bill_direction"] == "restricts"))
    ]

    rate = len(support_votes) / len(topic_votes)
    return {"rate": rate, "n": len(topic_votes), "flag": None}
```

### Step 2: Compute Statement Support Rate Per Topic

```python
def compute_statement_support_rate(claims_df, topic):
    topic_claims = claims_df[claims_df["topic"] == topic]

    if len(topic_claims) < 2:
        return {"rate": None, "n": len(topic_claims), "flag": "insufficient_data"}

    support_claims = topic_claims[topic_claims["stance"] == "supports"]
    rate = len(support_claims) / len(topic_claims)
    return {"rate": rate, "n": len(topic_claims), "flag": None}
```

### Step 3: Compute Alignment Score

```python
def compute_alignment(vote_rate, statement_rate):
    if vote_rate is None or statement_rate is None:
        return {"score": None, "flag": "insufficient_data"}

    score = 1 - abs(vote_rate - statement_rate)
    return {"score": round(score, 3), "flag": None}
```

- **Score = 1.0** — perfect alignment (e.g., votes 90% supportive, statements 90% supportive).
- **Score = 0.0** — complete contradiction (e.g., votes 100% supportive, statements 0% supportive).
- **Suggested threshold:** `score < 0.5` = flagged misalignment.

### Step 4: Build the Per-Topic Summary Table

```
topic | vote_support_rate | vote_n | statement_support_rate | statement_n | alignment_score | flags
```

### Interpreting Results

- **High alignment, high N** → strong evidence of consistency.
- **High alignment, low N** → may be consistent, but not enough data to be confident.
- **Low alignment, high N** → strong evidence of contradiction — investigate further.
- **Low alignment, low N** → possible contradiction, but could be noise.

---

## How-To: Disagreement Detection & Explanation Generation (M3 + M2, Day 2)

### Step 1: Identify Contradictions (M4 → M2)

From the alignment scoring output, extract all cases where `alignment_score < 0.5` AND `vote_n >= 3` AND `statement_n >= 2`. These are the high-confidence contradictions.

Additionally, find **individual vote-statement pairs** that directly contradict:

```python
def find_direct_contradictions(votes_df, claims_df):
    contradictions = []
    for _, vote in votes_df.iterrows():
        topic = vote["primary_topic"]
        vote_direction = "supports" if (
            (vote["vote_position"] == "Yes" and vote["bill_direction"] == "expands") or
            (vote["vote_position"] == "No" and vote["bill_direction"] == "restricts")
        ) else "opposes"

        # Find claims on same topic with opposite stance
        opposing_claims = claims_df[
            (claims_df["topic"] == topic) &
            (claims_df["stance"] != vote_direction) &
            (claims_df["stance"] != "neutral")
        ]

        for _, claim in opposing_claims.iterrows():
            contradictions.append({
                "topic": topic,
                "bill_id": vote["bill_id"],
                "bill_title": vote["bill_title"],
                "vote_position": vote["vote_position"],
                "vote_date": vote["vote_date"],
                "claim_text": claim["claim_text"],
                "claim_date": claim["date"],
                "claim_source": claim["source"],
            })

    return pd.DataFrame(contradictions)
```

### Step 2: Generate Explanations (M3)

For each flagged contradiction, use the LLM to propose possible reasons:

```
You are a political analyst. A politician's vote appears to contradict their
public statement. Propose 2-3 plausible explanations for this discrepancy.

BILL:
Title: {bill_title}
Summary: {bill_summary}
Politician's vote: {vote_position}
Vote date: {vote_date}

POLITICIAN'S STATEMENT:
"{claim_text}"
Statement date: {claim_date}
Source: {claim_source}

Respond in JSON only, no other text:
{
  "explanations": [
    {
      "type": "<party_line | omnibus_rider | strategic_vote | evolved_position | nuance_lost | other>",
      "explanation": "<2-3 sentence explanation>",
      "plausibility": "high" | "medium" | "low"
    }
  ]
}
```

### Explanation Types

| Type | Meaning |
|---|---|
| `party_line` | Voted with party despite personal disagreement |
| `omnibus_rider` | Opposed a specific provision but voted for the overall package (or vice versa) |
| `strategic_vote` | Voted tactically (e.g., to send bill to committee, to force a veto) |
| `evolved_position` | Position changed between statement date and vote date |
| `nuance_lost` | Statement and vote address different sub-aspects of the same topic |
| `other` | None of the above — explanation provided in free text |

### Step 3: Populate the Disagreement Register (M2)

Final schema:

```
topic | alignment_score | bill_id | bill_title | vote_position | vote_date | claim_text | claim_date | explanation_type | explanation_text | plausibility
```

Sort by `alignment_score` ascending (worst contradictions first). This register becomes the source material for M2's case study deep-dives in the presentation.

---

## How-To: Manual Validation Process (M2, Day 2 + Day 3)

### Purpose

LLM outputs are probabilistic. Manual validation quantifies error rates and catches systematic misclassifications before they corrupt the final results.

### Day 2: Stance Extraction Validation

**Sample:** 20–25 randomly selected claim extractions (stratified across topics to ensure coverage).

**Process per claim:**

| Check | Question | Pass/Fail Criteria |
|---|---|---|
| Topic accuracy | Does the assigned topic match the claim content? | Must match your independent assessment |
| Stance accuracy | Is supports/opposes/neutral correct? | Must match your independent assessment |
| Strength calibration | Is strong/moderate/weak reasonable? | Acceptable if within one level of your assessment |
| Claim segmentation | Is this a single coherent claim, or was it over/under-segmented? | Must be one clear position on one topic |

**Record results in a validation log:**

```
claim_id | topic_correct | stance_correct | strength_correct | segmentation_ok | notes
```

**Compute accuracy:** `(correct on all checks) / total`. Target: ≥ 80%.

**If < 80%:** Categorize error patterns → feed to M3 for prompt revision. Common issues:

- Topic confusion between related categories (e.g., "Economy" vs "Infrastructure") → refine taxonomy descriptions.
- Stance reversed on negatively-framed statements → add negative-framing examples to prompt.
- Over-segmentation of compound sentences → adjust segmentation prompt rules.

### Day 3: Contradiction Audit

**Sample:** Top 10 flagged contradictions from the disagreement register.

**Process per contradiction:**

1. **Read the original bill summary.** Does the bill classification (topic + direction) make sense?
2. **Read the original statement.** Does the claim extraction and stance assignment make sense?
3. **Assess the contradiction.** Is this a genuine contradiction, or a classification error?
4. **Evaluate the LLM explanation.** Is the proposed explanation plausible?

**Tag each as:**

- ✅ **True positive** — genuine contradiction, correctly identified.
- ❌ **False positive** — apparent contradiction caused by misclassification.
- ⚠️ **Debatable** — reasonable people could disagree.

**Report the precision:** `true_positives / (true_positives + false_positives)`. This number goes into the presentation's limitations section.

---

## How-To: Sensitivity Testing (M3 + M4, Day 3)

### Purpose

Determine whether the results are robust or fragile — i.e., do small changes in parameters significantly alter the findings?

### Tests to Run

**Test 1: Alignment Threshold Sensitivity (M4)**

Vary the "misalignment" threshold and count how many topics are flagged:

```python
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for t in thresholds:
    flagged = [topic for topic, score in alignment_scores.items() if score < t]
    print(f"Threshold {t}: {len(flagged)} topics flagged — {flagged}")
```

If results change dramatically between 0.4 and 0.5, the findings are threshold-sensitive — note this as a limitation.

**Test 2: Omnibus Inclusion/Exclusion (M4)**

Compare alignment scores computed three ways:

1. All votes included.
2. Procedural votes excluded.
3. Both procedural and omnibus excluded.

If a topic's alignment score swings by > 0.2 between variants, the omnibus bills are materially impacting results. Report the range.

**Test 3: Topic Granularity (M4)**

Merge the two most related topics (e.g., "Economy & Taxation" + "Infrastructure") and recompute. If the merged score differs significantly from individual scores, the taxonomy granularity matters.

**Test 4: LLM Confidence Threshold (M3)**

Rerun scoring using only LLM classifications with `confidence >= 0.7` (instead of all). Compare to baseline. If results change substantially, low-confidence classifications are introducing noise.

**Test 5: Temporal Stability (M4)**

Split the data into two time periods (e.g., first half and second half of the legislative session). Compute alignment per period. If a topic shows high alignment in one period and low in the other, the politician's position may have shifted — this is a finding, not an error.

### Reporting

Create a sensitivity summary table:

```
test | parameter_varied | baseline_result | varied_result | delta | interpretation
```

Include this in the methodology section of the presentation. If all tests show stable results, that strengthens the findings. If results are fragile, be transparent about which parameters matter.

---

## How-To: Building the Visualizations (M5, Day 1–2)

### Recommended Stack

- **matplotlib / seaborn** — heatmaps, bar charts. Simple, reliable.
- **plotly** — interactive timeline charts. Good for exploration.
- **pandas built-in plotting** — quick iteration during development.

### Chart 1: Topic Alignment Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

# alignment_df has columns: topic, alignment_score, vote_n, statement_n
pivot = alignment_df.set_index("topic")["alignment_score"]

fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(
    pivot.values.reshape(-1, 1),
    yticklabels=pivot.index,
    xticklabels=["Alignment"],
    annot=True, fmt=".2f",
    cmap="RdYlGn",       # Red = low alignment, Green = high
    vmin=0, vmax=1,
    linewidths=0.5,
    ax=ax
)
ax.set_title("Claims vs. Votes: Topic Alignment Scores")
plt.tight_layout()
plt.savefig("outputs/alignment_heatmap.png", dpi=150)
```

**Design notes:** Sort topics by alignment score (worst at top) so contradictions are immediately visible.

### Chart 2: Timeline View

Plot votes (as markers) and statements (as markers with different shape) on a shared timeline, faceted by topic.

```python
import plotly.express as px

# combined_df columns: date, topic, event_type ("vote"/"statement"), direction ("supports"/"opposes")
fig = px.scatter(
    combined_df,
    x="date", y="topic",
    color="direction",
    symbol="event_type",
    color_discrete_map={"supports": "green", "opposes": "red"},
    title="Votes & Statements Over Time by Topic"
)
fig.write_html("outputs/timeline.html")
```

**Design notes:** This chart reveals temporal patterns — e.g., a politician who changed position after an election, or who makes supportive statements before voting against.

### Chart 3: Contradiction Spotlight

A structured table/card layout showing the top 5 most significant contradictions. For each:

- Topic name + alignment score
- Bill title + vote + date
- Statement excerpt + date
- LLM-generated explanation

This is best presented as a formatted markdown table or a styled HTML card, not a traditional chart.

### Chart 4: Summary Statistics

A simple dashboard-style summary:

- Overall weighted alignment score (weighted by `vote_n + statement_n`)
- Total bills analysed / total statements analysed
- Most consistent topic (highest alignment)
- Most contradictory topic (lowest alignment)
- Number of topics with insufficient data

---

## How-To: End-to-End Pipeline Rerun (M1, Day 3)

### Purpose

Verify that the entire pipeline — from raw API fetch to final alignment scores — is reproducible and produces consistent results.

### Rerun Protocol

**Step 1: Fresh fetch.**

Delete or rename the `data/raw/` directory. Rerun the data acquisition pipeline from scratch. Compare new raw data against cached Day 1 data:

```python
import json, os

def compare_raw_files(old_dir, new_dir):
    differences = []
    for filename in os.listdir(old_dir):
        old = json.load(open(f"{old_dir}/{filename}"))
        new = json.load(open(f"{new_dir}/{filename}"))
        if old != new:
            differences.append(filename)
    return differences
```

If differences exist, log them. Minor differences (e.g., updated bill summaries) are expected. New votes appearing is a sign the data is live and changing — note the snapshot date.

**Step 2: Reprocess.**

Run the full pipeline using the *original* cached data (not the fresh fetch) to confirm deterministic output:

```bash
python pipeline.py --member-id S001191 --use-cache --output-dir data/rerun/
```

Compare `data/rerun/alignment_scores.json` against `data/processed/alignment_scores.json`. They should be identical if LLM temperature = 0.

**Step 3: Parameterization test.**

If time permits, swap the member ID for a different politician and run the pipeline. This tests:

- API handling for different data volumes.
- Taxonomy applicability across politicians.
- Edge cases in scoring (e.g., a politician with very few votes on certain topics).

### What to Fix

- **Hardcoded paths or IDs** — replace with config parameters.
- **Missing error handling** — API timeouts, empty responses, malformed JSON.
- **Undocumented dependencies** — ensure `requirements.txt` is complete.
