# Stance Extraction System

## Overview

The stance extraction system ([RAG/extract_stances.py](../RAG/extract_stances.py)) converts unstructured political statements into structured, categorized positions using LLM analysis.

**Core Function:** `extract_stance_from_text(text, issue_category)` → `IssueStance`

---

## How It Works

### 1. Input Processing

```python
from RAG.extract_stances import extract_stance_from_text

text = """
Burlison Announces Hearing on Making Housing More Affordable

For far too long, the federal government has imposed unnecessary
regulations that drive up housing costs. We need to remove these
barriers and let the free market work.
"""

stance = extract_stance_from_text(
    text=text,
    issue_category="government_reform",
    source_url="https://..."
)
```

### 2. LLM Prompt Generation

The system builds a structured prompt asking the LLM to:
- Analyze if the text addresses the specified issue
- Determine the politician's position
- Summarize their stance in 1-2 sentences

**Prompt Template:**
```
Analyze this press release and extract the politician's stance on {issue_category}.

Press Release Text:
{text[:3000]}  # Limited to avoid token limits

Determine:
1. Does this text express a clear position on {issue_category}?
2. If yes, what is their stance: supports, opposes, or mixed?
3. Provide a 1-2 sentence summary of their position.

Respond ONLY with valid JSON in this exact format:
{
    "status": "supports" | "opposes" | "mixed" | "no_stance_recorded",
    "summary": "Brief summary of their position (or null if no stance)"
}
```

### 3. Dual LLM Provider Support

The system supports two LLM backends (configured via `.env`):

#### **Option A: Archia API (Claude)**
```bash
LLM_PROVIDER=archia
ARCHIA_API_KEY=your_key
ARCHIA_MODEL_ANTHROPIC=priv-claude-3-5-haiku-20241022
```

**Flow:**
```
extract_stance_from_text()
  ↓
_extract_stance_archia()
  ↓
POST https://api.archia.app/v1/responses
  {
    "model": "priv-claude-3-5-haiku-20241022",
    "input": prompt
  }
  ↓
Parse JSON response → IssueStance
```

#### **Option B: Google Gemini**
```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.0-flash-exp
```

**Flow:**
```
extract_stance_from_text()
  ↓
_extract_stance_gemini()
  ↓
genai.Client.models.generate_content()
  ↓
Parse JSON response → IssueStance
```

### 4. Response Parsing

LLM returns JSON (possibly wrapped in markdown code blocks):

```json
{
  "status": "opposes",
  "summary": "Advocates for reducing federal regulations to lower housing costs"
}
```

The system:
1. Extracts JSON from markdown code blocks if present
2. Parses into Python dict
3. Validates and converts to `IssueStance` Pydantic model

### 5. Output Structure

```python
class IssueStance(BaseModel):
    status: "supports" | "opposes" | "mixed" | "no_stance_recorded"
    summary: Optional[str]  # 1-2 sentence description
    source_url: Optional[str]  # Link to press release
```

**Example:**
```python
IssueStance(
    status="opposes",
    summary="Advocates for reducing federal regulations to lower housing costs",
    source_url="https://burlison.house.gov/..."
)
```

---

## Bill Subject Mapping

**Note:** Bill subject mapping has been moved to [RAG/topic_matching.py](../RAG/topic_matching.py) for better separation of concerns.

The topic matching system connects bill subjects (from Congress.gov) to the 24 standardized issue categories.

### Function: `map_bill_subjects_to_issues()`

**Import:**
```python
from RAG.topic_matching import map_bill_subjects_to_issues
```

**Input:** List of bill subject tags from Congress.gov API
```python
subjects = [
    "Health care costs and insurance",
    "Medicare",
    "Immigration"
]
```

**Output:** List of matched issue categories
```python
["health_care", "immigration"]
```

### Mapping Strategy

**Two-phase matching:**

#### Phase 1: Exact Match
```python
SUBJECT_TO_ISSUE_MAPPING = {
    "Health": "health_care",
    "Immigration": "immigration",
    "Taxation": "tax_reform",
    ...
}

if subject in SUBJECT_TO_ISSUE_MAPPING:
    matched_issues.add(SUBJECT_TO_ISSUE_MAPPING[subject])
```

#### Phase 2: Fuzzy Match
```python
# If no exact match, check if subject contains key phrase
for subject_key, issue in SUBJECT_TO_ISSUE_MAPPING.items():
    if subject_key.lower() in subject.lower():
        matched_issues.add(issue)
        break
```

### Example Mappings

| Bill Subject | Issue Category |
|--------------|----------------|
| "Health care costs and insurance" | `health_care` |
| "Medicare" | `health_care` |
| "Border security and unlawful immigration" | `immigration` |
| "Taxation" | `tax_reform` |
| "Environmental protection" | `environment` |
| "Administrative law and regulatory procedures" | `government_reform` |

### Multi-Category Support

Some subjects map to multiple categories:
```python
"medicare" → ["health_care", "social_security"]
"border security" → ["immigration", "homeland_security"]
```

---

## The 24 Issue Categories

Defined in [dataset/memberOpinions.py](../dataset/memberOpinions.py):

```python
ISSUE_CATEGORIES = [
    "abortion",
    "budget_economy",
    "civil_rights",
    "corporations",
    "crime",
    "drugs",
    "education",
    "energy_oil",
    "environment",
    "families_children",
    "foreign_policy",
    "free_trade",
    "government_reform",
    "gun_control",
    "health_care",
    "homeland_security",
    "immigration",
    "jobs",
    "principles_values",
    "social_security",
    "tax_reform",
    "technology",
    "war_peace",
    "welfare_poverty",
]
```

---

## Error Handling

The system gracefully handles failures:

### LLM API Errors
```python
try:
    stance = extract_stance_from_text(...)
except Exception as e:
    # Returns "unknown" status instead of crashing
    return IssueStance(
        status="unknown",
        summary=None,
        source_url=source_url
    )
```

### Invalid JSON
```python
# If LLM returns malformed JSON
try:
    parsed = json.loads(content)
except json.JSONDecodeError:
    return IssueStance(status="unknown", ...)
```

### API Fallback (Gemini only)
```python
# If HF Inference API fails, fall back to CPU
if api_error:
    print("⚠️ HF Inference API failed")
    print("⚠️ Falling back to CPU for this batch")
    self._fallback_to_cpu()
    return self.embed_texts(texts)  # Retry with CPU
```

---

## Usage Examples

### Example 1: Single Issue Extraction

```python
from RAG.extract_stances import extract_stance_from_text

press_release = """
Burlison Sends Letter to ATF Director Demanding Immediate Corrective Action

I am deeply concerned about the ATF's overreach in regulating firearms.
The Second Amendment is clear, and we must protect law-abiding citizens'
rights to bear arms.
"""

stance = extract_stance_from_text(
    text=press_release,
    issue_category="gun_control",
    source_url="https://burlison.house.gov/..."
)

print(f"Status: {stance.status}")
# Output: Status: opposes
print(f"Summary: {stance.summary}")
# Output: Summary: Opposes ATF regulations as government overreach, supports Second Amendment rights
```

### Example 2: Multi-Issue Extraction

```python
from RAG.extract_stances import extract_stances_from_press_release

stances = extract_stances_from_press_release(
    title="Burlison Announces Hearing on Making Housing More Affordable",
    body_text="...",
    url="https://...",
    target_issues=["government_reform", "budget_economy"]
)

# Returns:
# {
#   "government_reform": IssueStance(status="opposes", ...),
#   "budget_economy": IssueStance(status="no_stance_recorded", ...)
# }
```

### Example 3: Bill Subject Mapping

```python
from RAG.topic_matching import map_bill_subjects_to_issues

bill_subjects = [
    "Health care costs and insurance",
    "Medicare",
    "Medicaid",
    "Social security and elderly assistance"
]

issues = map_bill_subjects_to_issues(bill_subjects)
print(issues)
# Output: ['health_care', 'social_security']
```

---

## Integration with RAG Pipeline

The stance extraction system is designed for **post-retrieval processing**:

```
User Query: "Find contradictions about healthcare"
    ↓
1. Semantic Search (search.py)
   → Retrieve 10 bills + 5 press releases
    ↓
2. Stance Extraction (extract_stances.py) ← THIS STEP
   → Extract stances ONLY from retrieved PRs
   → Extract: 5 press releases × 1-3 relevant issues = ~10 LLM calls
    ↓
3. Topic Matching
   → Map bill subjects to issue categories
   → Find overlapping topics
    ↓
4. Contradiction Detection
   → Compare vote direction vs stance
   → Generate explanation
```

**Why post-retrieval?**
- ✅ Cost-efficient: Only process relevant items
- ✅ Flexible: Can analyze any query without pre-categorization
- ✅ Accurate: Semantic search finds the right context first

---

## Performance Considerations

### Token Limits
- Text truncated to **3,000 characters** to avoid token limits
- Typically ~750 words max per extraction

### LLM Costs (Approximate)

**Per stance extraction:**
- Input: ~1,000 tokens (press release text + prompt)
- Output: ~100 tokens (JSON response)

**Example query cost:**
- 5 press releases × 2 issues = 10 extractions
- ~11,000 tokens total
- Cost: ~$0.01-0.05 depending on provider

### Optimization Tips

1. **Limit target issues:** Only extract for relevant categories
   ```python
   # Good: Only 2 issues
   target_issues=["health_care", "immigration"]

   # Bad: All 24 issues (expensive!)
   target_issues=ISSUE_CATEGORIES
   ```

2. **Use faster models for testing:**
   ```bash
   # Fast & cheap
   GEMINI_MODEL=gemini-2.0-flash-exp

   # Slower but more accurate
   ARCHIA_MODEL_ANTHROPIC=priv-claude-sonnet-4-5-20250929
   ```

3. **Cache results:** Save extracted stances to avoid re-processing

---

## Testing

Run the built-in test:
```bash
python RAG/extract_stances.py
```

**Output:**
```
================================================================================
Stance Extraction Test
================================================================================

Extracting stance on 'government_reform'...

Result:
  Status: opposes
  Summary: Opposes burdensome federal regulations that increase costs
  Source: https://example.com/press-release

================================================================================
Bill Subject Mapping Test
================================================================================

Bill subjects: ['Administrative law and regulatory procedures', 'Health care costs and insurance', 'Immigration', 'Taxation']
Mapped to issues: ['government_reform', 'health_care', 'immigration', 'tax_reform']
```

---

## Configuration Reference

### Environment Variables

```bash
# LLM Provider Selection
LLM_PROVIDER=archia  # or "gemini"

# Archia Configuration
ARCHIA=your_api_key
ARCHIA_BASE_URL=https://api.archia.app/v1
ARCHIA_MODEL_ANTHROPIC=priv-claude-3-5-haiku-20241022

# Gemini Configuration
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-2.0-flash-exp
```

### Function Parameters

```python
extract_stance_from_text(
    text: str,              # Press release or statement text
    issue_category: str,    # One of 24 issue categories
    source_url: Optional[str] = None,  # URL for citation
    max_tokens: int = 500   # Max LLM response tokens
) -> IssueStance
```

```python
map_bill_subjects_to_issues(
    subjects: List[str]     # Bill subject tags from Congress.gov
) -> List[str]              # Matched issue categories
```

---

## Troubleshooting

### "No stance recorded" for relevant text

**Cause:** LLM didn't detect clear position
**Solution:**
- Check if text actually addresses the issue
- Try different issue category
- Increase `max_tokens` for longer analysis

### JSON parsing errors

**Cause:** LLM returned malformed JSON
**Solution:**
- Check LLM model (some models are better at structured output)
- Verify prompt template hasn't been modified
- Inspect raw LLM response for debugging

### API quota errors (Archia)

```
Error: Your organization has exhausted its token quota
```

**Solution:**
- Switch to Gemini: `LLM_PROVIDER=gemini`
- Contact Archia support to increase quota
- Check account status at https://archia.app

### Empty model catalog (Archia)

```
All models (provider -> system_name | type | capabilities):
(empty)
```

**Solution:**
- Verify API key is correct
- Check if account has models provisioned
- Use Gemini as fallback

---

## Design Philosophy

### 1. Post-Retrieval Extraction
Don't pre-extract all stances. Extract on-demand for retrieved items only.

### 2. Structured Output
Use Pydantic models for type safety and validation.

### 3. Dual Provider Support
Archia (Claude) or Gemini for flexibility and redundancy.

### 4. Graceful Degradation
Return "unknown" status on errors instead of crashing.

### 5. Transparent Citations
Always include source URLs for verification.

---

## Next Steps

After stance extraction, the pipeline proceeds to:
1. **Topic Matching** - Align bill subjects with extracted stances
2. **Contradiction Detection** - Compare votes vs stances (not yet implemented)
3. **Report Generation** - Create `ContradictionReport` for UI

See [architecture.md](architecture.md) for the complete pipeline design.
