# Refactoring Checklist: Data Structure Reorganization

## Changes
- Move `data/votes_*.json` → `data/votes/{bioguideId}.json`
- Move `data/members/*.json` → `data/profiles/{bioguideId}.json`

## Scripts Requiring Updates

### Core Pipeline (7 scripts)
1. ✅ **CLAUDE.md** - Updated
2. ✅ **AGENTS.md** - Updated
3. ✅ **README.md** - Updated
4. ✅ **docs/architecture.md** - Updated
5. ⏳ **run_contradiction_pipeline.py** - Needs refactoring
   - Line 50: `MEMBERS_DIR` → `PROFILES_DIR` + add `VOTES_DIR`
   - Line 216: `votes_file = DATA_DIR / f"votes_{bioguide_id}.json"` → `VOTES_DIR / f"{bioguide_id}.json"`

6. ⏳ **dataset/voting_record.py** - Needs refactoring
   - Line 332: Update help text
   - Line 373: `f"data/votes_{bioguide_id}.json"` → `f"data/votes/{bioguide_id}.json"`
   - Add directory creation before writing

7. ⏳ **dataset/fetch_bill_details.py** - Needs refactoring
   - Line 5: Update docstring example

8. ⏳ **dataset/build_member_profile.py** - Needs refactoring
   - Line 18: `MEMBERS_DIR` → `PROFILES_DIR` + add `VOTES_DIR`
   - Line 46-47: Update docstring and path
   - Line 300-301: Update argparse defaults

### RAG System (2 scripts)
9. ⏳ **RAG/load_embeddings.py** - Needs refactoring
   - Line 34: `MEMBERS_DIR` → `PROFILES_DIR`
   - Line 57: Update reference

10. ⏳ **RAG/extract_member_stances.py** - Needs refactoring
    - Line 17: `MEMBERS_DIR` → `PROFILES_DIR`
    - Lines 39, 113: Update references

### Utilities (1 script)
11. ⏳ **dataset/zip_member_votes_bills.py** - Needs refactoring
    - Line 5: `MEMBERS_DIR` → `PROFILES_DIR`
    - Lines 51-56: Update all references

## Detailed Instructions

See [docs/REFACTORING_DATA_STRUCTURE.md](docs/REFACTORING_DATA_STRUCTURE.md) for:
- Line-by-line code changes
- Migration commands
- Testing procedures
- Rollback plan

## Quick Reference

**Search for old patterns:**
```bash
grep -r "votes_.*\.json" --include="*.py" .
grep -r "data/members" --include="*.py" .
grep -r "MEMBERS_DIR" --include="*.py" .
```

**After refactoring, verify no old paths remain:**
```bash
# Should return no results in main code (only in this doc)
grep -r "votes_" --include="*.py" . | grep -v REFACTORING
grep -r "data/members" --include="*.py" . | grep -v REFACTORING
```
