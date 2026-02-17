# Data Structure Refactoring

## Overview

This document tracks the refactoring of data file locations to improve organization:

**Changes:**
1. Move `data/votes_*.json` → `data/votes/{bioguideId}.json`
2. Move `data/members/*.json` → `data/profiles/{bioguideId}.json`

**Rationale:**
- Better organization with dedicated directories
- Clearer separation of intermediate data (votes) from final profiles
- Consistent naming convention (no prefix, just directory structure)

## Scripts Requiring Updates

### High Priority (Core Pipeline)

#### 1. `run_contradiction_pipeline.py`
**Lines to update:** 50, 216

**Current:**
```python
MEMBERS_DIR = DATA_DIR / "members"
votes_file = DATA_DIR / f"votes_{bioguide_id}.json"
```

**New:**
```python
PROFILES_DIR = DATA_DIR / "profiles"
VOTES_DIR = DATA_DIR / "votes"
votes_file = VOTES_DIR / f"{bioguide_id}.json"
```

**Changes needed:**
- Line 50: Change `MEMBERS_DIR = DATA_DIR / "members"` → `PROFILES_DIR = DATA_DIR / "profiles"`
- Line 216: Change `votes_file = DATA_DIR / f"votes_{bioguide_id}.json"` → `votes_file = VOTES_DIR / f"{bioguide_id}.json"`
- Add: `VOTES_DIR = DATA_DIR / "votes"` near line 50

---

#### 2. `dataset/voting_record.py`
**Lines to update:** 332, 373

**Current:**
```python
help="Output file path (default: data/votes_<bioguide_id>.json)"
output_path = Path(f"data/votes_{bioguide_id}.json")
```

**New:**
```python
help="Output file path (default: data/votes/<bioguide_id>.json)"
output_path = Path(f"data/votes/{bioguide_id}.json")
```

**Changes needed:**
- Line 332: Update help text
- Line 373: Change path construction
- Add directory creation: `Path("data/votes").mkdir(parents=True, exist_ok=True)` before writing

---

#### 3. `dataset/fetch_bill_details.py`
**Lines to update:** 5 (docstring example)

**Current:**
```python
python fetch_bill_details.py --from-votes data/votes_B001316.json
```

**New:**
```python
python fetch_bill_details.py --from-votes data/votes/B001316.json
```

**Changes needed:**
- Line 5: Update example in docstring

---

#### 4. `dataset/build_member_profile.py`
**Lines to update:** 18, 46-47, 300-301

**Current:**
```python
MEMBERS_DIR = DATA_DIR / "members"

def load_voting_record(bioguide_id: str) -> List[Dict]:
    """Load voting record from data/votes_{bioguideId}.json."""
    votes_file = DATA_DIR / f"votes_{bioguide_id}.json"
```

**New:**
```python
PROFILES_DIR = DATA_DIR / "profiles"
VOTES_DIR = DATA_DIR / "votes"

def load_voting_record(bioguide_id: str) -> List[Dict]:
    """Load voting record from data/votes/{bioguideId}.json."""
    votes_file = VOTES_DIR / f"{bioguide_id}.json"
```

**Changes needed:**
- Line 18: Rename `MEMBERS_DIR` → `PROFILES_DIR`
- Add: `VOTES_DIR = DATA_DIR / "votes"` near line 18
- Line 46: Update docstring
- Line 47: Change `votes_file = DATA_DIR / f"votes_{bioguide_id}.json"` → `votes_file = VOTES_DIR / f"{bioguide_id}.json"`
- Line 300: Update `default=MEMBERS_DIR` → `default=PROFILES_DIR`
- Line 301: Update help text
- All references to `MEMBERS_DIR` throughout the file → `PROFILES_DIR`

---

### Medium Priority (RAG System)

#### 5. `RAG/load_embeddings.py`
**Lines to update:** 34, 57

**Current:**
```python
MEMBERS_DIR = PROJECT_ROOT / "data" / "members"
profile_path = MEMBERS_DIR / f"{bioguide_id}.json"
```

**New:**
```python
PROFILES_DIR = PROJECT_ROOT / "data" / "profiles"
profile_path = PROFILES_DIR / f"{bioguide_id}.json"
```

**Changes needed:**
- Line 34: Rename `MEMBERS_DIR` → `PROFILES_DIR`
- Line 57: Update variable reference

---

#### 6. `RAG/extract_member_stances.py`
**Lines to update:** 17, 39, 113

**Current:**
```python
MEMBERS_DIR = Path(__file__).resolve().parents[1] / "data" / "members"
profile_path = MEMBERS_DIR / f"{args.bioguide_id}.json"
output_file = MEMBERS_DIR / f"{args.bioguide_id}_stances.json"
```

**New:**
```python
PROFILES_DIR = Path(__file__).resolve().parents[1] / "data" / "profiles"
profile_path = PROFILES_DIR / f"{args.bioguide_id}.json"
output_file = PROFILES_DIR / f"{args.bioguide_id}_stances.json"
```

**Changes needed:**
- Line 17: Rename `MEMBERS_DIR` → `PROFILES_DIR`
- Line 39: Update variable reference
- Line 113: Update variable reference

---

### Low Priority (Utilities)

#### 7. `dataset/zip_member_votes_bills.py`
**Lines to update:** 5, 51-56

**Current:**
```python
MEMBERS_DIR = Path("data/members")
if not MEMBERS_DIR.exists():
    raise FileNotFoundError(f"Directory not found: {MEMBERS_DIR}")
member_files = sorted(MEMBERS_DIR.glob("*.json"))
```

**New:**
```python
PROFILES_DIR = Path("data/profiles")
if not PROFILES_DIR.exists():
    raise FileNotFoundError(f"Directory not found: {PROFILES_DIR}")
member_files = sorted(PROFILES_DIR.glob("*.json"))
```

**Changes needed:**
- Line 5: Rename `MEMBERS_DIR` → `PROFILES_DIR`
- Lines 51-56: Update all references to `MEMBERS_DIR` → `PROFILES_DIR`

---

## Documentation Updates

### Files to update:

1. **`AGENTS.md`** (Line 48-49)
   - `data/members/{bioguideId}.json` → `data/profiles/{bioguideId}.json`
   - `data/votes_{bioguideId}.json` → `data/votes/{bioguideId}.json`

2. **`docs/architecture.md`** (Line 247)
   - `Stores in data/votes/{bioguide_id}.json` (already correct structure, just verify)
   - Update any references to `data/members/` → `data/profiles/`

3. **`README.md`**
   - Line 70-72: Update example paths
   - `data/members/B001316.json` → `data/profiles/B001316.json`

4. **`RAG/README.md`**
   - Update any references to member profiles path

## Migration Steps

### 1. Create new directories
```bash
mkdir -p data/votes
mkdir -p data/profiles
```

### 2. Move existing files
```bash
# Move voting records
mv data/votes_*.json data/votes/ 2>/dev/null || true

# Rename files in votes directory (remove prefix)
cd data/votes
for file in votes_*.json; do
    mv "$file" "${file#votes_}"
done
cd ../..

# Move member profiles
mv data/members/*.json data/profiles/ 2>/dev/null || true
```

### 3. Update .gitignore
```bash
# Add new patterns
echo "data/votes/*.json" >> .gitignore
echo "data/profiles/*.json" >> .gitignore

# Remove old patterns (if they exist)
# data/votes_*.json
# data/members/*.json
```

### 4. Run refactoring
Update all scripts as listed above.

### 5. Test with Burlison test case
```bash
# Test complete pipeline
./run_pipeline.sh B001316 --skip-all

# Verify files exist in new locations
ls -l data/votes/B001316.json
ls -l data/profiles/B001316.json
```

## Verification Checklist

- [ ] All scripts updated with new paths
- [ ] Documentation updated
- [ ] Existing data files migrated
- [ ] .gitignore updated
- [ ] Test pipeline runs successfully
- [ ] Check that no hardcoded old paths remain (`grep -r "votes_.*\.json" --include="*.py"`)
- [ ] Check that no hardcoded old paths remain (`grep -r "data/members" --include="*.py"`)

## Rollback Plan

If issues arise:
```bash
# Move files back
mv data/votes/*.json data/
cd data
for file in *.json; do
    if [[ $file != *_* ]]; then
        mv "$file" "votes_$file"
    fi
done

mv data/profiles/*.json data/members/
```

## Notes

- The shell script `run_pipeline.sh` doesn't directly reference file paths (it calls Python scripts)
- `server/pages/1_Voted_Bills.py` doesn't reference votes or member files directly
- The change is backward-incompatible but improves long-term maintainability
- Consider adding validation in scripts to check both old and new locations during transition period
