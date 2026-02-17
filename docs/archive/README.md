# Documentation Archive

This directory contains historical documentation from completed refactorings and migrations.

## Contents

### Data Structure Refactoring (Feb 2025)

- **REFACTORING_DATA_STRUCTURE.md** - Detailed guide for the data structure reorganization
- **REFACTORING_CHECKLIST.md** - Completion checklist for the refactoring

**Summary:** Reorganized data files from flat structure to organized directories:
- `data/votes_*.json` → `data/votes/{bioguideId}.json`
- `data/members/*.json` → `data/profiles/{bioguideId}.json`

**Completed:** Feb 17, 2025 (commits: ecfd02f, 15850f6, cd5c7f9)

**Current Structure:** See [CLAUDE.md](../../CLAUDE.md) for up-to-date architecture.
