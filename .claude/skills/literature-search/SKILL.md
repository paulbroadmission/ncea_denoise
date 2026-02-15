---
name: literature-search
description: >
  Systematic academic literature search methodology. Provides search query
  templates, paper extraction formats, SOTA table construction, and BibTeX
  verification. Use when searching for papers, building literature reviews,
  or identifying SOTA baselines.
---

# Systematic Literature Search

## Search Query Strategy

For a given research topic, formulate queries in 4 categories:

1. **Direct**: `"{topic}" "{method}" {year range}`
2. **Methodological**: `"{technique}" "{application domain}"`
3. **Benchmark**: `"{dataset name}" benchmark state-of-the-art`
4. **Survey**: `survey OR review "{topic}" {recent year}`

Search priority: arXiv → IEEE Xplore → Google Scholar → Semantic Scholar

## Paper Extraction Template

For each relevant paper:

```yaml
- title: "Exact Paper Title"
  authors: "First Author et al."
  venue: "IEEE TIM"
  year: 2024
  method_name: "ShortMethodName"
  key_contribution: "One sentence"
  datasets:
    - name: "CWRU"
      metrics: { accuracy: 0.985, f1: 0.972 }
  limitations: "What it doesn't handle"
  code_url: "github.com/... or null"
  doi: "10.1109/..."
  bibtex_key: "Author2024Keyword"
```

## SOTA Summary Table Format

```markdown
| Rank | Method | Venue | Year | Acc (%) | F1 (%) | Notes |
|------|--------|-------|------|---------|--------|-------|
| 1    | ...    | ...   | ...  | ...     | ...    | Current SOTA |
| 2    | ...    | ...   | ...  | ...     | ...    | |
```

## Citation Verification Rules

Every reference MUST be verified:
1. Search for the exact title to confirm it exists
2. Confirm authors and venue match
3. Confirm year matches
4. Extract DOI if available

**NEVER fabricate citations.** If unsure, mark as `[UNVERIFIED]` and flag.
