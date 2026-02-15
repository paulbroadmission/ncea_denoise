---
name: theory-audit
description: >
  Verify mathematical correctness and theoretical soundness of research papers.
  Checks derivation steps, notation consistency, assumption completeness,
  convergence arguments, and novelty assessment. Use after theory formulation
  or when reviewing mathematical content in LaTeX papers.
allowed-tools: Read, Grep, Glob, Bash
---

# Theory Audit

## Verification Checklist

Run these checks sequentially. STOP at the first CRITICAL failure.

### 1. Notation Consistency Scan

```bash
# Extract all math symbols from LaTeX and check for redefinitions
grep -n '\\mathbf\|\\mathcal\|\\alpha\|\\beta\|\\lambda\|\\eta' workspace/paper/main.tex
```

- [ ] Every symbol defined on first use
- [ ] No symbol used with two different meanings
- [ ] Vectors/matrices/scalars follow convention (see ieee-latex skill)

### 2. Derivation Completeness

For each equation block in the paper:
- [ ] No skipped steps (can you get from Eq.(n) to Eq.(n+1)?)
- [ ] Each step is mathematically valid
- [ ] Assumptions required for each step are stated
- [ ] Edge cases acknowledged (division by zero, empty sets, etc.)

### 3. Theorem/Lemma Verification

For each theorem or lemma:
- [ ] All conditions are explicitly stated
- [ ] Conditions are sufficient (not just necessary)
- [ ] Proof is complete (no "it can be shown that...")
- [ ] Connection to the main method is clear

### 4. Convergence & Stability

- [ ] Loss function is bounded below
- [ ] Gradient exists (differentiability stated)
- [ ] Learning rate conditions stated (if relevant)
- [ ] Any Lipschitz/smoothness assumptions are justified
- [ ] Known failure modes acknowledged

### 5. Novelty Cross-Check

Compare against `workspace/logs/literature_kb.json`:
- [ ] Not a trivial re-combination of methods A + B
- [ ] Clear differentiation from closest existing work
- [ ] Contribution statement is honest and accurate

## Output Format

Write findings to `workspace/logs/theory_audit.json`:

```json
{
  "timestamp": "...",
  "iteration": N,
  "status": "PASS | WARN | CRITICAL",
  "notation_issues": [],
  "derivation_gaps": [],
  "theorem_issues": [],
  "convergence_concerns": [],
  "novelty_assessment": "CONFIRMED | NEEDS_STRENGTHENING | INSUFFICIENT",
  "score": X  // 1-10
}
```
