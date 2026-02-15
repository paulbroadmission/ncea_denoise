---
name: results-audit
description: >
  Verify experimental results authenticity, detect fake data, check statistical
  significance, and validate reproducibility. Use after benchmark comparison
  or when verifying if reported results are genuine.
allowed-tools: Read, Grep, Glob, Bash
---

# Results Audit — Authenticity & Statistical Validity

## Automated Red Flag Scan

```bash
# Quick anomaly detection
python3 -c "
import json, sys

results_file = 'workspace/results/iteration_001/test_results.json'  # adjust iteration
try:
    with open(results_file) as f:
        r = json.load(f)

    flags = []

    # Check for perfect metrics
    for k, v in r.items():
        if isinstance(v, float) and v >= 1.0:
            flags.append(f'SUSPICIOUS: {k} = {v} (perfect score)')
        if isinstance(v, float) and v == 0.0:
            flags.append(f'SUSPICIOUS: {k} = {v} (zero)')

    # Check seed is recorded
    if 'seed' not in r:
        flags.append('MISSING: random seed not recorded')

    if flags:
        print('🚩 RED FLAGS:')
        for f in flags:
            print(f'  - {f}')
    else:
        print('✅ No obvious red flags')
except Exception as e:
    print(f'❌ Cannot read results: {e}')
"
```

## Verification Checklist

### 1. Training Log Integrity

- [ ] `training_log.json` exists and is complete
- [ ] Loss curve is monotonically decreasing (overall trend)
- [ ] No sudden jumps that suggest training restart without logging
- [ ] Number of epochs matches config
- [ ] Timestamps are sequential (not fabricated)

### 2. Results Plausibility

Read expected range from `workspace/logs/strategy_matrix.json`:
- [ ] Result within backward-induction expected range?
- [ ] If result > expected + 5%: investigate data leak or bug
- [ ] If result < expected - 10%: strategy may be failing
- [ ] Variance across seeds is realistic (typically 0.5-3% for classification)

### 3. Cross-Consistency

- [ ] `test_results.json` numbers match final epoch in `training_log.json`
- [ ] Numbers in `comparison_table.tex` match `comparison_results.json`
- [ ] Baseline numbers match their cited source (paper table/figure number)

### 4. Statistical Significance

For each "our method vs baseline" comparison:
- [ ] Paired t-test or Wilcoxon computed (p < 0.05)
- [ ] Multiple seeds used (minimum 3, recommended 5)
- [ ] Mean AND standard deviation reported
- [ ] No cherry-picked seeds

### 5. Reproducibility

- [ ] Random seeds listed
- [ ] `requirements.txt` complete
- [ ] Data preprocessing is deterministic
- [ ] Hardware info recorded

## IMMEDIATE FAIL Conditions

Any of these → status = CRITICAL, pipeline halts:
- Results too good (>5% above ALL baselines simultaneously)
- Zero variance across multiple runs
- Missing training logs entirely
- Metrics that are mathematically impossible (e.g., precision > 1.0)
- Baseline numbers don't match their original papers

## Output Format

Write to `workspace/logs/results_audit.json`:

```json
{
  "timestamp": "...",
  "iteration": N,
  "status": "PASS | WARN | CRITICAL",
  "expected_range": [low, high],
  "actual_result": X,
  "plausibility": "PLAUSIBLE | SUSPICIOUS | OUTSIDE_RANGE",
  "statistical_validity": "PASS | FAIL",
  "reproducibility": "CONFIRMED | UNCONFIRMED",
  "red_flags": [],
  "score": X
}
```
