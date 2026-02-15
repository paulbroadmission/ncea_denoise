---
name: code-audit
description: >
  Verify consistency between LaTeX paper and Python implementation.
  Checks hyperparameter matches, architecture alignment, loss function
  correctness, and algorithm step ordering. Use after implementation
  or when checking if code matches the paper.
allowed-tools: Read, Grep, Glob, Bash
---

# Code Audit — LaTeX ↔ Python Consistency

## Quick Scan Script

Run the automated check first:

```bash
# 1. Hyperparameter match via CONFIG-SYNC markers
echo "=== HYPERPARAMETER CHECK ==="
grep "CONFIG-SYNC" workspace/paper/main.tex | while read line; do
    param=$(echo "$line" | grep -oP '\w+ = [\w.e-]+')
    param_name=$(echo "$param" | cut -d= -f1 | xargs)
    param_val=$(echo "$param" | cut -d= -f2 | xargs)
    code_val=$(grep "$param_name" workspace/src/config.py | grep -oP '= .+' | head -1)
    echo "  LaTeX: $param_name = $param_val | Code: $param_name $code_val"
done

# 2. Architecture dimension check
echo "=== ARCHITECTURE CHECK ==="
grep -n "hidden_dim\|input_dim\|output_dim\|num_layers\|num_heads" workspace/src/config.py
grep -n "d_h\|d_{in}\|d_{out}\|L =" workspace/paper/main.tex
```

## Manual Verification Checklist

### 1. Hyperparameters (MUST all match)

| Parameter | LaTeX Location | config.py Variable | Match? |
|-----------|---------------|-------------------|--------|
| Learning rate | Section IV | `learning_rate` | |
| Batch size | Section IV | `batch_size` | |
| Epochs | Section IV | `num_epochs` | |
| Weight decay | Section IV | `weight_decay` | |
| Architecture dims | Section III | various | |

### 2. Loss Function

Read the loss equation in main.tex and compare term-by-term with `loss.py`:
- [ ] Main loss term matches
- [ ] Regularization terms match
- [ ] Weighting coefficients match
- [ ] Reduction method (mean vs sum) matches

### 3. Algorithm Steps

Read Algorithm 1 in main.tex and compare with `train.py`:
- [ ] Step ordering matches
- [ ] Gradient computation matches
- [ ] Special procedures (warmup, scheduling, clipping) match

### 4. Model Architecture

Read Section III and compare with `model.py`:
- [ ] Layer count matches
- [ ] Activation functions match
- [ ] Skip connections / residual structures match
- [ ] Each custom layer has equation reference comment

## Output Format

Write findings to `workspace/logs/code_audit.json`:

```json
{
  "timestamp": "...",
  "iteration": N,
  "status": "PASS | WARN | CRITICAL",
  "mismatches": [
    {
      "parameter": "learning_rate",
      "latex_value": "1e-3",
      "code_value": "1e-4",
      "severity": "CRITICAL"
    }
  ],
  "architecture_match": true,
  "loss_match": true,
  "algorithm_match": true,
  "score": X
}
```

**Any CRITICAL mismatch = immediate pipeline halt.**
