# Quick Verify: One-Command Verification

## TL;DR: Single Command

```bash
cd /Users/paul/prj/GenAI/vibe/paper_machine/auto-research-fleet
source venv/bin/activate
python3 workspace/src/verify_everything.py --dataset BETA
```

**Expected Output**: `✓ ALL VERIFICATION PHASES PASSED!`

---

## What It Does in 3 Minutes

```
Your Script
    ↓
Phase 1: Guardian Checks (7)
├─ Imports work?
├─ Config valid?
├─ Model builds?
├─ Loss functions compute?
├─ Data loads?
├─ Training step works?
└─ CONFIG-SYNC present?
    ↓
Phase 2: Data Integrity (8)
├─ Data shape correct?
├─ All-zeros detection
├─ Constant-values detection
├─ Signal statistics realistic?
├─ Noise present?
├─ Trials different?
├─ Class distribution valid?
└─ No data leakage?
    ↓
Phase 3: Forbidden Checks (6)
├─ Synthetic as real? (NO)
├─ 100% accuracy on real? (NO)
├─ Data leakage? (NO)
├─ Cherry-picking? (NO)
├─ Metric mismatch? (NO)
└─ Missing evidence? (NO)
    ↓
DECISION
✓ GO FOR TRAINING  or  ✗ FIX ISSUES
    ↓
JSON Audit Trail Saved
```

---

## Usage Examples

### Local Test with Synthetic Data
```bash
python3 workspace/src/verify_everything.py --dataset synthetic
```

### Before Real Training
```bash
python3 workspace/src/verify_everything.py --dataset BETA
```

### On Colab
```python
import subprocess

result = subprocess.run(
    ["python3", "workspace/src/verify_everything.py", "--dataset", "BETA"],
    cwd="/content/auto-research-fleet",
    capture_output=True,
    text=True
)

print(result.stdout)
assert "ALL VERIFICATION PHASES PASSED" in result.stdout, "Verification failed!"
```

---

## Expected Output

### Success ✅
```
======================================================================
  COMPREHENSIVE INTEGRITY VERIFICATION SYSTEM
======================================================================

Starting full verification for BETA dataset...

======================================================================
  PHASE 1: Guardian Pre-Flight Checks
======================================================================

[1/7] Imports...
  ✓ PASS All modules importable
[2/7] Configuration...
  ✓ PASS Config valid (epochs=100, batch=32, lr=0.001)
[3/7] Model Architecture...
  ✓ PASS Model created (546,128 params)
[4/7] Loss Functions...
  ✓ PASS Loss functions compute
[5/7] Data Loading...
  ✓ PASS Data loading works
[6/7] Training Step...
  ✓ PASS Training step successful
[7/7] CONFIG-SYNC Consistency...
  ✓ PASS CONFIG-SYNC tags found

======================================================================
  PHASE 2: Data Integrity Verification
======================================================================

[Validating BETA data...]

  ✓ PASS Data integrity checks:
    - Shape valid: True
    - All-zeros check: PASS
    - Signal std: 0.987 (valid: >0.01)
    - No data leakage

======================================================================
  PHASE 3: Forbidden Checks (Fraud Prevention)
======================================================================

[1/3] Data Source Verification
  ✓ PASS Data source valid (BETA)
[2/3] Hallucination Detection
  ⓘ INFO: Will check after training results available
[3/3] Data Leakage Prevention
  ✓ PASS No data leakage detected

======================================================================
✓ ALL VERIFICATION PHASES PASSED!
======================================================================

✓ Audit trail saved to verification_audit_trail.json
```

### Failure ❌
```
======================================================================
✗ VERIFICATION FAILED AT PHASE 2 (Data Integrity)
======================================================================

Data integrity check failed:
✗ FAIL Signal std too low (0.000009), likely FAKE

Fix the issue and try again!
```

---

## Audit Trail Output

Every run generates `verification_audit_trail.json`:

```json
{
  "timestamp": "2026-02-15T15:26:43",
  "dataset": "BETA",
  "device": "cuda",
  "phases": {
    "guardian": {"status": "PASS", "checks": 7, "passed": 7},
    "data_integrity": {"status": "PASS", "signal_std": 0.987},
    "forbidden": {"status": "PASS", "checks": ["synthetic_as_real", "data_leakage"]}
  },
  "summary": {
    "passed_checks": 9,
    "failed_checks": 0,
    "status": "PASS"
  }
}
```

---

## What Each Phase Checks

### Phase 1: Guardian (7 checks)
Validates that your **CODE IS CORRECT**

| Check | What | Fail If |
|-------|------|---------|
| Imports | All modules loadable | Missing dependency |
| Config | Hyperparameters valid | Invalid value |
| Model | Builds & runs | Shape mismatch |
| Loss | Computes & differentiates | NaN/Inf values |
| Data | Loads correctly | Shape wrong |
| Training | Forward→Backward→Update works | Gradient error |
| CONFIG-SYNC | Documentation present | Missing tags |

### Phase 2: Data Integrity (8 checks)
Validates that your **DATA IS REAL**

| Check | Detects | Fails If |
|-------|---------|----------|
| Shape | Correct dimensions | Wrong shape |
| All-zeros | Fake data (all 0s) | All zeros |
| Constants | Synthetic data | Constant values |
| Std | Signal variation | std < 0.01 |
| Range | Signal spread | range < 0.1 |
| Correlation | Duplicate trials | corr > 0.99 |
| Noise | Real noise present | noise < 0.001 |
| Classes | Balanced distribution | <5 samples/class |

### Phase 3: Forbidden Checks (6 checks)
Enforces **ZERO-TOLERANCE POLICY**

| Check | Forbids | Fails If |
|-------|---------|----------|
| Synthetic | Claiming synthetic as real | data_source mismatch |
| Leakage | Train/test overlap | Any shared samples |
| Hallucinate | 100% on real data | accuracy == 1.0 |
| Consistency | Metric mismatch | \|Acc - F1\| > 0.10 |
| Cherry-pick | Best-run only | max - mean > 0.05 |
| Evidence | Missing logs/checkpoint | No saved model |

---

## Quick Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| Module not found | Dependency missing | `pip install torch numpy scipy scikit-learn` |
| CUDA not available | GPU not installed | Falls back to CPU automatically |
| Data shape wrong | Corrupted data | Download fresh from BETA/OpenBMI |
| All zeros detected | Fake data | Use real data source |
| Training step fails | Config issue | Check config.py parameters |
| Data leakage detected | Train/test overlap | Fix split logic in data.py |

---

## The Workflow

### Before Local Testing
```bash
python3 verify_everything.py --dataset synthetic
# ✓ PASS → OK to test locally
# ✗ FAIL → Fix code first
```

### Before Real Training
```bash
python3 verify_everything.py --dataset BETA
# ✓ PASS → OK to train on GPU
# ✗ FAIL → Fix issues first
```

### Before Publishing
```bash
python3 verify_everything.py --dataset BETA
cp verification_audit_trail.json results/
# Include with paper: proof of validation
```

---

## Key Points

1. **Run BEFORE every training** — prevents wasted GPU time
2. **Takes 3 minutes** — worth it vs. hours of broken training
3. **Clear output** — easy to understand pass/fail
4. **JSON audit trail** — documents validation conditions
5. **Zero tolerance** — fraud detection built-in
6. **Device fallback** — works on CPU or GPU
7. **No configuration** — works out-of-the-box

---

## Next Steps After Verification Passes

### If Synthetic: Test Locally
```bash
python3 main.py --mode train --dataset synthetic --epochs 2
```

### If BETA: Train on Colab
1. Open `COLAB_GPU_SETUP.md`
2. Copy notebook to Colab
3. Run verification first
4. Then train

### Save Results with Audit Trail
```bash
cp verification_audit_trail.json results/
# Keep with results for reproducibility
```

---

**Remember**: This is your first line of defense. Always run verification before training!

🛡️ One command. Three layers of protection. Zero tolerance for fraud.
