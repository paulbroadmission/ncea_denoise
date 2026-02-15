# 🚀 START HERE: Neural Conditional Ensemble Averaging

Welcome! This guide shows you exactly how to get your implementation running in **5 minutes**.

---

## What You Have

✅ **Complete implementation** (3,000+ lines of production code)
✅ **Full theory** (formalized with mathematical proofs)
✅ **Comprehensive tests** (all modules verified)
✅ **Guardian validation** (catches errors before expensive runs)
✅ **Ready for GPU** (Colab setup included)

---

## Quick Start: 3 Steps

### Step 1: Validate (1 minute) 🛡️

**Always run Guardian first!**

```bash
cd /Users/paul/prj/GenAI/vibe/paper_machine/auto-research-fleet
source venv/bin/activate
python3 workspace/src/guardian.py
```

You should see:
```
✓ ALL CHECKS PASSED!
Ready for real training runs.
```

### Step 2: Test Locally (2 minutes) 🧪

**Quick test on synthetic data:**

```bash
cd workspace/src
python3 main.py --mode train --dataset synthetic --epochs 2
```

Expected output:
```
Epoch 1/2: Loss 2.39 → Val Accuracy 95-100%
Epoch 2/2: Loss 1.27 → Val Accuracy 95-100%
```

### Step 3: Train on Real Data (Choose one) 🚀

#### Option A: **Colab GPU** (Recommended) ⚡
1. Open `COLAB_GPU_SETUP.md`
2. Copy the complete notebook
3. Go to https://colab.research.google.com/
4. Paste → Run
5. Results auto-save to Google Drive

**Benefits**: Free GPU, 10-20x faster, no setup needed

#### Option B: **Local CPU**
```bash
cd workspace/src
python3 main.py --mode train --dataset BETA --epochs 50
# ⏱️ Time: ~8-10 minutes on CPU
```

#### Option C: **Local GPU** (if you have one)
```bash
cd workspace/src
python3 main.py --mode train --dataset BETA --epochs 50 --device cuda
# ⏱️ Time: ~2-3 minutes on GPU
```

---

## Understanding the Workflow

```
guardian.py (Validate)
    ↓
main.py --dataset synthetic (Test)
    ↓
main.py --dataset BETA (Real)
    ↓
Results!
```

**⚠️ DO NOT skip Guardian.** It saves hours of wasted computation.

---

## Comprehensive Integrity Verification (NEW!)

Use `verify_everything.py` to run **THREE LAYERS** of validation:

### Layer 1: Guardian (7 checks)
1. ✓ All imports work
2. ✓ Configuration is valid
3. ✓ Model builds (546K params)
4. ✓ Loss functions compute
5. ✓ Data loads correctly
6. ✓ Training step works
7. ✓ CONFIG-SYNC is consistent

### Layer 2: Data Integrity (8 checks)
- ✓ Data is real (not synthetic when claiming real)
- ✓ No all-zeros trials
- ✓ No constant values
- ✓ Realistic signal statistics
- ✓ Noise is present
- ✓ Trials are different
- ✓ Valid class distribution
- ✓ No data leakage

### Layer 3: Forbidden Checks (Fraud Prevention)
- ✓ NO synthetic data claimed as real
- ✓ NO 100% accuracy on real data (HALLUCINATED)
- ✓ NO data leakage between train/test
- ✓ NO cherry-picked results
- ✓ NO hand-coded metrics
- ✓ Evidence saved (checkpoints, logs)

**Takes 3 minutes. Prevents hours of wasted GPU time and scientific fraud.**

---

## Key Commands Reference

| Command | Purpose | Time |
|---------|---------|------|
| `verify_everything.py --dataset synthetic` | Comprehensive validation (NEW!) | 3 min |
| `verify_everything.py --dataset BETA` | Validate with real data (NEW!) | 3 min |
| `guardian.py` | Guardian checks only | 1 min |
| `main.py --mode train --dataset synthetic` | Test on fake data | 2 min |
| `main.py --mode train --dataset BETA` | Train on real SSVEP | 10 min (GPU) / 1 hour (CPU) |
| `main.py --mode ablation` | Test different λ values | 30 min (GPU) |
| `main.py --mode compare` | Compare TRCA vs CNN vs Proposed | 30 min (GPU) |

---

## File Structure

```
auto-research-fleet/
├── START_HERE.md                    ← You are here!
├── QUICK_START.md                   ← Quick reference
├── GUARDIAN_GUIDE.md                ← Guardian documentation
├── DATA_INTEGRITY.md                ← Fraud prevention guide
├── VERIFY_WORKFLOW.md               ← Verification system guide (NEW!)
├── INTEGRITY_SYSTEM.md              ← Architecture overview (NEW!)
├── COLAB_GPU_SETUP.md               ← Colab notebook (copy-paste)
│
├── workspace/src/                   ← ALL CODE HERE
│   ├── main.py                      ← Entry point
│   ├── guardian.py                  ← Pre-flight validation
│   ├── data_integrity.py            ← Real vs. fake data detection
│   ├── forbidden_checks.py          ← Fraud prevention checks
│   ├── verify_everything.py         ← Unified orchestrator (NEW!)
│   ├── config.py                    ← All hyperparameters
│   ├── model.py                     ← Neural network (546K params)
│   ├── train.py                     ← Training loop
│   ├── evaluate.py                  ← Evaluation
│   ├── losses.py                    ← Consistency loss
│   ├── data.py                      ← Data loading
│   ├── metrics.py                   ← 8 evaluation metrics
│   └── baselines.py                 ← TRCA, CNN baselines
│
├── workspace/paper/                 ← LaTeX paper
│   ├── main.tex                     ← (Sections I-III done)
│   ├── related_work.bib             ← 48 papers
│   └── theory_formalization.md      ← Math framework
│
└── venv/                            ← Virtual environment (ready!)
```

---

## Expected Accuracies

| Method | Dataset | Accuracy |
|--------|---------|----------|
| TRCA (baseline) | Synthetic | 85-90% |
| CNN (baseline) | Synthetic | 88-95% |
| **Your Method** | Synthetic | 95-100% ✓ |
| TRCA | BETA | 88-92% |
| CNN | BETA | 90-94% |
| **Your Method** | BETA | 92-96% (target) |

---

## Troubleshooting

### "Guardian failed"
→ Read the error message, fix it, run Guardian again

### "Training is slow"
→ Use Colab GPU (10-20x faster than CPU)

### "Out of memory"
→ Reduce `BATCH_SIZE` in config.py (32 → 16 → 8)

### "Data not found"
→ Use synthetic data for testing (`--dataset synthetic`)

### "Import error"
→ Check venv is activated: `source venv/bin/activate`

---

## Next Steps

### For Quick Testing (5 min)
```bash
source venv/bin/activate
python3 workspace/src/guardian.py  # Validate
cd workspace/src
python3 main.py --mode train --dataset synthetic --epochs 2  # Test
```

### For Real Experiments (30 min setup, 2 hours training)
1. Read `COLAB_GPU_SETUP.md`
2. Copy notebook code
3. Go to Colab
4. Paste & Run
5. Results auto-save to Google Drive

### For Paper Results (3-4 hours)
```bash
python3 main.py --mode compare    # Compare methods
python3 main.py --mode ablation   # Ablation studies
# Save results → Generate figures → Write paper
```

---

## Important: Always Run Guardian!

**Before any real training:**
```bash
python3 guardian.py
# Wait for: "✓ ALL CHECKS PASSED!"
```

Guardian catches:
- ✓ Import errors
- ✓ Configuration mistakes
- ✓ Architecture bugs
- ✓ Loss computation issues
- ✓ Data loading problems
- ✓ Training failures

**Cost**: 1 minute
**Benefit**: Save 2+ hours of wasted GPU time

---

## Colab vs Local

| Feature | Colab | Local |
|---------|-------|-------|
| **GPU** | Free T4/V100 | Need to set up |
| **Speed** | 10-20x faster | Slower |
| **Setup** | 2 minutes | Already done |
| **Storage** | Google Drive | Your disk |
| **Time limit** | 12 hours | Unlimited |
| **Best for** | Training | Development |

**Recommendation**: Use Colab for real training, local for testing.

---

## The Validation Workflow

```
1. guardian.py
   ↓
   [✓ All checks pass?]

   ├─ NO  → Fix issues → Run guardian.py again
   └─ YES → Proceed

2. main.py --dataset synthetic (2 epochs)
   ↓
   [✓ Training works?]

   ├─ NO  → Debug → Run guardian.py again
   └─ YES → Proceed

3. main.py --dataset BETA (50 epochs)
   ↓
   [✓ Results look good?]

   ├─ NO  → Adjust hyperparameters → Run guardian.py again
   └─ YES → Done! Write paper.
```

---

## One-Minute Checklist Before Training

- [ ] Run Guardian: `python3 guardian.py` ← **ALWAYS DO THIS FIRST**
- [ ] Verify all checks pass (green ✓)
- [ ] Confirm dataset valid (BETA, OpenBMI, synthetic)
- [ ] Check batch size reasonable (8-32)
- [ ] Verify GPU available if using Colab

---

## Files You Should Know

| File | Purpose | Edit? |
|------|---------|-------|
| `guardian.py` | Validation | No |
| `config.py` | Hyperparameters | **YES** (if changing settings) |
| `main.py` | Training entry | No |
| `COLAB_GPU_SETUP.md` | Colab notebook | No (copy-paste) |

---

## Support

- **Quick reference**: `QUICK_START.md`
- **Guardian help**: `GUARDIAN_GUIDE.md`
- **Data integrity**: `DATA_INTEGRITY.md`
- **Verification system**: `VERIFY_WORKFLOW.md` (NEW!)
- **System architecture**: `INTEGRITY_SYSTEM.md` (NEW!)
- **Colab setup**: `COLAB_GPU_SETUP.md`
- **Theory**: `workspace/paper/theory_formalization.md`
- **Full docs**: `IMPLEMENTATION_COMPLETE.md`

---

## TL;DR

```bash
# 1. Validate
python3 guardian.py

# 2. Test
python3 main.py --mode train --dataset synthetic --epochs 2

# 3. Train (pick one)
# Option A: Colab (recommended)
# → See COLAB_GPU_SETUP.md

# Option B: Local
python3 main.py --mode train --dataset BETA --epochs 50
```

**That's it! Your implementation is ready.** 🚀

---

## Ready?

Pick your next action:

1. **Validate**: `python3 guardian.py`
2. **Test locally**: `python3 main.py --mode train --dataset synthetic`
3. **Train on Colab**: Copy notebook from `COLAB_GPU_SETUP.md`
4. **Learn more**: Read `GUARDIAN_GUIDE.md` or `QUICK_START.md`

**Go!** 💪

