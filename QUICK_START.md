# Quick Start Guide

## Pre-Flight Validation (REQUIRED) ✅

**Always run Guardian before any training!**

```bash
cd /Users/paul/prj/GenAI/vibe/paper_machine/auto-research-fleet
source venv/bin/activate
python3 workspace/src/guardian.py
```

✓ Should see: `✓ ALL CHECKS PASSED!`

If any checks fail, fix them before proceeding.

---

## 5-Minute Setup

### Local Setup (CPU)
```bash
cd /Users/paul/prj/GenAI/vibe/paper_machine/auto-research-fleet
source venv/bin/activate

# STEP 1: Run Guardian first!
python3 workspace/src/guardian.py

# STEP 2: Train
cd workspace/src
python3 main.py --mode train --dataset synthetic --epochs 10
```

### Colab Setup (GPU) - Recommended
1. Go to https://colab.research.google.com/
2. Create new notebook
3. Copy code from `COLAB_GPU_SETUP.md`
4. Run cells
5. Results auto-save to Google Drive

---

## What is Guardian?

Guardian is a **pre-flight validation script** that checks:

1. ✅ All imports work (torch, numpy, etc.)
2. ✅ Configuration is valid (epochs, batch size, etc.)
3. ✅ Model architecture is correct (546K parameters)
4. ✅ Loss functions compute properly
5. ✅ Data loading works
6. ✅ A full training step completes successfully
7. ✅ CONFIG-SYNC consistency (theory matches code)

**Why?** Guardian catches 90% of issues BEFORE expensive GPU runs, saving hours of wasted computation.

**Run this before every real training run on Colab or GPU.**

---

## Running Experiments

### Test the Implementation (2 min)
```bash
source venv/bin/activate
cd workspace/src
python3 main.py --mode train --dataset synthetic --epochs 2 --batch-size 4
```

### Train Full Model (Local, ~30 min on CPU)
```bash
source venv/bin/activate
cd workspace/src
python3 main.py --mode train --dataset BETA --epochs 50
```

### Run Ablation Studies (Local, ~2 hours on CPU)
```bash
source venv/bin/activate
cd workspace/src
python3 main.py --mode ablation --dataset BETA --epochs 30
```

### Compare Methods (Local, ~3 hours on CPU)
```bash
source venv/bin/activate
cd workspace/src
python3 main.py --mode compare --dataset BETA --epochs 30
```

---

## Key Files

### Training & Evaluation
- `workspace/src/main.py` — Main entry point
- `workspace/src/train.py` — Training loop
- `workspace/src/evaluate.py` — Evaluation metrics
- `workspace/src/model.py` — Neural network architectures

### Configuration
- `workspace/src/config.py` — All hyperparameters
- Edit this file to change:
  - Learning rate: `LEARNING_RATE = 1e-3`
  - Batch size: `BATCH_SIZE = 32`
  - Lambda consistency: `LAMBDA_CONSISTENCY = 0.1`
  - Number of epochs: `NUM_EPOCHS = 100`

### Output Locations
- **Checkpoints**: `workspace/checkpoints/best_model.pt`
- **Logs**: `workspace/logs/history_*.json`
- **Results**: `workspace/results/`

---

## Common Commands

```python
# Train with L_consistency only (default)
python3 main.py --mode train --lambda-consistency 0.1

# Train without consistency (CNN baseline)
python3 main.py --mode train --lambda-consistency 0.0

# Train with different lambda values
for lambda in 0.01 0.1 0.5; do
    python3 main.py --mode train --lambda-consistency $lambda
done

# Evaluate on test set
python3 main.py --mode test --checkpoint best_model.pt

# Run ablation
python3 main.py --mode ablation

# Compare all methods
python3 main.py --mode compare
```

---

## GPU vs CPU

### Enable GPU (Colab)
1. Runtime → Change runtime type
2. Hardware accelerator → GPU (T4 or V100)
3. Click Save

### Use GPU Locally (if CUDA available)
```python
# In config.py, change:
DEVICE = "cuda"  # instead of "cpu"
```

### Check GPU
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## Expected Results

### Synthetic Data
- Accuracy: 95-100% (easy task)
- Time: 30 seconds per epoch

### Real SSVEP Data (BETA)
- TRCA: 85-90%
- CNN baseline: 88-95%
- Proposed (L_consistency): 90-96% (expected)
- Time: 2-5 minutes per epoch

---

## Debugging

### Issue: Model not improving
**Solution**: Increase `LAMBDA_CONSISTENCY` to strengthen regularization
```python
LAMBDA_CONSISTENCY = 0.5  # or higher
```

### Issue: Training too slow
**Solution**: Reduce batch size or dataset
```python
BATCH_SIZE = 16  # instead of 32
DEBUG = True  # Use smaller dataset
```

### Issue: GPU out of memory
**Solution**: Reduce batch size
```python
BATCH_SIZE = 8
```

### Issue: Can't load BETA dataset
**Solution**: Use synthetic data for now
```python
dataset_name = "synthetic"
```

---

## File Structure

```
workspace/
├── src/
│   ├── config.py          ← Edit hyperparameters here
│   ├── main.py            ← Entry point
│   ├── model.py           ← Neural network
│   ├── train.py           ← Training loop
│   ├── evaluate.py        ← Evaluation
│   ├── losses.py          ← Loss functions
│   ├── data.py            ← Data loading
│   ├── metrics.py         ← Metrics
│   ├── baselines.py       ← TRCA, CNN baselines
│   └── requirements.txt
│
├── checkpoints/           ← Saved models
├── logs/                  ← Training logs
├── results/               ← Experimental results
└── paper/                 ← Paper (LaTeX)
    ├── main.tex
    ├── related_work.bib
    └── theory_formalization.md
```

---

## Next Steps

1. **Run quick test**: `python3 main.py --mode train --dataset synthetic --epochs 2`
2. **Train on real data**: `python3 main.py --mode train --dataset BETA --epochs 50`
3. **Run ablations**: `python3 main.py --mode ablation`
4. **Compare methods**: `python3 main.py --mode compare`
5. **Write results** into `workspace/paper/main.tex` Sections IV-VI

---

## Getting Help

- **Code documentation**: See docstrings in each `.py` file
- **Theoretical background**: Read `theory_formalization.md`
- **Literature**: See `RESEARCH_SOURCES.md`
- **Colab setup**: See `COLAB_GPU_SETUP.md`

---

## Pro Tips

1. **Start with synthetic data** to verify everything works
2. **Use Colab GPU** for faster training (10-20x speedup)
3. **Save results to Google Drive** for easy access
4. **Run ablations** to understand which components help
5. **Compare with baselines** (TRCA, CNN, Li et al. 2024) for validation

---

**Ready to train!** 🚀

```bash
cd workspace/src
source ../../venv/bin/activate
python3 main.py --mode train --dataset synthetic --epochs 2
```
