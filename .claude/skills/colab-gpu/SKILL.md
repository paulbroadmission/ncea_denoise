---
name: colab-gpu
description: >
  Google Colab GPU integration for training and evaluation. Provides
  sync commands, completion detection, and troubleshooting. Use when
  pushing code to Colab, pulling results, or debugging Colab runs.
allowed-tools: Read, Bash, Grep
---

# Colab GPU Integration

## Quick Reference

```bash
# First time setup
./scripts/colab_sync.sh init

# Every iteration cycle
./scripts/colab_sync.sh push      # Push src/ to Drive
# → User runs Colab notebook (Run All)
./scripts/colab_sync.sh watch     # Auto-poll until complete
./scripts/colab_sync.sh pull      # Pull results/ back

# Check status manually
./scripts/colab_sync.sh status
```

## How It Works

```
Local (Claude Code)              Google Drive              Colab (GPU)
─────────────────              ────────────              ───────────
src/ ──push──────────→ research-fleet/src/ ──mount──→ /content/workspace/src/
                                                          ↓
                                                      GPU Training
                                                          ↓
results/ ←──pull───── research-fleet/results/ ←──sync── writes results + _colab_complete.json
```

## Completion Detection

The Colab notebook writes `_colab_complete.json` when finished:

```json
{
  "iteration": 1,
  "status": "complete",
  "gpu": "Tesla T4",
  "files_synced": 5
}
```

`colab_sync.sh watch` polls for this file every 30 seconds.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `push` fails | Drive not mounted/configured | Run `rclone config` or install Google Drive Desktop |
| `status` never completes | Colab disconnected | Reconnect Colab, re-run cells |
| Results missing | Train.py errored | Check Colab output cells for Python errors |
| Wrong iteration | State file stale | Check `orchestrator_state.json` iteration number |

## Local GPU Fallback

If GPU is available locally, skip Colab entirely:

```bash
cd workspace/src && python train.py
```

The rest of the pipeline doesn't care where results came from.
