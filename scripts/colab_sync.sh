#!/bin/bash
# ============================================================
# colab_sync.sh — Google Drive sync for Research Agent Fleet
# ============================================================
# Usage:
#   ./scripts/colab_sync.sh push          Push src/ + logs/ to Drive
#   ./scripts/colab_sync.sh pull          Pull results/ from Drive
#   ./scripts/colab_sync.sh status        Check if Colab run completed
#   ./scripts/colab_sync.sh watch         Poll until Colab completes
#   ./scripts/colab_sync.sh init          First-time Drive folder setup
#
# Prerequisites:
#   - rclone configured with Google Drive remote named "gdrive"
#     (run: rclone config → New remote → Google Drive)
#   - OR: Google Drive mounted locally (macOS/Linux)
# ============================================================

set -e

# ---- Configuration ----
DRIVE_REMOTE="gdrive"                          # rclone remote name
DRIVE_PROJECT="research-fleet"                 # folder in Drive root
LOCAL_WORKSPACE="workspace"                    # local workspace path
POLL_INTERVAL=30                               # seconds between status checks

# Auto-detect Drive mount path (macOS Google Drive Desktop)
if [ -d "$HOME/Google Drive/My Drive" ]; then
    DRIVE_MOUNT="$HOME/Google Drive/My Drive/$DRIVE_PROJECT"
    USE_MOUNT=true
elif [ -d "$HOME/GoogleDrive/My Drive" ]; then
    DRIVE_MOUNT="$HOME/GoogleDrive/My Drive/$DRIVE_PROJECT"
    USE_MOUNT=true
elif [ -d "/Volumes/GoogleDrive/My Drive" ]; then
    DRIVE_MOUNT="/Volumes/GoogleDrive/My Drive/$DRIVE_PROJECT"
    USE_MOUNT=true
else
    USE_MOUNT=false
fi

# Detect iteration from orchestrator_state.json
get_iteration() {
    if [ -f "$LOCAL_WORKSPACE/logs/orchestrator_state.json" ]; then
        python3 -c "import json; print(json.load(open('$LOCAL_WORKSPACE/logs/orchestrator_state.json'))['iteration'])" 2>/dev/null || echo "1"
    else
        echo "1"
    fi
}

ITERATION=$(get_iteration)
ITER_DIR=$(printf "iteration_%03d" "$ITERATION")

# ---- Sync functions ----

sync_push() {
    echo "📤 Pushing to Google Drive..."
    echo "   Iteration: $ITERATION"

    if [ "$USE_MOUNT" = true ]; then
        echo "   Method: Direct mount ($DRIVE_MOUNT)"
        mkdir -p "$DRIVE_MOUNT/src"
        mkdir -p "$DRIVE_MOUNT/baselines"
        mkdir -p "$DRIVE_MOUNT/logs"

        rsync -av --delete "$LOCAL_WORKSPACE/src/" "$DRIVE_MOUNT/src/"
        rsync -av "$LOCAL_WORKSPACE/logs/" "$DRIVE_MOUNT/logs/"

        # Only sync baselines if they have content
        if [ "$(ls -A $LOCAL_WORKSPACE/baselines/ 2>/dev/null)" ]; then
            rsync -av "$LOCAL_WORKSPACE/baselines/" "$DRIVE_MOUNT/baselines/"
        fi
    else
        echo "   Method: rclone ($DRIVE_REMOTE:$DRIVE_PROJECT)"
        rclone sync "$LOCAL_WORKSPACE/src/" "$DRIVE_REMOTE:$DRIVE_PROJECT/src/" --progress
        rclone copy "$LOCAL_WORKSPACE/logs/" "$DRIVE_REMOTE:$DRIVE_PROJECT/logs/" --progress

        if [ "$(ls -A $LOCAL_WORKSPACE/baselines/ 2>/dev/null)" ]; then
            rclone sync "$LOCAL_WORKSPACE/baselines/" "$DRIVE_REMOTE:$DRIVE_PROJECT/baselines/" --progress
        fi
    fi

    echo ""
    echo "✅ Push complete. Now:"
    echo "   1. Open Colab notebook: colab/research_fleet_runner.ipynb"
    echo "   2. Set DRIVE_PROJECT_ROOT = \"$DRIVE_PROJECT\""
    echo "   3. Run all cells"
    echo "   4. When done, run: ./scripts/colab_sync.sh pull"
}

sync_pull() {
    echo "📥 Pulling results from Google Drive..."
    echo "   Iteration: $ITERATION ($ITER_DIR)"

    local dest="$LOCAL_WORKSPACE/results/$ITER_DIR"
    mkdir -p "$dest"

    if [ "$USE_MOUNT" = true ]; then
        local src="$DRIVE_MOUNT/results/$ITER_DIR"
        if [ -d "$src" ]; then
            rsync -av "$src/" "$dest/"
            echo "✅ Results pulled to $dest"
        else
            echo "❌ No results found at: $src"
            echo "   Has the Colab notebook finished running?"
            return 1
        fi

        # Pull baseline results too
        if [ -d "$DRIVE_MOUNT/baselines/results" ]; then
            rsync -av "$DRIVE_MOUNT/baselines/results/" "$LOCAL_WORKSPACE/baselines/results/"
            echo "✅ Baseline results pulled"
        fi
    else
        rclone copy "$DRIVE_REMOTE:$DRIVE_PROJECT/results/$ITER_DIR/" "$dest/" --progress
        rclone copy "$DRIVE_REMOTE:$DRIVE_PROJECT/baselines/results/" "$LOCAL_WORKSPACE/baselines/results/" --progress 2>/dev/null || true
        echo "✅ Results pulled to $dest"
    fi

    # Show results summary
    if [ -f "$dest/test_results.json" ]; then
        echo ""
        echo "📊 Results:"
        python3 -c "
import json
with open('$dest/test_results.json') as f:
    r = json.load(f)
for k, v in r.items():
    if isinstance(v, float):
        print(f'   {k}: {v:.4f}')
    else:
        print(f'   {k}: {v}')
"
    fi
}

check_status() {
    echo "🔍 Checking Colab completion status..."
    echo "   Iteration: $ITERATION ($ITER_DIR)"

    local marker=""
    if [ "$USE_MOUNT" = true ]; then
        marker="$DRIVE_MOUNT/results/$ITER_DIR/_colab_complete.json"
    else
        # Download just the marker file
        local tmp="/tmp/colab_check_$$"
        mkdir -p "$tmp"
        rclone copy "$DRIVE_REMOTE:$DRIVE_PROJECT/results/$ITER_DIR/_colab_complete.json" "$tmp/" 2>/dev/null || true
        marker="$tmp/_colab_complete.json"
    fi

    if [ -f "$marker" ]; then
        echo "✅ Colab run COMPLETE"
        python3 -c "
import json
with open('$marker') as f:
    m = json.load(f)
print(f'   GPU: {m.get(\"gpu\", \"unknown\")}')
print(f'   Files: {m.get(\"files_synced\", \"?\")}')
"
        return 0
    else
        echo "⏳ Colab still running (or not started)"
        return 1
    fi
}

watch_completion() {
    echo "👀 Watching for Colab completion (poll every ${POLL_INTERVAL}s)..."
    echo "   Press Ctrl+C to stop"
    echo ""

    while true; do
        if check_status 2>/dev/null; then
            echo ""
            echo "🎉 Colab completed! Pulling results..."
            sync_pull
            return 0
        fi
        echo "   $(date '+%H:%M:%S') — Still waiting..."
        sleep "$POLL_INTERVAL"
    done
}

init_drive() {
    echo "🔧 Initializing Google Drive folder structure..."

    if [ "$USE_MOUNT" = true ]; then
        mkdir -p "$DRIVE_MOUNT/src"
        mkdir -p "$DRIVE_MOUNT/baselines"
        mkdir -p "$DRIVE_MOUNT/results"
        mkdir -p "$DRIVE_MOUNT/logs"
        echo "✅ Created: $DRIVE_MOUNT/{src,baselines,results,logs}"
    else
        echo "Creating folders via rclone..."
        rclone mkdir "$DRIVE_REMOTE:$DRIVE_PROJECT/src"
        rclone mkdir "$DRIVE_REMOTE:$DRIVE_PROJECT/baselines"
        rclone mkdir "$DRIVE_REMOTE:$DRIVE_PROJECT/results"
        rclone mkdir "$DRIVE_REMOTE:$DRIVE_PROJECT/logs"
        echo "✅ Created: $DRIVE_REMOTE:$DRIVE_PROJECT/{src,baselines,results,logs}"
    fi

    # Also upload the Colab notebook
    if [ -f "colab/research_fleet_runner.ipynb" ]; then
        if [ "$USE_MOUNT" = true ]; then
            cp "colab/research_fleet_runner.ipynb" "$DRIVE_MOUNT/"
            echo "✅ Notebook copied to Drive"
        else
            rclone copy "colab/research_fleet_runner.ipynb" "$DRIVE_REMOTE:$DRIVE_PROJECT/"
            echo "✅ Notebook uploaded to Drive"
        fi
    fi

    echo ""
    echo "Done! Open Google Drive → $DRIVE_PROJECT → research_fleet_runner.ipynb"
    echo "Then 'Open with' → Google Colaboratory"
}

# ---- Main ----

case "${1:-help}" in
    push)   sync_push ;;
    pull)   sync_pull ;;
    status) check_status ;;
    watch)  watch_completion ;;
    init)   init_drive ;;
    *)
        echo "Usage: $0 {init|push|pull|status|watch}"
        echo ""
        echo "  init    First-time Drive folder setup + upload notebook"
        echo "  push    Push src/ + logs/ to Drive (before Colab run)"
        echo "  pull    Pull results/ from Drive (after Colab run)"
        echo "  status  Check if Colab run completed"
        echo "  watch   Poll until Colab completes, then auto-pull"
        ;;
esac
