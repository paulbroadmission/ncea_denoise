#!/bin/bash
# ============================================================
# setup_colab.sh — Complete Colab GPU Mode Setup
# ============================================================
# 統一管理：
# 1. 檢查/安裝 rclone
# 2. rclone Google Drive 授權
# 3. 初始化 Drive 目錄結構
# 4. 更新 orchestrator_state.json
# ============================================================

set -e

echo "============================================================"
echo "  Colab GPU Setup — Complete Configuration"
echo "============================================================"
echo ""

# --- Step 1: Check/Install rclone ---
echo "[1/4] Checking rclone installation..."

if ! command -v rclone &> /dev/null; then
    echo "⚠️  rclone not found. Installing via Homebrew..."

    if command -v brew &> /dev/null; then
        brew install rclone
        echo "✅ rclone installed successfully"
    else
        echo "❌ Homebrew not found. Please install rclone manually:"
        echo "   https://rclone.org/install/"
        exit 1
    fi
else
    RCLONE_VERSION=$(rclone --version | head -1)
    echo "✅ $RCLONE_VERSION"
fi

echo ""

# --- Step 2: Check/Configure Google Drive ---
echo "[2/4] Checking Google Drive configuration..."

CONFIG_DIR="${HOME}/.config/rclone"
CONFIG_FILE="${CONFIG_DIR}/rclone.conf"

if [ -f "$CONFIG_FILE" ] && rclone listremotes 2>/dev/null | grep -q "^gdrive:$"; then
    echo "✅ Google Drive remote 'gdrive' already configured"

    # Test connection
    if rclone lsd gdrive: &>/dev/null; then
        echo "✅ Connected to Google Drive"
        GDRIVE_OK=true
    else
        echo "⚠️  Cannot connect to Google Drive"
        echo "    Trying to reconnect..."
        rclone config reconnect gdrive 2>/dev/null || {
            echo "❌ Reconnection failed. Please authorize manually:"
            echo "    rclone config"
            exit 1
        }
        GDRIVE_OK=true
    fi
else
    echo "⚠️  Google Drive not configured"
    echo ""

    # Interactive setup
    while true; do
        echo "Setup Google Drive authorization? (y/n)"
        read -r response
        case "$response" in
            [yY])
                mkdir -p "$CONFIG_DIR"
                echo ""
                echo "=========================================="
                echo "rclone Google Drive Setup Instructions"
                echo "=========================================="
                echo ""
                echo "When prompted by rclone, follow these steps:"
                echo ""
                echo "  Step 1: When asked 'n/s/q>':"
                echo "    → Enter: n"
                echo ""
                echo "  Step 2: When asked 'name>':"
                echo "    → Enter: gdrive"
                echo ""
                echo "  Step 3: When asked 'Type of storage>':"
                echo "    → Enter: 24 (or type: drive)"
                echo ""
                echo "  Step 4: For all other prompts (client_id, client_secret, etc.):"
                echo "    → Press Enter to use defaults"
                echo ""
                echo "  Step 5: When asked 'Use web browser to auto authenticate?':"
                echo "    → Enter: y"
                echo ""
                echo "  Step 6: Browser will open automatically"
                echo "    → Click 'Allow' to authorize rclone"
                echo "    → Return to terminal and proceed"
                echo ""
                echo "=========================================="
                echo ""

                read -p "Ready? Press Enter to start rclone config..." _
                echo ""

                # Run rclone config interactively
                rclone config

                # Verify setup
                if rclone listremotes 2>/dev/null | grep -q "^gdrive:$"; then
                    echo ""
                    echo "✅ Google Drive configured!"
                    GDRIVE_OK=true
                    break
                else
                    echo ""
                    echo "⚠️  Setup incomplete. Try again? (y/n)"
                    read -r retry
                    if [[ ! "$retry" =~ ^[yY]$ ]]; then
                        echo "⚠️  Skipping Google Drive setup for now"
                        GDRIVE_OK=false
                        break
                    fi
                fi
                ;;
            [nN])
                echo "⚠️  Skipping Google Drive setup"
                echo "You can set it up later with: rclone config"
                GDRIVE_OK=false
                break
                ;;
            *)
                echo "Please enter 'y' or 'n'"
                ;;
        esac
    done
fi

echo ""

# --- Step 3: Initialize Drive directories ---
if [ "$GDRIVE_OK" = true ]; then
    echo "[3/4] Initializing Google Drive project structure..."

    # Create base directory
    rclone mkdir gdrive:research-fleet 2>/dev/null || true

    # Create subdirectories
    for dir in src baselines logs results; do
        rclone mkdir "gdrive:research-fleet/$dir" 2>/dev/null || true
        echo "  ✅ research-fleet/$dir/"
    done
else
    echo "[3/4] Skipping Drive initialization (no connection)"
fi

echo ""

# --- Step 4: Initialize orchestrator_state.json ---
echo "[4/4] Updating orchestrator configuration..."

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LOG_DIR="workspace/logs"

mkdir -p "$LOG_DIR"

cat > "$LOG_DIR/orchestrator_state.json" << EOF
{
  "project_name": "Neural Conditional Ensemble Averaging",
  "research_domain": "SSVEP-BCI",
  "user_inspiration": "Improve SSVEP classification with consistency loss",
  "target_venue": "IEEE TMI / Top Venue",
  "success_criteria": "Exceed SOTA by 2-3%",
  "iteration": 1,
  "max_iterations": 10,
  "phase": "implementation",
  "status": "ready_for_colab",
  "sota_baseline": {
    "method_name": "Li et al. 2024 (TRCA+CNN)",
    "primary_metric_name": "accuracy",
    "primary_metric_value": 0.92,
    "dataset": "BETA"
  },
  "our_best_result": {
    "iteration": 0,
    "primary_metric_value": null,
    "all_metrics": {}
  },
  "gap_to_sota": null,
  "decision_history": [],
  "guardian_reports": [],
  "reviewer_scores": [],
  "created_at": "$TIMESTAMP",
  "last_updated": "$TIMESTAMP"
}
EOF

echo "✅ orchestrator_state.json initialized"
echo ""

# --- Make scripts executable ---
chmod +x scripts/colab_sync.sh

# --- Summary ---
echo "============================================================"
echo "  ✅ Colab Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1️⃣  Verify local code works:"
echo "    python3 workspace/src/main.py --dataset synthetic --epochs 2"
echo ""
echo "2️⃣  Push to Google Drive:"
echo "    ./scripts/colab_sync.sh push"
echo ""
echo "3️⃣  Run Colab notebook:"
echo "    - Open: https://colab.research.google.com"
echo "    - File → Open from GitHub"
echo "    - Search: paulbroadmission/ncea_denoise"
echo "    - Open: colab/COLAB_READY_AGENT_INTEGRATED.ipynb"
echo "    - Runtime → Change runtime type → GPU"
echo "    - Run All"
echo ""
echo "4️⃣  Pull results back:"
echo "    ./scripts/colab_sync.sh pull"
echo ""
echo "For detailed guide, see: MODE2_QUICK_START.md"
echo ""
echo "============================================================"
