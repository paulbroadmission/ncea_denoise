#!/bin/bash
# ============================================================
# Autonomous Research Agent Fleet — Setup Script
# ============================================================

set -e

echo "============================================"
echo "  Autonomous Research Agent Fleet - Setup"
echo "============================================"
echo ""

# --- 1. Check Claude Code ---
echo "[1/5] Checking Claude Code installation..."
if command -v claude &> /dev/null; then
    CLAUDE_VERSION=$(claude --version 2>/dev/null || echo "unknown")
    echo "  ✅ Claude Code found (version: $CLAUDE_VERSION)"
else
    echo "  ❌ Claude Code not found."
    echo "     Install: npm install -g @anthropic-ai/claude-code"
    echo "     Or visit: https://code.claude.com/docs/en/overview"
    exit 1
fi

# --- 2. Enable Agent Teams ---
echo ""
echo "[2/5] Configuring Agent Teams..."

SETTINGS_FILE="$HOME/.claude/settings.json"
mkdir -p "$HOME/.claude"

if [ -f "$SETTINGS_FILE" ]; then
    # Check if already enabled
    if grep -q "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS" "$SETTINGS_FILE" 2>/dev/null; then
        echo "  ✅ Agent Teams already enabled in settings.json"
    else
        echo "  ⚠️  settings.json exists but Agent Teams not enabled."
        echo "     Please add manually or run:"
        echo '     export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1'
    fi
else
    echo "  Creating settings.json with Agent Teams enabled..."
    cat > "$SETTINGS_FILE" << 'EOF'
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "permissions": {
    "allow": [],
    "deny": []
  }
}
EOF
    echo "  ✅ settings.json created"
fi

# Also export for current session
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
echo "  ✅ Environment variable set for current session"

# --- 3. Check directory structure ---
echo ""
echo "[3/5] Verifying project structure..."

REQUIRED_DIRS=(
    ".claude/agents"
    "workspace/paper/figures"
    "workspace/src"
    "workspace/baselines"
    "workspace/results"
    "workspace/logs"
)

ALL_OK=true
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/"
    else
        echo "  ❌ Missing: $dir/"
        mkdir -p "$dir"
        echo "     → Created"
        ALL_OK=false
    fi
done

# --- 4. Verify agent files ---
echo ""
echo "[4/5] Verifying agent definitions..."

AGENTS=(
    ".claude/agents/orchestrator.md:🎯 Orchestrator (Team Lead)"
    ".claude/agents/literature-explorer.md:📚 Literature Explorer"
    ".claude/agents/strategy-matrix.md:🎲 Strategy Matrix (Backward Induction)"
    ".claude/agents/theory-writer.md:✍️  Theory & LaTeX Writer"
    ".claude/agents/implementer.md:⚙️  Implementation Agent"
    ".claude/agents/benchmark-comparator.md:📊 Benchmark Comparator"
    ".claude/agents/watchdog.md:🛡️  Watchdog / Guardian"
)

for agent_entry in "${AGENTS[@]}"; do
    IFS=':' read -r agent_file agent_name <<< "$agent_entry"
    if [ -f "$agent_file" ]; then
        echo "  ✅ $agent_name"
    else
        echo "  ❌ Missing: $agent_file ($agent_name)"
        ALL_OK=false
    fi
done

# --- 4b. Verify skill files ---
echo ""
echo "[4b/5] Verifying skills..."

SKILLS=(
    ".claude/skills/ieee-latex/SKILL.md:📄 IEEE LaTeX"
    ".claude/skills/game-theory/SKILL.md:🎲 Game Theory"
    ".claude/skills/literature-search/SKILL.md:🔍 Literature Search"
    ".claude/skills/theory-audit/SKILL.md:✅ Theory Audit"
    ".claude/skills/code-audit/SKILL.md:✅ Code Audit"
    ".claude/skills/results-audit/SKILL.md:✅ Results Audit"
    ".claude/skills/colab-gpu/SKILL.md:☁️  Colab GPU"
)

for skill_entry in "${SKILLS[@]}"; do
    IFS=':' read -r skill_file skill_name <<< "$skill_entry"
    if [ -f "$skill_file" ]; then
        echo "  ✅ $skill_name"
    else
        echo "  ❌ Missing: $skill_file ($skill_name)"
        ALL_OK=false
    fi
done

# --- 5. Check optional dependencies ---
echo ""
echo "[5/5] Checking optional dependencies..."

# LaTeX
if command -v pdflatex &> /dev/null; then
    echo "  ✅ pdflatex (LaTeX compilation)"
else
    echo "  ⚠️  pdflatex not found (install texlive for LaTeX compilation)"
fi

# Python
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version)
    echo "  ✅ $PY_VERSION"
else
    echo "  ⚠️  Python 3 not found"
fi

# PyTorch
python3 -c "import torch; print(f'  ✅ PyTorch {torch.__version__}')" 2>/dev/null || echo "  ⚠️  PyTorch not installed (pip install torch)"

# Git
if command -v git &> /dev/null; then
    echo "  ✅ Git $(git --version | cut -d' ' -f3)"
else
    echo "  ⚠️  Git not found"
fi

# rclone (for Colab sync)
if command -v rclone &> /dev/null; then
    echo "  ✅ rclone $(rclone version | head -1 | cut -d' ' -f2) (for Colab sync)"
else
    # Check for Google Drive Desktop mount instead
    if [ -d "$HOME/Google Drive/My Drive" ] || [ -d "$HOME/GoogleDrive/My Drive" ]; then
        echo "  ✅ Google Drive Desktop detected (for Colab sync)"
    else
        echo "  ⚠️  Neither rclone nor Google Drive Desktop found"
        echo "     Install one for Colab GPU integration:"
        echo "     - rclone: curl https://rclone.org/install.sh | sudo bash"
        echo "     - Or install Google Drive Desktop"
    fi
fi

# Make colab_sync.sh executable
if [ -f "scripts/colab_sync.sh" ]; then
    chmod +x scripts/colab_sync.sh
    echo "  ✅ colab_sync.sh ready"
fi

# --- Summary ---
echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  To start a research project, run:"
echo ""
echo "    cd $(pwd)"
echo ""
echo "    # If using Colab GPU (recommended):"
echo "    ./scripts/colab_sync.sh init"
echo ""
echo "    # Then start Claude Code:"
echo "    claude"
echo ""
echo "  Then tell the orchestrator your research idea:"
echo ""
echo '    > Use the orchestrator agent to start research on:'
echo '      Domain: [your domain]'
echo '      Inspiration: [your key idea]'
echo '      Target venue: [e.g., IEEE TIM]'
echo '      Datasets: [benchmark datasets]'
echo '      Success criteria: [e.g., exceed SOTA by 2%]'
echo ""
echo "============================================"
