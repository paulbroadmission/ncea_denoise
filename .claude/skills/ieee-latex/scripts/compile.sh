#!/bin/bash
# IEEE LaTeX paper compilation script
# Usage: ./compile.sh [workspace/paper]

PAPER_DIR="${1:-workspace/paper}"
cd "$PAPER_DIR" || { echo "❌ Directory not found: $PAPER_DIR"; exit 1; }

echo "📄 Compiling IEEE paper..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

if [ -f main.pdf ]; then
    echo "✅ main.pdf generated ($(du -h main.pdf | cut -f1))"
else
    echo "❌ Compilation failed. Check main.log for errors:"
    grep -A2 "^!" main.log 2>/dev/null || echo "  No log found"
    exit 1
fi
