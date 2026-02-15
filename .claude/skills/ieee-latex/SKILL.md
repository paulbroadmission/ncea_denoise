---
name: ieee-latex
description: >
  Write and compile IEEE-format LaTeX papers. Provides notation conventions,
  section templates, compilation scripts, and BibTeX management. Use when
  writing academic papers, formatting equations, or compiling LaTeX.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# IEEE LaTeX Paper Writing

## Notation Conventions

Maintain consistent notation throughout the paper:

| Type | Convention | Example |
|------|-----------|---------|
| Vectors | lowercase bold | $\mathbf{x}$, $\mathbf{h}$ |
| Matrices | uppercase bold | $\mathbf{W}$, $\mathbf{A}$ |
| Scalars | italic | $\alpha$, $\beta$, $N$ |
| Sets | calligraphic | $\mathcal{X}$, $\mathcal{D}$ |
| Operators | roman | $\mathrm{ReLU}$, $\mathrm{softmax}$ |
| Norms | double bars | $\|\mathbf{x}\|_2$ |
| Expectations | blackboard | $\mathbb{E}[\cdot]$ |

**Rule**: Define every symbol on first use.

## Required Packages

```latex
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{color}
\usepackage{multirow}
\usepackage{balance}
```

## Section Checklist

Each section has mandatory elements:

- **Abstract** (150–250 words): Problem → Method → Key result → Significance
- **Introduction**: Motivation → Limitations of prior work → 3 contributions → Paper organization
- **Related Work**: Organized by methodology categories, not chronologically
- **Proposed Method**: Notation table → Overview figure → Derivations → Pseudocode → Complexity
- **Experimental Setup**: Datasets (with statistics table) → Metrics → Baselines → Implementation details
- **Results**: Main comparison table → Ablation → Visualization → Statistical significance
- **Conclusion**: Contributions → Findings → Limitations (honest) → Future work

## Implementation Details Section — Consistency Rule

Every value in Section IV MUST exactly match `workspace/src/config.py`. Use this comment pattern:

```latex
% CONFIG-SYNC: learning_rate = 1e-3
The learning rate is set to $\eta = 10^{-3}$.
```

The code-audit skill grep for `CONFIG-SYNC` comments to verify matches.

## Compilation

Use the script at `scripts/compile.sh` or run:

```bash
cd workspace/paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## BibTeX Rules

- Every entry MUST have a DOI if available
- Every entry MUST be a verified, real publication
- Use consistent key format: `{FirstAuthorLastName}{Year}{FirstKeyword}`
- Example: `Zhang2024Bearing`

For the paper template, see [templates/ieee_template.tex](templates/ieee_template.tex).
