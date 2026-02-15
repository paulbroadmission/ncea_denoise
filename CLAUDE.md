# Autonomous Research Agent Fleet

## Architecture: Agents + Skills

**Agents** = WHO does it (role, decision logic, deliverables)
**Skills** = HOW to do it (reusable knowledge, templates, scripts)

### Agents (7) — `.claude/agents/`

| Agent | Model | Role | Skills Used |
|-------|-------|------|-------------|
| orchestrator | opus | Team Lead — coordinates, decides | — |
| literature-explorer | sonnet | Find SOTA, datasets, gaps | literature-search |
| strategy-matrix | opus | Backward induction strategy | game-theory |
| theory-writer | opus | Math theory + LaTeX paper | ieee-latex |
| implementer | sonnet | PyTorch from LaTeX | colab-gpu |
| benchmark-comparator | sonnet | Fair SOTA comparison | colab-gpu |
| watchdog | opus | Quality gate (aggregator) | theory-audit, code-audit, results-audit |

### Skills (7) — `.claude/skills/`

| Skill | Purpose | Used By |
|-------|---------|---------|
| ieee-latex | LaTeX conventions, templates, compilation | theory-writer |
| game-theory | Backward induction framework, templates | strategy-matrix |
| literature-search | Search methodology, extraction templates | literature-explorer |
| theory-audit | Mathematical verification checklist | watchdog |
| code-audit | LaTeX↔Code consistency checker | watchdog |
| results-audit | Fake data detection, statistical tests | watchdog |
| colab-gpu | Google Drive sync, Colab execution | implementer, benchmark-comparator |

## Directory Structure

```
.claude/
├── agents/          # 7 lean agent definitions (role + rules only)
└── skills/          # 7 modular skill packages (knowledge + scripts)
    ├── ieee-latex/
    │   ├── SKILL.md
    │   ├── scripts/compile.sh
    │   └── templates/
    ├── game-theory/SKILL.md
    ├── literature-search/SKILL.md
    ├── theory-audit/SKILL.md
    ├── code-audit/SKILL.md
    ├── results-audit/SKILL.md
    └── colab-gpu/SKILL.md

workspace/
├── paper/           # LaTeX paper
├── src/             # Python implementation
├── baselines/       # SOTA baseline implementations
├── results/         # Per-iteration results
└── logs/            # Agent coordination (JSON + markdown)
```

## Iteration Flow

```
Step 1: literature-explorer → SOTA + datasets
Step 2: strategy-matrix → backward induction → best strategy
Step 3: theory-writer → LaTeX paper for recommended strategy
Step 4: watchdog (theory-audit) → GATE
Step 5: implementer → PyTorch code
Step 6: watchdog (code-audit) → GATE
Step 7: benchmark-comparator → experiments
Step 8: watchdog (results-audit + reviewer) → GATE
Step 9: orchestrator → ACCEPT / TUNE / REVISE / PIVOT / RECOMPUTE
```

## Key Design Principles

1. **Agents are lean** — role definition + decision logic only (~50-80 lines each)
2. **Skills are reusable** — can be used by any agent, independently testable
3. **Watchdog delegates** — invokes audit skills, then adds expert judgment
4. **LaTeX is source of truth** — `CONFIG-SYNC` comments enable automated checking
5. **Colab for GPU** — local agents think, Colab computes
6. **Files as communication** — JSON logs are the shared bus between agents
