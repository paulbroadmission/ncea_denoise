---
name: game-theory
description: >
  Backward induction and game tree analysis for research strategy selection.
  Provides dimension extraction templates, strategy card formats, sensitivity
  analysis framework, and expected value calculations. Use when evaluating
  which research approach combination has the highest probability of
  exceeding SOTA, or when building a strategy matrix.
---

# Game-Theoretic Research Strategy Analysis

## Dimension Extraction Template

From the literature knowledge base, extract independent axes of variation:

```yaml
dimensions:
  D1_representation:
    name: "Feature Representation"
    options:
      - id: d1a
        name: "Raw time-domain"
        known_in: ["Paper1", "Paper2"]     # who already used this
      - id: d1b
        name: "Frequency-domain (FFT/STFT)"
        known_in: ["Paper3"]
    # ... more options

  D2_architecture:
    name: "Core Architecture"
    options: [...]

  # Typically 4-7 dimensions, 3-5 options each
```

**Independence rule**: Changing one dimension must NOT force changes in another.
**Mutual exclusivity**: Options within a dimension are mutually exclusive.

## Strategy Card Template

For each viable combination:

```yaml
strategy_id: S-{number}
combination: { D1: d1c, D2: d2d, D3: d3b, D4: d4b, D5: d5b }

scores:
  feasibility: X/10          # Can we actually build this?
  novelty: X/10              # Is this new enough for the target venue?
  sota_potential: X/10        # Probability of beating SOTA?
  theoretical_soundness: X/10 # Is the math clean?

backward_induction:
  target_outcome: "Metric > threshold on Dataset"
  required_properties:
    - property: "Must capture multi-scale features"
      provided_by: "D1=d1c (CWT)"
    - property: "Must be robust to noise"
      provided_by: "D3=d3b (Lipschitz)"
  critical_success_factors:
    - "Parameter X must be tuned correctly"

expected_value:
  improvement_over_sota: "+X.X%"
  confidence: "LOW | MEDIUM | HIGH"
```

## Backward Induction Procedure

1. **Define terminal payoffs**: For each strategy, estimate P(beat SOTA) and E[improvement]
2. **Identify chance nodes**: Training stability, data quality, hyperparameter sensitivity
3. **Assign probabilities**: Based on literature evidence and domain knowledge
4. **Propagate backwards**: E[value] = Σ p(chance) × payoff(decision)
5. **Select**: Strategy with highest risk-adjusted E[value]

**Risk adjustment**: `risk_adjusted = E[value] × (1 - variance_penalty)`

## Sensitivity Analysis Template

For top 3 strategies, evaluate each critical parameter:

```
Parameter         Base    Range        Impact    Notes
────────────────────────────────────────────────────
[param_name]      [val]   [low, high]  HIGH/MED  [why it matters]
```

**Robustness verdict**:
- All critical params right → X% chance of beating SOTA
- 2/3 right → Y% chance
- 1/3 right → Z% chance

## Probability Calibration Rules

- NEVER assign P > 0.90 to any single strategy (overconfidence kills research)
- NEVER assign P < 0.05 unless the approach is fundamentally flawed
- Use literature reproduction rates as anchors (~60-70% of ML papers reproduce)
- Downweight novel combinations (less evidence) vs well-tested ones
