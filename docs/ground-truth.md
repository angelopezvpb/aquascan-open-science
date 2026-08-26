# Ground Truth Protocol

## What is Ground Truth?

Ground Truth in AquaScan is **independent verification data** obtained from
actual borehole drilling, pump tests, and water quality measurements.

Ground Truth is fundamentally different from model predictions or
Candidate Priority Scores. It is the *answer key* against which we
evaluate our predictions.

## Ground Truth Classes

### A — Strong Ground Truth
- Reliable coordinates
- Measured total depth
- Lithology log available
- Water entry depths documented
- Pump test data (flow rate + duration)
- Identifiable documentation source

### B — Good Ground Truth
- Reliable coordinates
- Measured depth
- Some water data (flow OR static level)
- Partial documentation

### C — Weak Ground Truth
- Coordinates present but possibly imprecise
- Depth reported but not verified
- Limited water data
- Documentation hard to trace

### D — Not Suitable for Primary Training
- Missing critical information
- Unverifiable source
- Conflicting information
- Too incomplete for reliable training

## Ground Truth Score

A continuous 0-1 score computed from available evidence:
- Coordinates present: +0.15
- Depth recorded: +0.15
- Lithology log: +0.20
- Water entries documented: +0.15
- Pump test data: +0.15
- Static water level: +0.10
- Water quality: +0.05
- Documentation identifiable: +0.02 (bonus)

## Candidate Priority Score ≠ Ground Truth Score

The **Candidate Priority Score** is used to prioritize public records for
investigation. It is NOT a scientific quality measure. It only answers:
"Which records are worth looking into?"

The **Ground Truth Score** is computed AFTER real evidence is available.
It answers: "How reliable is this ground truth data for training?"

## Data Governance

- Private client Ground Truth is NEVER included in this repository
- All public Ground Truth must be from verifiable public sources
- Synthetic data is labeled as such and excluded from validation