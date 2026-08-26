# Terminology

## Core Concepts

### AquaScan
The open science framework for multimodal hydrogeophysical groundwater
prospecting research.

### Candidate Priority Score
A heuristic score (0-1) used to prioritize public records for investigation.
Based on metadata completeness (has depth? coordinates? flow rate?).
NOT a scientific quality measure. Does NOT measure ground truth quality.

### Campaign
A single geophysical survey event at a site. Has one set of ERT and/or
seismic measurements. Identified by `campaign_id`.

### Site
A geographic location where one or more campaigns may be conducted.
Identified by `site_id`.

### Well / Borehole
A drilled hole that provides Ground Truth data. Identified by `well_id`.

## Data Levels

### L0 — RAW
Immutable original data. Never modified. Preserves source hash and metadata.

### L1 — CALIBRATED / QC
Normalized, quality-controlled data. Deterministic corrections only.

### L2 — GEOPHYSICS
Derived geophysical products (resistivity, Vp, etc.).

### L3 — ML READY
Feature vectors ready for machine learning.

### L4 — GROUND TRUTH
Independent borehole verification data.

## Scores

### Water Score
Probability/score that a favorable hydrogeological target exists (0-1).
Output of the predictive model.

### Evidence Score
Quality of the evidence backing a prediction (0-1).
Separate from Water Score.

### Ground Truth Score
Quality of the ground truth data itself (0-1).
How reliable is our "answer key"?

### Ground Truth Class
A/B/C/D categorization of Ground Truth quality:
- A: Strong (comprehensive, verifiable)
- B: Good (sufficient but incomplete)
- C: Weak (partial, poorly verifiable)
- D: Not suitable for primary training

## Evaluation

### NO CONCLUYENTE
Flag indicating a prediction should not be used. Triggered when
evidence score is below threshold, even if water score is high.

### Abstention
The model's ability to decline making a prediction when evidence is
insufficient. Measured via coverage vs. error curves.

### Geographic Holdout
Split where train and test are geographically distant. Simulates
predicting on unexplored areas.

### Data Leakage
Error where information from the test set influences training.
In geo/hydro data, this happens when points from the same well/site
appear in both train and test.