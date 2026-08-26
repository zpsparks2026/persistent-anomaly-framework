# Persistent Anomaly Framework (PAF)

**Author:** Zach Sparks  
**Affiliation:** Sparks Solutions LLC / Technological Leadership, Arizona State University

## Current public preprint

**v17.1 - final public-release edition (July 2026)**

[Download the current paper (PDF)](PAF_v17_1_FINAL_Sparks_2026.pdf)

The Persistent Anomaly Framework is a proposed methodological architecture for testing whether human behavioral response streams exhibit reproducible dependence on concealed, independently randomized targets. It is designed to be falsifiable and mechanism-agnostic.

The current confirmatory architecture uses:

- target-conditioned held-out predictive improvement;
- a target-independent null model whose covariates contain no candidate-key information;
- full-pipeline legal-key randomization inference;
- full-path repeated-look control;
- independent replication before any persistent designation;
- fresh, non-reusable HSM-generated session secrets committed before each unpredictable public-beacon pulse and released only after that session's behavioral log is cryptographically frozen;
- deterministic, auditable target derivation with rejection sampling for arbitrary option counts.

The paper reports **no empirical evidence that anomalous cognition, retrocausality, or any particular mechanism exists**. It specifies a protocol for testing those hypotheses under explicit controls.

## Repository structure

```text
├── paf.tex                              # Canonical LaTeX source for the current paper
├── PAF_v17_1_FINAL_Sparks_2026.pdf      # Current compiled public preprint
├── experiment.py                        # Legacy synthetic pilot from the pre-v17 architecture
├── results.json                         # Legacy pilot output
├── requirements.txt                     # Dependencies for the legacy pilot
└── README.md
```

## Important note on the legacy simulation

`experiment.py` and `results.json` are retained for historical reproducibility of an earlier synthetic proof-of-concept. They implement a **pre-v17** screening/Bayesian/classification pipeline and are **not** an implementation or validation of the current v17.1 target-conditioned confirmatory engine. Results from that legacy simulation should not be cited as evidence that the current framework is statistically validated or that anomalous cognition exists.

The next empirical step for the current framework is a preregistered simulation and pilot package implementing the v17.1 legal-key, cross-fitted, full-path randomization design.

## Scientific status

PAF v17.1 is a proposed methodological framework awaiting empirical validation. Any future operational or applied use requires independent replication, calibration, incremental validity against existing methods, and domain-specific ethics review.
