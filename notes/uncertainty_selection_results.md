# Uncertainty-Aware Selection Results

## Setup

- Dataset: platinum-8272, full 2308
- Model: TLP + MC Dropout
- Checkpoint: `runs/tlp_mc_dropout_2308/tlp_model_19.pkl`
- MC samples: 8
- Dropout: 0.1
- Platform: llvm

## Main Strategy Results

| Strategy | top-1 | top-5 | top-10 | top-20 |
| --- | ---: | ---: | ---: | ---: |
| mean | 0.842217 | 0.948306 | 0.954543 | 0.966392 |
| global_penalty_0.1 | 0.842217 | 0.948330 | 0.954216 | 0.966426 |
| global_penalty_0.5 | 0.844448 | 0.947315 | 0.954181 | 0.964381 |
| global_penalty_1.0 | 0.844745 | 0.948241 | 0.953073 | 0.963422 |
| pool5_low_unc | 0.845861 | 0.948306 | 0.954543 | 0.966392 |
| pool10_low_unc | 0.786106 | 0.923822 | 0.954543 | 0.966392 |
| pool20_low_unc | 0.711356 | 0.863161 | 0.935200 | 0.966392 |
| pool50_low_unc | 0.461580 | 0.823916 | 0.889773 | 0.943803 |
| drop_high_uncertain_0.2 | 0.843818 | 0.948430 | 0.952507 | 0.962906 |

## Observation

The best observed uncertainty-aware top-1 result is `pool5_low_unc`:

| Method | top-1 |
| --- | ---: |
| MC mean | 0.842217 |
| pool5_low_unc | 0.845861 |

This suggests that uncertainty can help as a local reranking signal inside a small high-quality candidate pool.

However, larger low-uncertainty pools degrade ranking quality. This means uncertainty should not dominate global ranking. Low uncertainty may select schedules that are stable but not fast.

## Current Interpretation

A promising direction is two-stage uncertainty-aware reranking:

1. Use prediction mean to select a small candidate pool.
2. Use uncertainty only inside that candidate pool for risk-aware reranking.

The current result is preliminary. It does not yet outperform the original TLP baseline, but it shows that uncertainty can affect schedule selection in a useful direction when used locally.
