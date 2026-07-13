# Next Steps: TLP + Uncertainty Estimation

## Goal

Build an uncertainty-aware cost model on top of the TLP baseline.

## Current Baseline

- Model: TLP baseline
- Checkpoint: `runs/tlp_baseline_2308_retry/tlp_model_19.pkl`
- Average top-1: 0.8737
- Average top-5: 0.9522
- Average top-10: 0.9612
- Average top-20: 0.9719

## First Method

Try MC Dropout first.

## Uncertainty-Aware Ranking

Ranking rule:

rank_score = prediction_mean - lambda * prediction_std

## Minimum Experiments

- Baseline TLP
- TLP + MC Dropout
- TLP + uncertainty-aware ranking
