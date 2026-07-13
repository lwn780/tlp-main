# MC Dropout Uncertainty Results

## Setup

- Dataset: platinum-8272, full 2308
- Model: TLP + MC Dropout
- Checkpoint: `runs/tlp_mc_dropout_2308/tlp_model_19.pkl`
- MC samples: 8
- Dropout: 0.1
- Platform: llvm

## Ranking Results

| Method | top-1 | top-5 | top-10 | top-20 |
| --- | ---: | ---: | ---: | ---: |
| TLP baseline | 0.8737 | 0.9522 | 0.9612 | 0.9719 |
| Dropout single eval | 0.8385 | 0.9241 | 0.9560 | 0.9690 |
| MC Dropout mean, lambda=0 | 0.8492 | 0.9378 | 0.9542 | 0.9677 |
| MC penalty, lambda=0.1 | 0.8492 | 0.9376 | 0.9542 | 0.9681 |
| MC penalty, lambda=0.3 | 0.8450 | 0.9372 | 0.9543 | 0.9681 |
| MC penalty, lambda=0.5 | 0.8449 | 0.9359 | 0.9544 | 0.9669 |
| MC penalty, lambda=1.0 | 0.8447 | 0.9363 | 0.9545 | 0.9687 |

## Uncertainty Reliability

Pearson correlation between uncertainty and absolute prediction error: 0.296971369630091

| Group | Avg uncertainty | Avg abs error | Count |
| --- | ---: | ---: | ---: |
| 0 lowest | 0.226060 | 3.725182 | 80855 |
| 1 | 0.332293 | 4.118557 | 80855 |
| 2 | 0.422748 | 4.391543 | 80855 |
| 3 | 0.537151 | 4.759072 | 80855 |
| 4 highest | 0.863398 | 5.477030 | 80855 |

## Current Interpretation

MC Dropout does not improve top-k schedule ranking in the current simple setting. However, its estimated uncertainty is positively correlated with prediction error.

The uncertainty group analysis shows a monotonic increase in absolute error from low-uncertainty schedules to high-uncertainty schedules. This suggests that uncertainty is useful as a reliability signal, even though the current linear penalty ranking rule is not yet strong enough to improve top-k schedule selection.
