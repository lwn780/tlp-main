# TLP Baseline: platinum-8272

## Experiment

- Date: 2026-06-16
- Repository on server: `/root/mycode/tlp-main`
- Dataset source: `scripts/dataset/measure_records/platinum-8272`
- Dataset size: full 2308 measurement-record files
- Platform: `llvm`
- Model: original TLP baseline
- Training dataset: `tlp_dataset_platinum_8272_2308_train_and_val.pkl`
- Test dataset: `tlp_dataset_platinum_8272_2308_test.pkl`
- Training run: `runs/tlp_baseline_2308_retry`
- Epochs: 20
- CUDA argument: `cuda:0`

## Main Result: Epoch 19

| Metric | Score |
| --- | ---: |
| Average top-1 | 0.8737 |
| Average top-5 | 0.9522 |
| Average top-10 | 0.9612 |
| Average top-20 | 0.9719 |

## Checkpoint Comparison

| Checkpoint | Average top-1 | Average top-5 | Average top-10 | Average top-20 |
| --- | ---: | ---: | ---: | ---: |
| Epoch 17 | 0.8682 | 0.9407 | 0.9574 | 0.9747 |
| Epoch 19 | 0.8737 | 0.9522 | 0.9612 | 0.9719 |

## Notes

- `top-k` means the best true schedule among the model's top-k predicted schedules.
- A score closer to 1 is better.
- The held-out networks in `tlp_eval.py` are ResNet-50, MobileNetV2, ResNeXt-50, BERT-base, and BERT-tiny.
