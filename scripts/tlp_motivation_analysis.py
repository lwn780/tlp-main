"""
TLP Motivation Analysis: Prove that cost model is inaccurate AND doesn't know it.

Produces 3 key analyses:
  1. Prediction error distribution (how often / how badly does TLP get it wrong?)
  2. Per-network performance variation (is it uniformly bad or selectively bad?)
  3. Calibration analysis - ECE (does the model know when it's wrong?)

Usage:
  cd scripts/
  python3 tlp_motivation_analysis.py \
    --test_dataset_name tlp_dataset_platinum_8272_2308_test.pkl \
    --load_name runs/tlp_baseline_2308_retry/tlp_model_19.pkl \
    --mc_dropout_model runs/tlp_mc_dropout_2308/tlp_model_19.pkl \
    --platform llvm \
    --cuda cuda:0 \
    --mc_samples 30 \
    | tee ../notes/motivation_analysis.txt
"""
import argparse
import os
import pickle
import sys
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tlp_train_mc_dropout import (
    AttentionModule,
    SegmentDataLoader,
    set_seed,
)


def load_model(model_file, device):
    with open(model_file, "rb") as f:
        loaded = pickle.load(f)
    model = loaded.module if hasattr(loaded, "module") else loaded
    if not hasattr(model, 'dropout'):
        model.dropout = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predict(model, datas, device, mc_samples=1, use_dropout=False):
    file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = datas
    test_loader = SegmentDataLoader(line_vecs, 4000, False)

    preds_samples = []
    labels_all = None

    with torch.no_grad():
        for _ in range(mc_samples):
            model.eval()
            if use_dropout:
                enable_dropout(model)

            preds_all = []
            labels_epoch = []
            for batch_datas_steps, batch_labels in test_loader:
                batch_datas_steps = batch_datas_steps.to(device)
                preds = model(batch_datas_steps)
                if isinstance(preds, list) and len(preds) > 1:
                    preds = preds[0]
                preds_all.append(preds.detach().cpu())
                labels_epoch.append(batch_labels.detach().cpu())

            preds_samples.append(torch.cat(preds_all, dim=0))
            if labels_all is None:
                labels_all = torch.cat(labels_epoch, dim=0)

    preds_stack = torch.stack(preds_samples, dim=0)
    preds_mean = preds_stack.mean(dim=0).numpy()
    preds_std = preds_stack.std(dim=0, unbiased=False).numpy()
    labels = labels_all.numpy()
    min_latency = test_loader.min_latency.min().numpy()

    return preds_mean, preds_std, labels, min_latency


def main():
    device = args.cuda

    with open(args.test_dataset_name, "rb") as f:
        test_datasets = pickle.load(f)

    pred_a_dataset_dict = {}
    for data in test_datasets:
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        pred_a_dataset_dict[workloadkey] = data

    network_files = [
        ("resnet_50", "dataset/network_info/((resnet_50,[(1,3,224,224)]),%s).task.pkl" % args.platform),
        ("mobilenet_v2", "dataset/network_info/((mobilenet_v2,[(1,3,224,224)]),%s).task.pkl" % args.platform),
        ("resnext_50", "dataset/network_info/((resnext_50,[(1,3,224,224)]),%s).task.pkl" % args.platform),
        ("bert_base", "dataset/network_info/((bert_base,[(1,128)]),%s).task.pkl" % args.platform),
        ("bert_tiny", "dataset/network_info/((bert_tiny,[(1,128)]),%s).task.pkl" % args.platform),
    ]

    # Load baseline model
    model = load_model(args.load_name, device)

    # ================================================================
    # Analysis 1: Prediction Error Distribution
    # ================================================================
    print("=" * 70)
    print("Analysis 1: Prediction Error Distribution")
    print("=" * 70)

    all_preds = []
    all_labels = []
    all_rel_errors = []
    all_wrong_speedup_losses = []
    total_tasks = 0
    top1_wrong = 0

    for net_name, network_file in network_files:
        tasks, task_weights = pickle.load(open(network_file, "rb"))

        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                continue

            datas = pred_a_dataset_dict[task.workload_key]
            preds, _, labels, min_lat = predict(model, datas, device, mc_samples=1)

            total_tasks += 1

            # Check top-1 correctness
            model_best = np.argmax(preds)
            true_best = np.argmax(labels)

            if model_best != true_best:
                top1_wrong += 1
                # speedup loss = how much slower is model's choice vs true best
                speedup_loss = labels[model_best] / max(labels[true_best], 1e-5)
                all_wrong_speedup_losses.append(speedup_loss)

            # Relative errors
            for p, l in zip(preds, labels):
                all_preds.append(p)
                all_labels.append(l)
                rel_err = abs(p - l) / max(abs(l), 1e-5)
                all_rel_errors.append(rel_err)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_rel_errors = np.array(all_rel_errors)
    all_wrong_speedup_losses = np.array(all_wrong_speedup_losses)

    print(f"\nTotal tasks: {total_tasks}")
    print(f"Total samples: {len(all_preds)}")
    print(f"Top-1 wrong count: {top1_wrong} / {total_tasks} ({top1_wrong/total_tasks*100:.1f}%)")

    print(f"\n--- Relative Error Distribution (abs(pred-true)/abs(true)) ---")
    print(f"  Mean:   {np.mean(all_rel_errors):.4f}")
    print(f"  Median: {np.median(all_rel_errors):.4f}")
    print(f"  Std:    {np.std(all_rel_errors):.4f}")
    print(f"  > 10%:  {np.mean(all_rel_errors > 0.10)*100:.1f}%")
    print(f"  > 20%:  {np.mean(all_rel_errors > 0.20)*100:.1f}%")
    print(f"  > 50%:  {np.mean(all_rel_errors > 0.50)*100:.1f}%")
    print(f"  > 100%: {np.mean(all_rel_errors > 1.00)*100:.1f}%")

    print(f"\n--- Top-1 Selection Error Cost ---")
    if len(all_wrong_speedup_losses) > 0:
        print(f"  Wrong count: {len(all_wrong_speedup_losses)}")
        print(f"  Mean speedup loss: {np.mean(all_wrong_speedup_losses):.4f}")
        print(f"  Median speedup loss: {np.median(all_wrong_speedup_losses):.4f}")
        print(f"  Max speedup loss: {np.max(all_wrong_speedup_losses):.4f}")
        print(f"  > 1.2x slower: {np.mean(all_wrong_speedup_losses > 1.2)*100:.1f}%")
        print(f"  > 1.5x slower: {np.mean(all_wrong_speedup_losses > 1.5)*100:.1f}%")
        print(f"  > 2.0x slower: {np.mean(all_wrong_speedup_losses > 2.0)*100:.1f}%")

    # ================================================================
    # Analysis 2: Per-Network Performance Variation
    # ================================================================
    print("\n" + "=" * 70)
    print("Analysis 2: Per-Network Performance Variation")
    print("=" * 70)

    top_ks = [1, 5, 10, 20]
    print(f"\n{'Network':<16s} {'top-1':>8s} {'top-5':>8s} {'top-10':>8s} {'top-20':>8s} {'MAE':>10s} {'#tasks':>8s}")
    print("-" * 70)

    for net_name, network_file in network_files:
        tasks, task_weights = pickle.load(open(network_file, "rb"))
        best_latency = 0
        latencies = [0] * len(top_ks)
        net_errors = []

        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                continue
            datas = pred_a_dataset_dict[task.workload_key]
            preds, _, labels, min_lat = predict(model, datas, device, mc_samples=1)

            order = np.argsort(-preds)
            real_values = labels[order]
            real_latency = min_lat / np.maximum(real_values, 1e-5)
            for i, top_k in enumerate(top_ks):
                latencies[i] += np.min(real_latency[:top_k]) * weight
            best_latency += min_lat * weight
            net_errors.extend(np.abs(preds - labels).tolist())

        scores = [best_latency / lat for lat in latencies]
        mae = np.mean(net_errors)
        print(f"{net_name:<16s} {scores[0]:>8.4f} {scores[1]:>8.4f} {scores[2]:>8.4f} {scores[3]:>8.4f} {mae:>10.4f} {len(tasks):>8d}")

    # ================================================================
    # Analysis 3: Calibration Analysis (ECE)
    # ================================================================
    print("\n" + "=" * 70)
    print("Analysis 3: Calibration Analysis (ECE)")
    print("=" * 70)

    del model
    torch.cuda.empty_cache()

    mc_model = load_model(args.mc_dropout_model, device)

    all_mc_means = []
    all_mc_stds = []
    all_mc_labels = []
    all_mc_abs_errors = []

    for net_name, network_file in network_files:
        tasks, task_weights = pickle.load(open(network_file, "rb"))

        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                continue
            datas = pred_a_dataset_dict[task.workload_key]
            mc_mean, mc_std, labels, min_lat = predict(
                mc_model, datas, device,
                mc_samples=args.mc_samples, use_dropout=True)

            all_mc_means.extend(mc_mean.tolist())
            all_mc_stds.extend(mc_std.tolist())
            all_mc_labels.extend(labels.tolist())
            all_mc_abs_errors.extend(np.abs(mc_mean - labels).tolist())

    all_mc_stds = np.array(all_mc_stds)
    all_mc_abs_errors = np.array(all_mc_abs_errors)
    all_mc_means = np.array(all_mc_means)
    all_mc_labels = np.array(all_mc_labels)

    # --- 3a: ECE (Expected Calibration Error) ---
    # 把样本按 uncertainty (std) 分成 10 组
    # 每组算: 预期误差 (mean std) vs 实际误差 (mean abs_error)
    # ECE = sum(组占比 * |预期误差 - 实际误差| / max(实际误差))
    n_groups = 10
    sorted_idx = np.argsort(all_mc_stds)
    group_size = len(sorted_idx) // n_groups

    print(f"\n--- ECE Calculation ({n_groups} groups, {len(all_mc_stds)} samples) ---")
    print(f"{'Group':<8s} {'Uncertainty':>14s} {'Actual Error':>14s} {'Gap':>14s} {'Count':>8s}")
    print("-" * 62)

    ece = 0
    max_actual = np.max(all_mc_abs_errors)
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else len(sorted_idx)
        idx = sorted_idx[start:end]

        expected = np.mean(all_mc_stds[idx])
        actual = np.mean(all_mc_abs_errors[idx])
        gap = abs(expected - actual)
        weight = len(idx) / len(sorted_idx)
        ece += weight * gap / max(actual, 1e-5)

        print(f"{g:<8d} {expected:>14.6f} {actual:>14.6f} {gap:>14.6f} {len(idx):>8d}")

    print(f"\n  ECE (normalized) = {ece:.4f}")
    print(f"  (0 = perfectly calibrated, 1 = completely miscalibrated)")

    # --- 3b: Pearson correlation ---
    corr = np.corrcoef(all_mc_stds, all_mc_abs_errors)[0, 1]
    print(f"\n  Pearson correlation (uncertainty vs abs_error) = {corr:.4f}")

    # --- 3c: 5-group analysis ---
    print(f"\n--- 5-Group Analysis (sorted by uncertainty) ---")
    print(f"{'Group':<8s} {'Uncertainty':>14s} {'Actual Error':>14s} {'Count':>8s}")
    print("-" * 48)
    sorted_idx5 = np.argsort(all_mc_stds)
    group_size5 = len(sorted_idx5) // 5
    for g in range(5):
        start = g * group_size5
        end = (g + 1) * group_size5 if g < 4 else len(sorted_idx5)
        idx = sorted_idx5[start:end]
        print(f"{g:<8d} {np.mean(all_mc_stds[idx]):>14.6f} "
              f"{np.mean(all_mc_abs_errors[idx]):>14.6f} {len(idx):>8d}")

    # --- 3d: Overconfidence analysis ---
    # 高置信（低 uncertainty）样本中，有多少实际误差很大？
    print(f"\n--- Overconfidence Analysis ---")
    low_unc_threshold = np.quantile(all_mc_stds, 0.2)  # 最低 20% uncertainty
    high_unc_threshold = np.quantile(all_mc_stds, 0.8)  # 最高 20% uncertainty

    low_unc_mask = all_mc_stds <= low_unc_threshold
    high_unc_mask = all_mc_stds >= high_unc_threshold

    median_error = np.median(all_mc_abs_errors)
    low_unc_high_error = np.mean(all_mc_abs_errors[low_unc_mask] > median_error) * 100
    high_unc_high_error = np.mean(all_mc_abs_errors[high_unc_mask] > median_error) * 100

    print(f"  Median abs error: {median_error:.4f}")
    print(f"  Low-uncertainty samples (model is confident):")
    print(f"    -> {low_unc_high_error:.1f}% actually have above-median error (OVERCONFIDENT)")
    print(f"  High-uncertainty samples (model is uncertain):")
    print(f"    -> {high_unc_high_error:.1f}% actually have above-median error")

    # --- 3e: Top-1 selection error rate by uncertainty group ---
    print(f"\n--- Top-1 Selection Error Rate by Uncertainty Group ---")
    print(f"{'Group':<8s} {'Uncertainty':>14s} {'Top-1 Error Rate':>18s} {'Count':>8s}")
    print("-" * 54)

    # Re-collect per-task data for top-1 error rate
    del mc_model
    torch.cuda.empty_cache()
    mc_model = load_model(args.mc_dropout_model, device)

    task_uncertainties = []
    task_top1_errors = []

    for net_name, network_file in network_files:
        tasks, task_weights = pickle.load(open(network_file, "rb"))

        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                continue
            datas = pred_a_dataset_dict[task.workload_key]
            mc_mean, mc_std, labels, min_lat = predict(
                mc_model, datas, device,
                mc_samples=args.mc_samples, use_dropout=True)

            model_best = np.argmax(mc_mean)
            true_best = np.argmax(labels)
            is_wrong = 1 if model_best != true_best else 0

            task_uncertainties.append(np.mean(mc_std))
            task_top1_errors.append(is_wrong)

    task_uncertainties = np.array(task_uncertainties)
    task_top1_errors = np.array(task_top1_errors)

    sorted_task_idx = np.argsort(task_uncertainties)
    task_group_size = len(sorted_task_idx) // 5
    print(f"{'Group':<8s} {'Mean Unc':>14s} {'Top-1 Err%':>18s} {'Count':>8s}")
    print("-" * 54)
    for g in range(5):
        start = g * task_group_size
        end = (g + 1) * task_group_size if g < 4 else len(sorted_task_idx)
        idx = sorted_task_idx[start:end]
        print(f"{g:<8d} {np.mean(task_uncertainties[idx]):>14.6f} "
              f"{np.mean(task_top1_errors[idx])*100:>17.1f}% {len(idx):>8d}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("MOTIVATION SUMMARY")
    print("=" * 70)
    print(f"""
1. PREDICTION INACCURACY:
   - Top-1 selection error rate: {top1_wrong}/{total_tasks} = {top1_wrong/total_tasks*100:.1f}%
   - {np.mean(all_rel_errors > 0.20)*100:.1f}% of samples have >20% relative error
   - {np.mean(all_rel_errors > 0.50)*100:.1f}% of samples have >50% relative error
   - When top-1 is wrong, avg speedup loss = {np.mean(all_wrong_speedup_losses):.2f}x

2. NON-UNIFORM ERRORS (see per-network table above)

3. POOR CALIBRATION:
   - ECE = {ece:.4f}
   - Pearson r = {corr:.4f}
   - {low_unc_high_error:.1f}% of confident predictions have above-median error
   - Top-1 error rate increases with uncertainty (see group table above)

CONCLUSION: TLP cost model is inaccurate, non-uniformly so, and poorly calibrated.
=> Uncertainty quantification is needed.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TLP Motivation Analysis")
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--test_dataset_name", type=str,
                        default="tlp_dataset_platinum_8272_2308_test.pkl")
    parser.add_argument("--load_name", type=str,
                        default="runs/tlp_baseline_2308_retry/tlp_model_19.pkl",
                        help="Baseline model for error analysis")
    parser.add_argument("--mc_dropout_model", type=str,
                        default="runs/tlp_mc_dropout_2308/tlp_model_19.pkl",
                        help="MC Dropout model for calibration analysis")
    parser.add_argument("--platform", type=str, default="llvm")
    parser.add_argument("--mc_samples", type=int, default=30,
                        help="MC Dropout samples for uncertainty estimation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("TLP Motivation Analysis")
    print("Proving: (1) model is inaccurate, (2) errors are non-uniform,")
    print("         (3) model doesn't know when it's wrong (poor calibration)")
    print("=" * 70)
    print(f"Baseline model: {args.load_name}")
    print(f"MC Dropout model: {args.mc_dropout_model}")
    print(f"MC samples: {args.mc_samples}")
    print()

    main()
