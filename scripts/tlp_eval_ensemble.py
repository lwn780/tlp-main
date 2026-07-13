"""
Deep Ensemble evaluation for TLP cost model.

Loads multiple model checkpoints (trained with different seeds using
tlp_train_mc_dropout.py or tlp_train_seed.py), computes ensemble mean
(prediction) and ensemble std (uncertainty), then evaluates:

  1. Ensemble top-k scores
  2. Uncertainty-error correlation (Pearson + group analysis)
  3. Uncertainty-aware selection strategies
  4. Comparison: single model vs ensemble vs MC Dropout

Usage:
  cd scripts/
  python3 tlp_eval_ensemble.py \
    --test_dataset_name tlp_dataset_platinum_8272_2308_test.pkl \
    --load_names runs/tlp_baseline_2308_retry/tlp_model_19.pkl \
                 runs/tlp_baseline_2308_seed1/tlp_model_19.pkl \
                 runs/tlp_baseline_2308_seed2/tlp_model_19.pkl \
                 runs/tlp_baseline_2308_seed3/tlp_model_19.pkl \
    --platform llvm \
    --cuda cuda:0 \
    --mc_samples 8 \
    | tee ../notes/ensemble_eval.txt

IMPORTANT: All models must have been trained with tlp_train_mc_dropout.py
(or tlp_train_seed.py which imports from it) to ensure pickle compatibility.
If you have models trained with the original tlp_train.py (without dropout
layers), they CANNOT be loaded here. Retrain them with:
  python3 tlp_train_mc_dropout.py --seed 0 --dropout 0.0
"""
import argparse
import os
import pickle
import sys
from collections import OrderedDict

import numpy as np
import torch

# CRITICAL: Import from tlp_train_mc_dropout, NOT tlp_train.
# All ensemble models must have been trained with this module's AttentionModule
# (which includes dropout layers) for pickle.load to work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tlp_train_mc_dropout import (
    AttentionModule,
    TransformerEncoderLayerModule,
    TransformerModule,
    LSTMModule,
    GPTModule,
    BertModule,
    SegmentDataLoader,
    GPTSegmentDataLoader,
    BertSegmentDataLoader,
    set_seed,
)


top_ks = [1, 5, 10, 20]


# ============================================================
#  Utility
# ============================================================

def enable_dropout(model):
    """Turn on Dropout layers while keeping the rest in eval mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def load_model(model_file, device):
    with open(model_file, "rb") as f:
        loaded = pickle.load(f)
    model = loaded.module if hasattr(loaded, "module") else loaded
    # 兼容旧模型：如果 dropout=0.0 训练的模型没有 self.dropout 属性，
    # 补一个 Identity 层，避免 forward 里 self.dropout(...) 报错
    if not hasattr(model, 'dropout'):
        model.dropout = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def append_remaining(order_prefix, mean_order):
    used = set(order_prefix.tolist())
    rest = [idx for idx in mean_order.tolist() if idx not in used]
    if rest:
        return np.concatenate([order_prefix, np.array(rest, dtype=mean_order.dtype)])
    return order_prefix


def build_strategy_orders(pred_mean, pred_std, args):
    orders = OrderedDict()
    mean_order = np.argsort(-pred_mean)
    orders["mean"] = mean_order

    for penalty_lambda in args.penalty_lambdas:
        name = "global_penalty_%s" % penalty_lambda
        orders[name] = np.argsort(-(pred_mean - penalty_lambda * pred_std))

    for pool_size in args.pool_sizes:
        actual_pool_size = min(pool_size, len(mean_order))
        pool = mean_order[:actual_pool_size]
        low_unc_pool = pool[np.argsort(pred_std[pool])]
        # 用原始 pool_size 命名，保持跨 task 一致
        orders["pool%d_low_unc" % pool_size] = append_remaining(low_unc_pool, mean_order)

        for penalty_lambda in args.pool_lambdas:
            pool_score = pred_mean[pool] - penalty_lambda * pred_std[pool]
            pool_penalty = pool[np.argsort(-pool_score)]
            name = "pool%d_penalty_%s" % (pool_size, penalty_lambda)
            orders[name] = append_remaining(pool_penalty, mean_order)

    for drop_ratio in args.drop_uncertain_ratios:
        threshold = np.quantile(pred_std, 1.0 - drop_ratio)
        safe = np.where(pred_std <= threshold)[0]
        unsafe = np.where(pred_std > threshold)[0]
        safe_order = safe[np.argsort(-pred_mean[safe])]
        unsafe_order = unsafe[np.argsort(-pred_mean[unsafe])]
        name = "drop_high_uncertain_%s" % drop_ratio
        orders[name] = np.concatenate([safe_order, unsafe_order])

    return orders


# ============================================================
#  Prediction helpers
# ============================================================

def predict_single_model(model, datas, device, mc_samples=1, use_dropout=False):
    file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = datas

    if isinstance(model, BertModule):
        test_loader = BertSegmentDataLoader(line_vecs, 512, False)
    elif isinstance(model, GPTModule):
        test_loader = GPTSegmentDataLoader(line_vecs, 512, False)
    else:
        test_loader = SegmentDataLoader(line_vecs, 4000, False)
    assert test_loader.min_latency.min() == test_loader.min_latency.max()

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


def predict_ensemble(model_files, datas, device, mc_samples=1):
    per_model_means = []
    per_model_stds = []
    labels = None
    min_latency = None

    for mf in model_files:
        model = load_model(mf, device)
        pm, ps, labels, min_latency = predict_single_model(
            model, datas, device, mc_samples=mc_samples, use_dropout=False)
        per_model_means.append(pm)
        per_model_stds.append(ps)
        del model
        torch.cuda.empty_cache()

    means_stack = np.stack(per_model_means, axis=0)
    ens_mean = means_stack.mean(axis=0)
    ens_std = means_stack.std(axis=0, ddof=0)

    return ens_mean, ens_std, labels, min_latency, per_model_means


# ============================================================
#  Uncertainty analysis
# ============================================================

def uncertainty_analysis(all_uncertainties, all_abs_errors, label=""):
    if len(all_uncertainties) < 2:
        return None

    unc_arr = np.array(all_uncertainties)
    err_arr = np.array(all_abs_errors)
    corr = np.corrcoef(unc_arr, err_arr)[0, 1]

    print(f"\n{'='*60}")
    print(f"Uncertainty Analysis {label}")
    print(f"{'='*60}")
    print(f"Pearson correlation (uncertainty vs abs_error): {corr:.6f}")
    print(f"Samples: {len(unc_arr)}")
    print(f"Uncertainty  mean={unc_arr.mean():.6f}  std={unc_arr.std():.6f}")
    print(f"Abs Error    mean={err_arr.mean():.6f}  std={err_arr.std():.6f}")

    print("\nGroup analysis (sorted by uncertainty, 5 groups):")
    order = np.argsort(unc_arr)
    unc_sorted = unc_arr[order]
    err_sorted = err_arr[order]
    for group_idx, index_group in enumerate(
            np.array_split(np.arange(len(unc_sorted)), 5)):
        g_unc = unc_sorted[index_group]
        g_err = err_sorted[index_group]
        print(f"  group {group_idx}: uncertainty={g_unc.mean():.6f}  "
              f"abs_error={g_err.mean():.6f}  count={len(index_group)}")

    return corr


# ============================================================
#  Main evaluation
# ============================================================

def eval_ensemble():
    device = args.cuda

    with open(args.test_dataset_name, "rb") as f:
        test_datasets = pickle.load(f)

    pred_a_dataset_dict = {}
    for data in test_datasets:
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        pred_a_dataset_dict[workloadkey] = data

    network_files = [
        "dataset/network_info/((resnet_50,[(1,3,224,224)]),%s).task.pkl" % args.platform,
        "dataset/network_info/((mobilenet_v2,[(1,3,224,224)]),%s).task.pkl" % args.platform,
        "dataset/network_info/((resnext_50,[(1,3,224,224)]),%s).task.pkl" % args.platform,
        "dataset/network_info/((bert_base,[(1,128)]),%s).task.pkl" % args.platform,
        "dataset/network_info/((bert_tiny,[(1,128)]),%s).task.pkl" % args.platform,
    ]

    # ============================================================
    #  Part 1: Per-model single evaluation
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 1: Single Model Evaluation")
    print("=" * 60)

    single_model_results = {}
    for mf in args.load_names:
        seed_name = mf.split("/")[-2] if "/" in mf else mf
        print(f"\n--- Evaluating: {mf} ---")
        model = load_model(mf, device)

        best_latency_total = 0
        top_latency_totals = [0] * len(top_ks)
        all_errors = []

        for network_file in network_files:
            tasks, task_weights = pickle.load(open(network_file, "rb"))
            latencies = [0] * len(top_ks)
            best_latency = 0

            for task, weight in zip(tasks, task_weights):
                if task.workload_key not in pred_a_dataset_dict:
                    continue
                pm, ps, labels, min_lat = predict_single_model(
                    model, pred_a_dataset_dict[task.workload_key], device)
                order = np.argsort(-pm)
                real_values = labels[order]
                real_latency = min_lat / np.maximum(real_values, 1e-5)
                for i, top_k in enumerate(top_ks):
                    latencies[i] += np.min(real_latency[:top_k]) * weight
                best_latency += min_lat * weight
                all_errors.extend(np.abs(pm - labels).tolist())

            best_latency_total += best_latency
            for i in range(len(top_ks)):
                top_latency_totals[i] += latencies[i]

        scores = [best_latency_total / tlt for tlt in top_latency_totals]
        print(f"  top-1={scores[0]:.4f}  top-5={scores[1]:.4f}  "
              f"top-10={scores[2]:.4f}  top-20={scores[3]:.4f}")
        print(f"  mean abs error: {np.mean(all_errors):.4f}")
        single_model_results[seed_name] = {
            "scores": scores,
            "mean_error": np.mean(all_errors),
        }
        del model
        torch.cuda.empty_cache()

    # ============================================================
    #  Part 2: Ensemble evaluation + uncertainty analysis
    # ============================================================
    print("\n" + "=" * 60)
    print(f"Part 2: Deep Ensemble ({len(args.load_names)} models)")
    print("=" * 60)

    strategy_latencies = OrderedDict()
    best_latency_total = 0
    all_ens_errors = []
    all_ens_uncertainties = []
    all_mc_errors = []
    all_mc_uncertainties = []

    for network_file in network_files:
        tasks, task_weights = pickle.load(open(network_file, "rb"))
        best_latency = 0
        latencies = [0] * len(top_ks)

        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                continue

            datas = pred_a_dataset_dict[task.workload_key]

            # Ensemble prediction
            ens_mean, ens_std, labels, min_lat, per_model = predict_ensemble(
                args.load_names, datas, device, mc_samples=1)

            all_ens_errors.extend(np.abs(ens_mean - labels).tolist())
            all_ens_uncertainties.extend(ens_std.tolist())

            # MC Dropout on the dedicated MC Dropout model (for comparison)
            if args.compare_mc_dropout and args.mc_dropout_model:
                mc_model = load_model(args.mc_dropout_model, device)
                mc_mean, mc_std, mc_labels, mc_min_lat = predict_single_model(
                    mc_model, datas, device,
                    mc_samples=args.mc_samples, use_dropout=True)
                all_mc_errors.extend(np.abs(mc_mean - mc_labels).tolist())
                all_mc_uncertainties.extend(mc_std.tolist())
                del mc_model
                torch.cuda.empty_cache()

            # Initialize strategy dict on first task
            if not strategy_latencies:
                for sname in build_strategy_orders(ens_mean, ens_std, args).keys():
                    strategy_latencies[sname] = [0] * len(top_ks)

            # Compute strategies
            current_orders = build_strategy_orders(ens_mean, ens_std, args)
            for strategy_name, order in current_orders.items():
                # 如果当前 task 的策略名之前没出现过（pool_size 被 clamp 了），
                # 跳过它，只累加之前初始化过的策略
                if strategy_name not in strategy_latencies:
                    continue
                real_values = labels[order]
                real_latency = min_lat / np.maximum(real_values, 1e-5)
                for i, top_k in enumerate(top_ks):
                    strategy_latencies[strategy_name][i] += \
                        np.min(real_latency[:top_k]) * weight

            # Also compute simple ensemble mean top-k
            order = np.argsort(-ens_mean)
            real_values = labels[order]
            real_latency = min_lat / np.maximum(real_values, 1e-5)
            for i, top_k in enumerate(top_ks):
                latencies[i] += np.min(real_latency[:top_k]) * weight
            best_latency += min_lat * weight

        best_latency_total += best_latency

    # Print ensemble mean top-k
    ens_topk = []
    for i in range(len(top_ks)):
        total_lat = strategy_latencies["mean"][i]
        score = best_latency_total / total_lat if total_lat > 0 else 0
        ens_topk.append(score)

    print(f"\nEnsemble Mean Top-k:")
    for i, k in enumerate(top_ks):
        print(f"  top-{k}: {ens_topk[i]:.4f}")

    # ============================================================
    #  Part 3: Strategy comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 3: Uncertainty-Aware Selection Strategies (Ensemble)")
    print("=" * 60)
    print(f"{'strategy':<30s} {'top-1':>8s} {'top-5':>8s} {'top-10':>8s} {'top-20':>8s}")
    print("-" * 66)
    for sname, latencies in strategy_latencies.items():
        scores = [best_latency_total / lat if lat > 0 else 0 for lat in latencies]
        print(f"{sname:<30s} {scores[0]:>8.4f} {scores[1]:>8.4f} "
              f"{scores[2]:>8.4f} {scores[3]:>8.4f}")

    # ============================================================
    #  Part 4: Uncertainty analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 4: Uncertainty Quality Comparison")
    print("=" * 60)

    ens_corr = uncertainty_analysis(
        all_ens_uncertainties, all_ens_errors, label="(Ensemble std)")

    mc_corr = None
    if args.compare_mc_dropout and len(all_mc_uncertainties) > 1:
        mc_corr = uncertainty_analysis(
            all_mc_uncertainties, all_mc_errors,
            label=f"(MC Dropout {args.mc_samples} samples)")

    # ============================================================
    #  Part 5: Summary table
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 5: Summary Comparison Table")
    print("=" * 60)
    print(f"\n{'Method':<40s} {'top-1':>8s} {'top-5':>8s} {'top-10':>8s} "
          f"{'top-20':>8s} {'Pearson':>8s}")
    print("-" * 80)

    for name, info in single_model_results.items():
        s = info["scores"]
        print(f"{'Single: ' + name:<40s} {s[0]:>8.4f} {s[1]:>8.4f} "
              f"{s[2]:>8.4f} {s[3]:>8.4f} {'N/A':>8s}")

    ens_corr_str = f"{ens_corr:>8.4f}" if ens_corr else "{'N/A':>8s}"
    print(f"{'Ensemble mean (%d models)' % len(args.load_names):<40s} "
          f"{ens_topk[0]:>8.4f} {ens_topk[1]:>8.4f} {ens_topk[2]:>8.4f} "
          f"{ens_topk[3]:>8.4f} {ens_corr_str}")

    best_strategy = None
    best_top1 = 0
    for sname, latencies in strategy_latencies.items():
        if sname == "mean":
            continue
        score = best_latency_total / latencies[0] if latencies[0] > 0 else 0
        if score > best_top1:
            best_top1 = score
            best_strategy = sname
    if best_strategy:
        sl = strategy_latencies[best_strategy]
        ss = [best_latency_total / l if l > 0 else 0 for l in sl]
        print(f"{'Ensemble + ' + best_strategy:<40s} "
              f"{ss[0]:>8.4f} {ss[1]:>8.4f} {ss[2]:>8.4f} {ss[3]:>8.4f} "
              f"{ens_corr_str}")

    if mc_corr is not None:
        print(f"{'MC Dropout (%d samples)' % args.mc_samples:<40s} "
              f"{'':>8s} {'':>8s} {'':>8s} {'':>8s} {mc_corr:>8.4f}")

    print("\n" + "=" * 60)
    print("Done. Copy this output to notes/ensemble_eval.txt")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Ensemble evaluation for TLP")
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--test_dataset_name", type=str,
                        default="tlp_dataset_platinum_8272_2308_test.pkl")
    parser.add_argument("--load_names", type=str, nargs="+",
                        default=[
                            "runs/tlp_baseline_2308_retry/tlp_model_19.pkl",
                            "runs/tlp_baseline_2308_seed1/tlp_model_19.pkl",
                            "runs/tlp_baseline_2308_seed2/tlp_model_19.pkl",
                            "runs/tlp_baseline_2308_seed3/tlp_model_19.pkl",
                        ],
                        help="List of model checkpoint paths for ensemble")
    parser.add_argument("--platform", type=str, default="llvm")
    parser.add_argument("--mc_samples", type=int, default=8,
                        help="MC Dropout samples (for comparison)")
    parser.add_argument("--mc_dropout_model", type=str,
                        default="runs/tlp_mc_dropout_2308/tlp_model_19.pkl",
                        help="Path to the MC Dropout trained model for comparison")
    parser.add_argument("--compare_mc_dropout", action="store_true", default=True,
                        help="Also run MC Dropout on first model for comparison")

    parser.add_argument("--penalty_lambdas", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--pool_lambdas", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--pool_sizes", type=int, nargs="+",
                        default=[3, 5, 8, 10, 20, 50])
    parser.add_argument("--drop_uncertain_ratios", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 60)
    print("Deep Ensemble Evaluation for TLP")
    print("=" * 60)
    print(f"Models ({len(args.load_names)}):")
    for mf in args.load_names:
        print(f"  {mf}")
    print(f"MC Dropout samples: {args.mc_samples}")
    print(f"MC Dropout model: {args.mc_dropout_model}")
    print(f"Compare MC Dropout: {args.compare_mc_dropout}")
    print()

    eval_ensemble()
