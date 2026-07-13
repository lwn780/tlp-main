"""
Ablation Study for TLP + Uncertainty Selection Strategies.

Computes predictions ONCE (using Deep Ensemble), then sweeps:
  - global_penalty: lambda in [0.1, 0.3, 0.5, 0.7, 1.0]
  - pool_low_unc:   pool_size in [3, 5, 10]

Output: compact comparison tables for paper Section 5.4 (Ablation).

Usage:
  cd scripts/
  python3 tlp_ablation_study.py \
    --test_dataset_name tlp_dataset_platinum_8272_2308_test.pkl \
    --ensemble_names \
      runs/tlp_baseline_2308_retry/tlp_model_19.pkl \
      runs/tlp_ensemble_seed1_2308/tlp_model_19.pkl \
      runs/tlp_ensemble_seed2_2308/tlp_model_19.pkl \
      runs/tlp_baseline_2308_seed3/tlp_model_19.pkl \
    --platform llvm \
    --cuda cuda:0 \
    | tee ../notes/ablation_study.txt
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
    GPTSegmentDataLoader,
    BertSegmentDataLoader,
    set_seed,
)


# ============================================================
#  Reuse core functions from end-to-end script
# ============================================================

def load_model(model_file, device):
    with open(model_file, "rb") as f:
        loaded = pickle.load(f)
    model = loaded.module if hasattr(loaded, "module") else loaded
    if not hasattr(model, 'dropout'):
        model.dropout = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def predict(model, datas, device, mc_samples=1, use_dropout=False):
    file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = datas
    test_loader = SegmentDataLoader(line_vecs, 4000, False)
    assert test_loader.min_latency.min() == test_loader.min_latency.max()

    preds_samples = []
    labels_all = None

    with torch.no_grad():
        for _ in range(mc_samples):
            model.eval()
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


def predict_ensemble(model_files, datas, device):
    per_model_means = []
    labels = None
    min_latency = None

    for mf in model_files:
        model = load_model(mf, device)
        pm, _, labels, min_latency = predict(model, datas, device, mc_samples=1, use_dropout=False)
        per_model_means.append(pm)
        del model
        torch.cuda.empty_cache()

    means_stack = np.stack(per_model_means, axis=0)
    ens_mean = means_stack.mean(axis=0)
    ens_std = means_stack.std(axis=0, ddof=0)

    return ens_mean, ens_std, labels, min_latency


# ============================================================
#  Selection strategies (parameterized)
# ============================================================

def select_baseline(pred_mean, budget):
    order = np.argsort(-pred_mean)
    return order[:budget]


def select_pool_low_unc(pred_mean, pred_std, budget, pool_size=3):
    mean_order = np.argsort(-pred_mean)
    actual_pool = min(pool_size, len(mean_order))
    pool = mean_order[:actual_pool]
    low_unc_pool = pool[np.argsort(pred_std[pool])]
    used = set(low_unc_pool.tolist())
    rest = [idx for idx in mean_order.tolist() if idx not in used]
    full_order = np.concatenate([low_unc_pool, np.array(rest, dtype=mean_order.dtype)])
    return full_order[:budget]


def select_global_penalty(pred_mean, pred_std, budget, lam=0.5):
    score = pred_mean - lam * pred_std
    order = np.argsort(-score)
    return order[:budget]


def select_random(pred_mean, budget, rng):
    n = len(pred_mean)
    return rng.choice(n, size=min(budget, n), replace=False)


def select_oracle(labels, budget):
    order = np.argsort(-labels)
    return order[:budget]


def compute_speedup(labels, idx, min_latency, best_total, weight):
    """Compute weighted speedup for a single task."""
    lat = min_latency / max(labels[idx].max(), 1e-5)
    return lat * weight


# ============================================================
#  Main
# ============================================================

def main():
    device = args.cuda
    rng = np.random.RandomState(args.seed)

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

    budgets = args.budgets
    lambdas = [0.1, 0.3, 0.5, 0.7, 1.0]
    pool_sizes = [3, 5, 10]

    # ============================================================
    #  Step 1: Compute predictions ONCE
    # ============================================================
    print("=" * 70)
    print("Ablation Study: Sweeping lambda and pool_size")
    print("=" * 70)
    print(f"Ensemble models: {len(args.ensemble_names)}")
    print(f"Lambdas: {lambdas}")
    print(f"Pool sizes: {pool_sizes}")
    print(f"Budgets: {budgets}")
    print()

    all_preds = {}
    print("Computing Ensemble predictions...")
    for net_name, network_file in network_files:
        tasks, task_weights = pickle.load(open(network_file, "rb"))
        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                continue
            if task.workload_key in all_preds:
                continue
            datas = pred_a_dataset_dict[task.workload_key]
            ens_mean, ens_std, labels, min_lat = predict_ensemble(
                args.ensemble_names, datas, device)
            all_preds[task.workload_key] = (ens_mean, ens_std, labels, min_lat, weight, net_name)

    print(f"Total tasks: {len(all_preds)}")

    # ============================================================
    #  Step 2: Run all strategy variants
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 2: Running ablation variants")
    print("=" * 70)

    # For each variant, accumulate weighted latency per budget
    # Structure: variant_name -> {budget -> total_weighted_lat}
    # Also need baseline, random, oracle

    # Define all variants
    variants = OrderedDict()
    variants['baseline'] = {'type': 'baseline'}
    variants['random'] = {'type': 'random'}
    for ps in pool_sizes:
        variants[f'pool_low_unc(ps={ps})'] = {'type': 'pool', 'pool_size': ps}
    for lam in lambdas:
        variants[f'global_penalty(lam={lam})'] = {'type': 'penalty', 'lam': lam}
    variants['oracle'] = {'type': 'oracle'}

    # Initialize accumulators
    # variant -> {budget -> total_weighted_lat}
    variant_results = OrderedDict()
    for vname in variants:
        variant_results[vname] = OrderedDict()
        for b in budgets:
            variant_results[vname][b] = 0.0

    aggregate_best = 0.0

    for wkl, (pred_mean, pred_std, labels, min_lat, weight, net_name) in all_preds.items():
        best_lat = min_lat / max(labels.max(), 1e-5)
        aggregate_best += best_lat * weight

        for budget in budgets:
            actual_budget = min(budget, len(pred_mean))

            for vname, vcfg in variants.items():
                if vcfg['type'] == 'baseline':
                    idx = select_baseline(pred_mean, actual_budget)
                elif vcfg['type'] == 'random':
                    idx = select_random(pred_mean, actual_budget, rng)
                elif vcfg['type'] == 'pool':
                    idx = select_pool_low_unc(pred_mean, pred_std, actual_budget, vcfg['pool_size'])
                elif vcfg['type'] == 'penalty':
                    idx = select_global_penalty(pred_mean, pred_std, actual_budget, vcfg['lam'])
                elif vcfg['type'] == 'oracle':
                    idx = select_oracle(labels, actual_budget)

                lat = min_lat / max(labels[idx].max(), 1e-5)
                variant_results[vname][budget] += lat * weight

    # ============================================================
    #  Step 3: Output results
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 3: Ablation Results (Speedup, higher = better)")
    print("=" * 70)

    # --- Table A: global_penalty lambda sweep ---
    print("\n--- Table A: global_penalty lambda sweep ---")
    print(f"\n{'Budget':>8s}  {'baseline':>10s}", end="")
    for lam in lambdas:
        print(f"  {'lam='+str(lam):>10s}", end="")
    print(f"  {'oracle':>10s}")
    print("-" * (12 + 12 * (len(lambdas) + 2)))

    for budget in budgets:
        print(f"{budget:>8d}", end="")
        for vname in ['baseline'] + [f'global_penalty(lam={l})' for l in lambdas] + ['oracle']:
            total_lat = variant_results[vname][budget]
            speedup = aggregate_best / total_lat if total_lat > 0 else 0
            print(f"  {speedup:>10.4f}", end="")
        print()

    # Improvement over baseline for each lambda
    print(f"\n{'Budget':>8s}", end="")
    for lam in lambdas:
        print(f"  {'lam='+str(lam):>10s}", end="")
    print()
    print("-" * (12 + 12 * len(lambdas)))
    for budget in budgets:
        base_lat = variant_results['baseline'][budget]
        base_sp = aggregate_best / base_lat if base_lat > 0 else 0
        print(f"{budget:>8d}", end="")
        for lam in lambdas:
            vname = f'global_penalty(lam={lam})'
            total_lat = variant_results[vname][budget]
            sp = aggregate_best / total_lat if total_lat > 0 else 0
            diff = (sp - base_sp) / base_sp * 100 if base_sp > 0 else 0
            print(f"  {diff:>+9.2f}%", end="")
        print()

    # --- Table B: pool_low_unc pool_size sweep ---
    print("\n\n--- Table B: pool_low_unc pool_size sweep ---")
    print(f"\n{'Budget':>8s}  {'baseline':>10s}", end="")
    for ps in pool_sizes:
        print(f"  {'ps='+str(ps):>10s}", end="")
    print(f"  {'oracle':>10s}")
    print("-" * (12 + 12 * (len(pool_sizes) + 2)))

    for budget in budgets:
        print(f"{budget:>8d}", end="")
        for vname in ['baseline'] + [f'pool_low_unc(ps={p})' for p in pool_sizes] + ['oracle']:
            total_lat = variant_results[vname][budget]
            speedup = aggregate_best / total_lat if total_lat > 0 else 0
            print(f"  {speedup:>10.4f}", end="")
        print()

    # Improvement over baseline for each pool_size
    print(f"\n{'Budget':>8s}", end="")
    for ps in pool_sizes:
        print(f"  {'ps='+str(ps):>10s}", end="")
    print()
    print("-" * (12 + 12 * len(pool_sizes)))
    for budget in budgets:
        base_lat = variant_results['baseline'][budget]
        base_sp = aggregate_best / base_lat if base_lat > 0 else 0
        print(f"{budget:>8d}", end="")
        for ps in pool_sizes:
            vname = f'pool_low_unc(ps={ps})'
            total_lat = variant_results[vname][budget]
            sp = aggregate_best / total_lat if total_lat > 0 else 0
            diff = (sp - base_sp) / base_sp * 100 if base_sp > 0 else 0
            print(f"  {diff:>+9.2f}%", end="")
        print()

    # --- Table C: Best variant at each budget ---
    print("\n\n--- Table C: Best strategy at each budget ---")
    print(f"\n{'Budget':>8s}  {'Best Variant':>30s}  {'Speedup':>10s}  {'vs baseline':>12s}")
    print("-" * 66)

    # Only compare our methods (exclude baseline, random, oracle)
    our_variants = [v for v in variants if v not in ('baseline', 'random', 'oracle')]

    for budget in budgets:
        base_lat = variant_results['baseline'][budget]
        base_sp = aggregate_best / base_lat if base_lat > 0 else 0

        best_vname = None
        best_sp = 0
        for vname in our_variants:
            total_lat = variant_results[vname][budget]
            sp = aggregate_best / total_lat if total_lat > 0 else 0
            if sp > best_sp:
                best_sp = sp
                best_vname = vname

        diff = (best_sp - base_sp) / base_sp * 100 if base_sp > 0 else 0
        print(f"{budget:>8d}  {best_vname:>30s}  {best_sp:>10.4f}  {diff:>+11.2f}%")

    # --- Plotting data ---
    print("\n\n" + "=" * 70)
    print("Plotting Data")
    print("=" * 70)

    print("\n# Lambda sweep data")
    print(f"budgets = {budgets}")
    for lam in lambdas:
        vname = f'global_penalty(lam={lam})'
        vals = []
        for b in budgets:
            total_lat = variant_results[vname][b]
            sp = aggregate_best / total_lat if total_lat > 0 else 0
            vals.append(round(float(sp), 4))
        print(f"lam_{lam} = {vals}")

    print("\n# Pool size sweep data")
    for ps in pool_sizes:
        vname = f'pool_low_unc(ps={ps})'
        vals = []
        for b in budgets:
            total_lat = variant_results[vname][b]
            sp = aggregate_best / total_lat if total_lat > 0 else 0
            vals.append(round(float(sp), 4))
        print(f"ps_{ps} = {vals}")

    print("\nbaseline = " + str([round(float(aggregate_best / variant_results['baseline'][b])
                                     if variant_results['baseline'][b] > 0 else 0, 4) for b in budgets]))
    print("oracle = " + str([round(float(aggregate_best / variant_results['oracle'][b])
                                    if variant_results['oracle'][b] > 0 else 0, 4) for b in budgets]))

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study for Selection Strategies")
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--test_dataset_name", type=str,
                        default="tlp_dataset_platinum_8272_2308_test.pkl")
    parser.add_argument("--ensemble_names", type=str, nargs="+",
                        default=None,
                        help="Ensemble model checkpoints")
    parser.add_argument("--platform", type=str, default="llvm")
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[1, 2, 3, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    main()
