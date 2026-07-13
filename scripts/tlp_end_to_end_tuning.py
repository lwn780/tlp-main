"""
End-to-End Tuning Experiment for TLP + Uncertainty.

Simulates the Ansor tuning loop with a fixed measurement budget:
  1. Cost model predicts all candidate programs for each task
  2. Select N programs to "measure" (using different strategies)
  3. The best measured program's true latency = result
  4. Compare: baseline vs uncertainty-aware vs random vs oracle

This does NOT require actual TVM execution — it uses pre-measured labels
as ground truth, which is the standard evaluation methodology in Ansor/TLP papers.

Strategies compared:
  - random:       randomly pick N programs to measure (lower bound)
  - baseline:     pick top-N by prediction mean (TLP original)
  - pool_low_unc: pick top-N by prediction mean, rerank by uncertainty (ours)
  - global_penalty: pick top-N by (pred_mean - lambda * pred_std)
  - oracle:       pick top-N by true labels (upper bound)

Output:
  - Per-budget speedup table
  - Aggregated speedup curves (for plotting)
  - Per-network breakdown

Usage:
  cd scripts/
  python3 tlp_end_to_end_tuning.py \
    --test_dataset_name tlp_dataset_platinum_8272_2308_test.pkl \
    --ensemble_names \
      runs/tlp_baseline_2308_retry/tlp_model_19.pkl \
      runs/tlp_baseline_2308_seed1/tlp_model_19.pkl \
      runs/tlp_baseline_2308_seed2/tlp_model_19.pkl \
      runs/tlp_baseline_2308_seed3/tlp_model_19.pkl \
    --mc_dropout_model runs/tlp_mc_dropout_2308/tlp_model_19.pkl \
    --platform llvm \
    --cuda cuda:0 \
    --mc_samples 8 \
    | tee ../notes/end_to_end_tuning.txt
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
#  Model loading & prediction
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


def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predict(model, datas, device, mc_samples=1, use_dropout=False):
    file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = datas
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
#  Selection strategies
# ============================================================

def select_baseline(pred_mean, pred_std, budget):
    """TLP original: pick top-N by prediction mean."""
    order = np.argsort(-pred_mean)
    return order[:budget]


def select_pool_low_unc(pred_mean, pred_std, budget, pool_size=3):
    """
    Our method: take top-pool_size by mean, rerank by uncertainty (low first),
    then fill remaining budget with rest of mean-ordered candidates.
    """
    mean_order = np.argsort(-pred_mean)
    actual_pool = min(pool_size, len(mean_order))
    pool = mean_order[:actual_pool]

    # Rerank pool by uncertainty (ascending)
    low_unc_pool = pool[np.argsort(pred_std[pool])]

    # Fill remaining budget with rest of mean-ordered candidates
    used = set(low_unc_pool.tolist())
    rest = [idx for idx in mean_order.tolist() if idx not in used]
    full_order = np.concatenate([low_unc_pool, np.array(rest, dtype=mean_order.dtype)])

    return full_order[:budget]


def select_global_penalty(pred_mean, pred_std, budget, lam=0.5):
    """Global penalty: rank by (pred_mean - lam * pred_std)."""
    score = pred_mean - lam * pred_std
    order = np.argsort(-score)
    return order[:budget]


def select_random(pred_mean, pred_std, budget, rng):
    """Random selection (lower bound)."""
    n = len(pred_mean)
    return rng.choice(n, size=min(budget, n), replace=False)


def select_oracle(labels, budget):
    """Oracle: pick top-N by true labels (upper bound)."""
    order = np.argsort(-labels)
    return order[:budget]


# ============================================================
#  Tuning simulation
# ============================================================

def simulate_tuning(pred_mean, pred_std, labels, min_latency, budgets, rng, pool_size=3):
    """
    Simulate tuning for a single task.

    For each budget N, select N programs to "measure", return the best
    measured program's true latency.

    Returns dict: strategy_name -> {budget -> best_latency}
    """
    n_candidates = len(pred_mean)
    results = OrderedDict()

    for budget in budgets:
        actual_budget = min(budget, n_candidates)

        # Baseline: top-N by prediction
        idx_base = select_baseline(pred_mean, pred_std, actual_budget)
        lat_base = min_latency / max(labels[idx_base].max(), 1e-5)

        # Ours: pool_low_unc
        idx_ours = select_pool_low_unc(pred_mean, pred_std, actual_budget, pool_size)
        lat_ours = min_latency / max(labels[idx_ours].max(), 1e-5)

        # Global penalty
        idx_pen = select_global_penalty(pred_mean, pred_std, actual_budget, lam=0.5)
        lat_pen = min_latency / max(labels[idx_pen].max(), 1e-5)

        # Random
        idx_rand = select_random(pred_mean, pred_std, actual_budget, rng)
        lat_rand = min_latency / max(labels[idx_rand].max(), 1e-5)

        # Oracle
        idx_oracle = select_oracle(labels, actual_budget)
        lat_oracle = min_latency / max(labels[idx_oracle].max(), 1e-5)

        # Best possible (measure all)
        lat_best = min_latency / max(labels.max(), 1e-5)

        results[budget] = {
            'baseline': lat_base,
            'pool_low_unc': lat_ours,
            'global_penalty': lat_pen,
            'random': lat_rand,
            'oracle': lat_oracle,
            'best': lat_best,
        }

    return results


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

    # ============================================================
    #  Step 1: Get predictions for all tasks
    # ============================================================
    print("=" * 70)
    print("Step 1: Computing predictions")
    print("=" * 70)

    # Collect predictions: workload_key -> (pred_mean, pred_std, labels, min_latency)
    all_preds = {}

    if args.use_ensemble and args.ensemble_names:
        print(f"Using Deep Ensemble ({len(args.ensemble_names)} models)")
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
                all_preds[task.workload_key] = (ens_mean, ens_std, labels, min_lat)
                print(f"  {task.workload_key[:40]}... {len(labels)} candidates")

    elif args.mc_dropout_model:
        print(f"Using MC Dropout model (mc_samples={args.mc_samples})")
        mc_model = load_model(args.mc_dropout_model, device)
        for net_name, network_file in network_files:
            tasks, task_weights = pickle.load(open(network_file, "rb"))
            for task, weight in zip(tasks, task_weights):
                if task.workload_key not in pred_a_dataset_dict:
                    continue
                if task.workload_key in all_preds:
                    continue
                datas = pred_a_dataset_dict[task.workload_key]
                mc_mean, mc_std, labels, min_lat = predict(
                    mc_model, datas, device,
                    mc_samples=args.mc_samples, use_dropout=True)
                all_preds[task.workload_key] = (mc_mean, mc_std, labels, min_lat)
                print(f"  {task.workload_key[:40]}... {len(labels)} candidates")
        del mc_model
        torch.cuda.empty_cache()

    else:
        print("ERROR: No model specified. Use --ensemble_names or --mc_dropout_model")
        return

    print(f"\nTotal tasks with predictions: {len(all_preds)}")

    # ============================================================
    #  Step 2: Simulate tuning for each network
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 2: Simulating end-to-end tuning")
    print("=" * 70)

    # Per-network results: net_name -> {budget -> {strategy -> weighted_latency_sum}}
    per_network = OrderedDict()
    per_network_best = OrderedDict()  # net_name -> best_latency_total

    # Aggregate results: budget -> {strategy -> weighted_latency_sum}
    aggregate = OrderedDict()
    aggregate_best = 0  # total best latency across all networks

    for net_name, network_file in network_files:
        tasks, task_weights = pickle.load(open(network_file, "rb"))
        net_best_total = 0

        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in all_preds:
                continue

            pred_mean, pred_std, labels, min_lat = all_preds[task.workload_key]

            # Simulate tuning for this task
            task_results = simulate_tuning(
                pred_mean, pred_std, labels, min_lat,
                budgets, rng, pool_size=args.pool_size)

            # Accumulate weighted latencies
            for budget in budgets:
                if budget not in aggregate:
                    aggregate[budget] = OrderedDict()
                    for sname in task_results[budget]:
                        aggregate[budget][sname] = 0

                if net_name not in per_network:
                    per_network[net_name] = OrderedDict()
                    per_network_best[net_name] = 0

                if budget not in per_network[net_name]:
                    per_network[net_name][budget] = OrderedDict()
                    for sname in task_results[budget]:
                        per_network[net_name][budget][sname] = 0

                for sname, lat in task_results[budget].items():
                    aggregate[budget][sname] += lat * weight
                    per_network[net_name][budget][sname] += lat * weight

            # Best possible (measure all)
            best_lat = min_lat / max(labels.max(), 1e-5)
            net_best_total += best_lat * weight

        per_network_best[net_name] = net_best_total
        aggregate_best += net_best_total
        print(f"  {net_name}: {len(tasks)} tasks processed")

    # ============================================================
    #  Step 3: Compute speedup scores
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 3: Speedup Results (higher = better)")
    print("=" * 70)

    strategies = ['random', 'baseline', 'pool_low_unc', 'global_penalty', 'oracle']

    # Aggregate speedup table
    print(f"\n{'Budget':>8s}", end="")
    for s in strategies:
        print(f" {s:>16s}", end="")
    print()
    print("-" * (8 + 17 * len(strategies)))

    for budget in budgets:
        print(f"{budget:>8d}", end="")
        for s in strategies:
            total_lat = aggregate[budget][s]
            speedup = aggregate_best / total_lat if total_lat > 0 else 0
            print(f" {speedup:>16.4f}", end="")
        print()

    # ============================================================
    #  Step 4: Per-network breakdown
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 4: Per-Network Breakdown")
    print("=" * 70)

    for net_name in per_network:
        print(f"\n--- {net_name} ---")
        print(f"{'Budget':>8s}", end="")
        for s in strategies:
            print(f" {s:>16s}", end="")
        print()
        print("-" * (8 + 17 * len(strategies)))

        for budget in budgets:
            print(f"{budget:>8d}", end="")
            for s in strategies:
                total_lat = per_network[net_name][budget][s]
                net_best = per_network_best[net_name]
                speedup = net_best / total_lat if total_lat > 0 else 0
                print(f" {speedup:>16.4f}", end="")
            print()

    # ============================================================
    #  Step 5: Improvement over baseline
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 5: Improvement of pool_low_unc over baseline")
    print("=" * 70)

    print(f"\n{'Budget':>8s} {'Baseline':>16s} {'Ours':>16s} {'Improv':>10s} {'Improv%':>10s}")
    print("-" * 62)

    for budget in budgets:
        base_lat = aggregate[budget]['baseline']
        ours_lat = aggregate[budget]['pool_low_unc']
        base_speedup = aggregate_best / base_lat if base_lat > 0 else 0
        ours_speedup = aggregate_best / ours_lat if ours_lat > 0 else 0
        improv = ours_speedup - base_speedup
        improv_pct = improv / base_speedup * 100 if base_speedup > 0 else 0
        print(f"{budget:>8d} {base_speedup:>16.4f} {ours_speedup:>16.4f} {improv:>+10.4f} {improv_pct:>+9.2f}%")

    # ============================================================
    #  Step 6: Output for plotting
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 6: Data for plotting (copy to plot script)")
    print("=" * 70)

    print("\n# Aggregate speedup curves")
    print(f"budgets = {budgets}")
    for s in strategies:
        vals = []
        for budget in budgets:
            total_lat = aggregate[budget][s]
            speedup = aggregate_best / total_lat if total_lat > 0 else 0
            vals.append(round(speedup, 4))
        print(f"{s} = {vals}")

    print("\n# Per-network speedup (baseline vs ours) at each budget")
    for net_name in per_network:
        print(f"\n# {net_name}")
        base_vals = []
        ours_vals = []
        for budget in budgets:
            net_best = per_network_best[net_name]
            base_lat = per_network[net_name][budget]['baseline']
            ours_lat = per_network[net_name][budget]['pool_low_unc']
            base_vals.append(round(net_best / base_lat if base_lat > 0 else 0, 4))
            ours_vals.append(round(net_best / ours_lat if ours_lat > 0 else 0, 4))
        print(f"baseline = {base_vals}")
        print(f"ours = {ours_vals}")

    print("\n" + "=" * 70)
    print("Done. Copy this output to notes/end_to_end_tuning.txt")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Tuning Experiment")
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--test_dataset_name", type=str,
                        default="tlp_dataset_platinum_8272_2308_test.pkl")
    parser.add_argument("--ensemble_names", type=str, nargs="+",
                        default=None,
                        help="Ensemble model checkpoints for uncertainty")
    parser.add_argument("--mc_dropout_model", type=str,
                        default=None,
                        help="MC Dropout model checkpoint (alternative to ensemble)")
    parser.add_argument("--use_ensemble", action="store_true", default=True,
                        help="Use ensemble (if ensemble_names provided)")
    parser.add_argument("--platform", type=str, default="llvm")
    parser.add_argument("--mc_samples", type=int, default=8,
                        help="MC Dropout samples (only used if mc_dropout_model)")
    parser.add_argument("--pool_size", type=int, default=3,
                        help="Pool size for pool_low_unc strategy")
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[1, 2, 3, 5, 10, 20, 50, 100, 200],
                        help="Measurement budgets to simulate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("End-to-End Tuning Experiment for TLP + Uncertainty")
    print("=" * 70)
    if args.ensemble_names:
        print(f"Ensemble models ({len(args.ensemble_names)}):")
        for mf in args.ensemble_names:
            print(f"  {mf}")
    if args.mc_dropout_model:
        print(f"MC Dropout model: {args.mc_dropout_model}")
        print(f"MC samples: {args.mc_samples}")
    print(f"Pool size: {args.pool_size}")
    print(f"Budgets: {args.budgets}")
    print()

    main()
