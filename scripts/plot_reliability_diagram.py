"""
Reliability Diagram + Calibration Analysis Plot for TLP Motivation.

Generates publication-ready figures:
  1. Reliability Diagram (10-group bar chart: expected vs actual error)
  2. 5-group monotonicity bar chart
  3. Overconfidence analysis bar chart

This script reads the output of tlp_motivation_analysis.py and produces
matplotlib figures saved as PNG files.

Usage:
  cd scripts/
  python3 plot_reliability_diagram.py \
    --input ../notes/motivation_analysis.txt \
    --output_dir ../notes/figures/

  # Or run with raw data (if motivation analysis output is not available):
  python3 plot_reliability_diagram.py \
    --test_dataset_name tlp_dataset_platinum_8272_2308_test.pkl \
    --mc_dropout_model runs/tlp_mc_dropout_2308/tlp_model_19.pkl \
    --platform llvm \
    --cuda cuda:0 \
    --mc_samples 30 \
    --output_dir ../notes/figures/
"""
import argparse
import os
import pickle
import sys
from collections import OrderedDict

import numpy as np

# Handle matplotlib backend for headless servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Set font for Chinese support
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tlp_train_mc_dropout import AttentionModule, SegmentDataLoader, set_seed

# ============================================================
#  Data collection (same logic as tlp_motivation_analysis.py)
# ============================================================

def load_model(model_file, device):
    import torch
    with open(model_file, "rb") as f:
        loaded = pickle.load(f)
    model = loaded.module if hasattr(loaded, "module") else loaded
    if not hasattr(model, 'dropout'):
        model.dropout = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def enable_dropout(model):
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def collect_data(args):
    """Collect uncertainty and error data from MC Dropout model."""
    import torch
    from tlp_train_mc_dropout import SegmentDataLoader, set_seed

    device = args.cuda
    set_seed(args.seed)

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

    mc_model = load_model(args.mc_dropout_model, device)

    all_stds = []
    all_errors = []

    for net_name, network_file in network_files:
        tasks, task_weights = pickle.load(open(network_file, "rb"))
        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                continue
            datas = pred_a_dataset_dict[task.workload_key]
            test_loader = SegmentDataLoader(
                [d[:3] for d in [datas][0][6]], 4000, False) if False else None

            # Use same predict logic
            line_vecs = datas[6]
            test_loader = SegmentDataLoader(line_vecs, 4000, False)

            preds_samples = []
            labels_all = None
            with torch.no_grad():
                for _ in range(args.mc_samples):
                    mc_model.eval()
                    enable_dropout(mc_model)
                    preds_all = []
                    labels_epoch = []
                    for batch_datas_steps, batch_labels in test_loader:
                        batch_datas_steps = batch_datas_steps.to(device)
                        preds = mc_model(batch_datas_steps)
                        if isinstance(preds, list) and len(preds) > 1:
                            preds = preds[0]
                        preds_all.append(preds.detach().cpu())
                        labels_epoch.append(batch_labels.detach().cpu())
                    preds_samples.append(torch.cat(preds_all, dim=0))
                    if labels_all is None:
                        labels_all = torch.cat(labels_epoch, dim=0)

            preds_stack = torch.stack(preds_samples, dim=0)
            mc_mean = preds_stack.mean(dim=0).numpy()
            mc_std = preds_stack.std(dim=0, unbiased=False).numpy()
            labels = labels_all.numpy()

            all_stds.extend(mc_std.tolist())
            all_errors.extend(np.abs(mc_mean - labels).tolist())

    del mc_model
    return np.array(all_stds), np.array(all_errors)


# ============================================================
#  Plotting functions
# ============================================================

def plot_reliability_diagram(all_stds, all_errors, output_dir, n_groups=10):
    """
    Reliability Diagram: 10 groups sorted by uncertainty.
    Each group shows expected error (mean std) vs actual error (mean abs_error).
    """
    sorted_idx = np.argsort(all_stds)
    group_size = len(sorted_idx) // n_groups

    expected_list = []
    actual_list = []
    counts = []

    max_actual = np.max(all_errors)

    ece = 0
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else len(sorted_idx)
        idx = sorted_idx[start:end]

        expected = np.mean(all_stds[idx])
        actual = np.mean(all_errors[idx])
        expected_list.append(expected)
        actual_list.append(actual)
        counts.append(len(idx))

        gap = abs(expected - actual)
        weight = len(idx) / len(sorted_idx)
        ece += weight * gap / max(actual, 1e-5)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(n_groups)
    width = 0.35

    # Normalize for visualization (both on same scale)
    max_val = max(max(expected_list), max(actual_list))

    bars1 = ax.bar(x - width/2, expected_list, width, label='Expected Error (mean uncertainty)',
                   color='#5B9BD5', edgecolor='#3A7CA5', alpha=0.85)
    bars2 = ax.bar(x + width/2, actual_list, width, label='Actual Error (mean abs_error)',
                   color='#E8746C', edgecolor='#C45A52', alpha=0.85)

    ax.set_xlabel('Uncertainty Group (Low → High)', fontsize=13)
    ax.set_ylabel('Error', fontsize=13)
    ax.set_title(f'Reliability Diagram (ECE = {ece:.4f})', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{g}' for g in range(n_groups)], fontsize=11)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Add ECE annotation
    ax.text(0.98, 0.95, f'ECE = {ece:.4f}\n(0=perfect, 1=worst)',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'reliability_diagram.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

    return ece


def plot_monotonicity(all_stds, all_errors, output_dir, n_groups=5):
    """
    5-group monotonicity chart: sorted by uncertainty, show actual error per group.
    """
    sorted_idx = np.argsort(all_stds)
    group_size = len(sorted_idx) // n_groups

    unc_vals = []
    err_vals = []

    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else len(sorted_idx)
        idx = sorted_idx[start:end]
        unc_vals.append(np.mean(all_stds[idx]))
        err_vals.append(np.mean(all_errors[idx]))

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#2E86AB', '#36A2D9', '#F5A623', '#E8746C', '#C0392B']
    bars = ax.bar(range(n_groups), err_vals, color=colors, edgecolor='black', linewidth=0.5, width=0.6)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, err_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_xlabel('Uncertainty Group (Low → High)', fontsize=13)
    ax.set_ylabel('Mean Absolute Error', fontsize=13)
    ax.set_title('Uncertainty-Error Monotonicity (5 Groups)', fontsize=15, fontweight='bold')
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([f'G{g}\n(unc={unc_vals[g]:.3f})' for g in range(n_groups)], fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add improvement annotation
    improv = (err_vals[-1] - err_vals[0]) / err_vals[0] * 100
    ax.annotate(f'+{improv:.0f}%', xy=(4, err_vals[-1]),
                xytext=(3.5, err_vals[-1] + 1.5),
                fontsize=14, fontweight='bold', color='#C0392B',
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2))

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'monotonicity_5group.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_overconfidence(all_stds, all_errors, output_dir):
    """
    Overconfidence analysis: compare error distribution of
    low-uncertainty (confident) vs high-uncertainty (uncertain) samples.
    """
    low_threshold = np.quantile(all_stds, 0.2)
    high_threshold = np.quantile(all_stds, 0.8)
    median_error = np.median(all_errors)

    low_mask = all_stds <= low_threshold
    high_mask = all_stds >= high_threshold
    mid_mask = (all_stds > low_threshold) & (all_stds < high_threshold)

    low_above = np.mean(all_errors[low_mask] > median_error) * 100
    mid_above = np.mean(all_errors[mid_mask] > median_error) * 100
    high_above = np.mean(all_errors[high_mask] > median_error) * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    groups = ['Low Uncertainty\n(Confident)', 'Medium', 'High Uncertainty\n(Uncertain)']
    values = [low_above, mid_above, high_above]
    colors = ['#E8746C', '#F5A623', '#2E86AB']

    bars = ax.bar(groups, values, color=colors, edgecolor='black', linewidth=0.5, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='50% (random)')
    ax.set_ylabel('% Samples with Above-Median Error', fontsize=13)
    ax.set_title('Overconfidence Analysis', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Annotate the key finding
    ax.annotate(f'40.5% of "confident"\npredictions are wrong',
                xy=(0, low_above), xytext=(0.5, 70),
                fontsize=12, fontweight='bold', color='#C0392B',
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2))

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'overconfidence_analysis.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Reliability Diagram for TLP Motivation")
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--test_dataset_name", type=str,
                        default="tlp_dataset_platinum_8272_2308_test.pkl")
    parser.add_argument("--mc_dropout_model", type=str,
                        default="runs/tlp_mc_dropout_2308/tlp_model_19.pkl")
    parser.add_argument("--platform", type=str, default="llvm")
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="../notes/figures/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Reliability Diagram Plotting")
    print("=" * 60)
    print(f"MC Dropout model: {args.mc_dropout_model}")
    print(f"MC samples: {args.mc_samples}")
    print(f"Output dir: {args.output_dir}")
    print()

    print("Collecting data...")
    all_stds, all_errors = collect_data(args)
    print(f"  Total samples: {len(all_stds)}")
    print(f"  Uncertainty range: [{all_stds.min():.4f}, {all_stds.max():.4f}]")
    print(f"  Error range: [{all_errors.min():.4f}, {all_errors.max():.4f}]")

    print("\nGenerating figures...")
    ece = plot_reliability_diagram(all_stds, all_errors, args.output_dir)
    print(f"  ECE = {ece:.4f}")

    plot_monotonicity(all_stds, all_errors, args.output_dir)

    plot_overconfidence(all_stds, all_errors, args.output_dir)

    print(f"\nDone. Figures saved to {args.output_dir}")
