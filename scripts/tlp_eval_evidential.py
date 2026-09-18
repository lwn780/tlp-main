"""Evaluate evidential TLP on the test dataset.

Mirrors the protocol of tlp_eval_mc_dropout.py: weighted top-k scores over the
5 held-out networks, pooled Pearson(sigma, |gamma - label|) and 5-group
uncertainty analysis. Single deterministic forward pass per candidate
(dropout disabled), so the point prediction is gamma and the uncertainty
comes from the NIG posterior (epistemic / aleatoric / predictive).
"""

import argparse
import pickle

import numpy as np
import torch

from tlp_train_evidential import AttentionModule, SegmentDataLoader, nig_uncertainties


top_ks = [1, 5, 10, 20]


def pred_a_dataset(datas, task_pred_dict, model):
    datas_new = []
    for data in [datas]:
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        datas_new.extend(line_vecs)

    test_loader = SegmentDataLoader(datas_new, 4000, False)
    if test_loader.min_latency.min() != test_loader.min_latency.max():
        print("warning: min_latency varies within this group (expected only for smoke mode)")

    preds_all = []
    labels_all = []

    model.eval()
    with torch.no_grad():
        for batch_datas_steps, batch_labels in test_loader:
            batch_datas_steps = batch_datas_steps.to(device)
            preds = model(batch_datas_steps)
            preds_all.append(preds.detach().cpu())
            labels_all.append(batch_labels.detach().cpu())

    preds_all = torch.cat(preds_all, dim=0)
    labels = torch.cat(labels_all, dim=0).numpy()

    gamma_t, ale_t, epi_t = nig_uncertainties(preds_all)
    gamma = gamma_t.numpy()
    aleatoric = ale_t.numpy()
    epistemic = epi_t.numpy()
    total = aleatoric + epistemic

    if args.unc_source == "epistemic":
        sigma = epistemic
    elif args.unc_source == "aleatoric":
        sigma = aleatoric
    else:
        sigma = total
    rank_score = gamma - args.uncertainty_lambda * sigma

    task_pred_dict[workloadkey] = (
        rank_score,
        gamma,
        epistemic,
        aleatoric,
        total,
        test_loader.min_latency.min().numpy(),
        labels,
    )


def report_uncertainty_quality(network_name, all_abs_errors, all_epis, all_ales, all_tots):
    abs_error_arr = np.array(all_abs_errors)
    epi_arr = np.array(all_epis)
    ale_arr = np.array(all_ales)
    tot_arr = np.array(all_tots)

    corr_epi = np.corrcoef(epi_arr, abs_error_arr)[0, 1]
    corr_ale = np.corrcoef(ale_arr, abs_error_arr)[0, 1]
    corr_tot = np.corrcoef(tot_arr, abs_error_arr)[0, 1]
    print(f"[{network_name}] samples {len(abs_error_arr)}\t"
          f"Pearson(epistemic) {corr_epi:.4f}\t"
          f"Pearson(aleatoric) {corr_ale:.4f}\t"
          f"Pearson(total) {corr_tot:.4f}")
    return corr_epi, corr_ale, corr_tot


def group_analysis(sigma_arr, abs_error_arr):
    print("uncertainty group analysis (sorted by %s):" % args.unc_source)
    order = np.argsort(sigma_arr)
    sigma_sorted = sigma_arr[order]
    error_sorted = abs_error_arr[order]
    for group_idx, index_group in enumerate(np.array_split(np.arange(len(sigma_sorted)), 5)):
        print("group %d uncertainty %.6f abs_error %.6f count %d" % (
            group_idx,
            sigma_sorted[index_group].mean(),
            error_sorted[index_group].mean(),
            len(index_group),
        ))


def eval_model(model_file):
    with open(model_file, "rb") as f:
        loaded_model = pickle.load(f)

    if hasattr(loaded_model, "module"):
        model = loaded_model.module.to(device)
    else:
        model = loaded_model.to(device)

    task_pred_dict = {}
    pred_a_dataset_dict = {}
    if not (args.smoke and len(test_datasets[0]) == 3):
        for data in test_datasets:
            file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
            pred_a_dataset_dict[workloadkey] = data

    if args.smoke:
        all_abs_errors, all_epis, all_ales, all_tots = [], [], [], []
        for data in test_datasets:
            if len(data) == 3:
                # flat train-format pkl: (datas_step, label, min_lat) per entry,
                # treat the whole file as one pseudo task
                line_vecs = test_datasets
                workloadkey = "smoke_task"
            else:
                file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
            if workloadkey in task_pred_dict:
                continue
            pred_a_dataset((None, None, None, workloadkey, None, None, line_vecs),
                           task_pred_dict, model)
            rank_score, gamma, epi, ale, tot, min_latency, labels = task_pred_dict[workloadkey]
            all_abs_errors.extend(np.abs(gamma - labels).tolist())
            all_epis.extend(epi.tolist())
            all_ales.extend(ale.tolist())
            all_tots.extend(tot.tolist())
        report_uncertainty_quality("SMOKE(all)", all_abs_errors, all_epis, all_ales, all_tots)
        sigma_arr = {"epistemic": np.array(all_epis),
                     "aleatoric": np.array(all_ales),
                     "total": np.array(all_tots)}[args.unc_source]
        group_analysis(sigma_arr, np.array(all_abs_errors))
        return

    files = [
        "dataset/network_info/((resnet_50,[(1,3,224,224)]),%s).task.pkl" % args.platform,
        "dataset/network_info/((mobilenet_v2,[(1,3,224,224)]),%s).task.pkl" % args.platform,
        "dataset/network_info/((resnext_50,[(1,3,224,224)]),%s).task.pkl" % args.platform,
        "dataset/network_info/((bert_base,[(1,128)]),%s).task.pkl" % args.platform,
        "dataset/network_info/((bert_tiny,[(1,128)]),%s).task.pkl" % args.platform,
    ]

    best_latency_total = 0
    top_latency_totals = [0 for _ in top_ks]

    all_abs_errors, all_epis, all_ales, all_tots = [], [], [], []

    for file in files:
        network_name = file.split("((")[1].split(",[(")[0]
        tasks, task_weights = pickle.load(open(file, "rb"))
        latencies = [0] * len(top_ks)
        best_latency = 0

        net_abs_errors, net_epis, net_ales, net_tots = [], [], [], []

        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                print("error task.workload_key not in pred_a_dataset_dict")
                continue

            pred_a_dataset(pred_a_dataset_dict[task.workload_key], task_pred_dict, model)
            rank_score, gamma, epi, ale, tot, min_latency, labels = task_pred_dict[task.workload_key]

            real_values = labels[np.argsort(-rank_score)]
            real_latency = min_latency / np.maximum(real_values, 1e-5)

            for i, top_k in enumerate(top_ks):
                latencies[i] += np.min(real_latency[:top_k]) * weight

            best_latency += min_latency * weight

            net_abs_errors.extend(np.abs(gamma - labels).tolist())
            net_epis.extend(epi.tolist())
            net_ales.extend(ale.tolist())
            net_tots.extend(tot.tolist())

        print(f"top 1 score: {best_latency / latencies[0]}")
        print(f"top 5 score: {best_latency / latencies[1]}")

        report_uncertainty_quality(network_name, net_abs_errors, net_epis, net_ales, net_tots)

        all_abs_errors.extend(net_abs_errors)
        all_epis.extend(net_epis)
        all_ales.extend(net_ales)
        all_tots.extend(net_tots)

        best_latency_total += best_latency
        for i in range(len(top_ks)):
            top_latency_totals[i] += latencies[i]

    for i, top_k in enumerate(top_ks):
        print(f"average top {top_k} score is {best_latency_total / top_latency_totals[i]}")

    print("=== pooled uncertainty quality (same protocol as MC dropout eval) ===")
    report_uncertainty_quality("TOTAL", all_abs_errors, all_epis, all_ales, all_tots)

    sigma_arr = {"epistemic": np.array(all_epis),
                 "aleatoric": np.array(all_ales),
                 "total": np.array(all_tots)}[args.unc_source]
    group_analysis(sigma_arr, np.array(all_abs_errors))

    print("=== reference (published paper) ===")
    print("MC Dropout pooled Pearson: 0.339, top-1 0.861 / top-5 0.947 / top-10 0.953")
    print("Deep Ensemble pooled Pearson: 0.125, top-1 0.882 / top-5 0.948 / top-10 0.975")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--test_dataset_name", type=str, default="tlp_dataset_platinum_8272_2308_test.pkl")
    parser.add_argument("--load_name", type=str, default="runs/tlp_evidential_seed0_2308/tlp_model_19.pkl")
    parser.add_argument("--platform", type=str, default="llvm")
    parser.add_argument("--uncertainty_lambda", type=float, default=0.0)
    parser.add_argument("--unc_source", type=str, default="epistemic",
                        choices=["epistemic", "aleatoric", "total"])
    parser.add_argument("--smoke", action="store_true",
                        help="skip network_info task files, only run uncertainty quality on the given pkl")
    args = parser.parse_args()
    print(args)

    device = args.cuda

    with open(args.test_dataset_name, "rb") as f:
        test_datasets = pickle.load(f)

    eval_model(args.load_name)
