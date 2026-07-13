import argparse
import pickle

import numpy as np
import torch
from tlp_train_mc_dropout import *


top_ks = [1, 5, 10, 20]


def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def pred_a_dataset(datas, task_pred_dict, model):
    datas_new = []
    for data in [datas]:
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        datas_new.extend(line_vecs)

    if isinstance(model, BertModule):
        test_loader = BertSegmentDataLoader(datas_new, 512, False)
    elif isinstance(model, GPTModule):
        test_loader = GPTSegmentDataLoader(datas_new, 512, False)
    else:
        test_loader = SegmentDataLoader(datas_new, 4000, False)

    assert test_loader.min_latency.min() == test_loader.min_latency.max()

    preds_samples = []
    labels_all = None

    with torch.no_grad():
        for _ in range(args.mc_samples):
            preds_all = []
            labels_epoch = []

            model.eval()
            enable_dropout(model)

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
    pred_mean = preds_stack.mean(dim=0).numpy()
    pred_std = preds_stack.std(dim=0, unbiased=False).numpy()
    labels = labels_all.numpy()

    rank_score = pred_mean - args.uncertainty_lambda * pred_std

    task_pred_dict[workloadkey] = (
        rank_score,
        pred_mean,
        pred_std,
        test_loader.min_latency.min().numpy(),
        labels,
    )


def eval_model(model_file):
    with open(model_file, "rb") as f:
        loaded_model = pickle.load(f)

    if hasattr(loaded_model, "module"):
        model = loaded_model.module.to(device)
    else:
        model = loaded_model.to(device)

    task_pred_dict = {}

    pred_a_dataset_dict = {}
    for data in test_datasets:
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        pred_a_dataset_dict[workloadkey] = data

    files = [
        "dataset/network_info/((resnet_50,[(1,3,224,224)]),%s).task.pkl" % args.platform,
        "dataset/network_info/((mobilenet_v2,[(1,3,224,224)]),%s).task.pkl" % args.platform,
        "dataset/network_info/((resnext_50,[(1,3,224,224)]),%s).task.pkl" % args.platform,
        "dataset/network_info/((bert_base,[(1,128)]),%s).task.pkl" % args.platform,
        "dataset/network_info/((bert_tiny,[(1,128)]),%s).task.pkl" % args.platform,
    ]

    best_latency_total = 0
    top_latency_totals = [0 for _ in top_ks]

    all_abs_errors = []
    all_uncertainties = []

    for file in files:
        tasks, task_weights = pickle.load(open(file, "rb"))
        latencies = [0] * len(top_ks)
        best_latency = 0

        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                print("error task.workload_key not in pred_a_dataset_dict")
                continue

            pred_a_dataset(pred_a_dataset_dict[task.workload_key], task_pred_dict, model)
            rank_score, pred_mean, pred_std, min_latency, labels = task_pred_dict[task.workload_key]

            real_values = labels[np.argsort(-rank_score)]
            real_latency = min_latency / np.maximum(real_values, 1e-5)

            for i, top_k in enumerate(top_ks):
                latencies[i] += np.min(real_latency[:top_k]) * weight

            best_latency += min_latency * weight

            all_abs_errors.extend(np.abs(pred_mean - labels).tolist())
            all_uncertainties.extend(pred_std.tolist())

        print(f"top 1 score: {best_latency / latencies[0]}")
        print(f"top 5 score: {best_latency / latencies[1]}")

        best_latency_total += best_latency
        for i in range(len(top_ks)):
            top_latency_totals[i] += latencies[i]

    for i, top_k in enumerate(top_ks):
        print(f"average top {top_k} score is {best_latency_total / top_latency_totals[i]}")


    if len(all_abs_errors) > 1:
        uncertainty_arr = np.array(all_uncertainties)
        abs_error_arr = np.array(all_abs_errors)
        corr = np.corrcoef(uncertainty_arr, abs_error_arr)[0, 1]
        print(f"uncertainty error pearson is {corr}")
        print("uncertainty group analysis:")
        order = np.argsort(uncertainty_arr)
        uncertainty_arr = uncertainty_arr[order]
        abs_error_arr = abs_error_arr[order]
        for group_idx, index_group in enumerate(np.array_split(np.arange(len(uncertainty_arr)), 5)):
            group_uncertainty = uncertainty_arr[index_group]
            group_abs_error = abs_error_arr[index_group]
            print(
                "group %d uncertainty %.6f abs_error %.6f count %d"
                % (
                    group_idx,
                    group_uncertainty.mean(),
                    group_abs_error.mean(),
                    len(index_group),
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--test_dataset_name", type=str, default="tlp_dataset_platinum_8272_2308_test.pkl")
    parser.add_argument("--load_name", type=str, default="tlp_i7/tlp_model_0.pkl")
    parser.add_argument("--platform", type=str, default="llvm")
    parser.add_argument("--mc_samples", type=int, default=8)
    parser.add_argument("--uncertainty_lambda", type=float, default=0.0)
    args = parser.parse_args()
    print(args)

    device = args.cuda

    with open(args.test_dataset_name, "rb") as f:
        test_datasets = pickle.load(f)

    eval_model(args.load_name)