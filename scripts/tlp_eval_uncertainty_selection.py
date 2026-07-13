import argparse
import pickle
from collections import OrderedDict

import numpy as np
import torch
from tlp_train_mc_dropout import *


top_ks = [1, 5, 10, 20]


def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predict_a_dataset(datas, task_pred_dict, model):
    file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = datas

    if workloadkey in task_pred_dict:
        return task_pred_dict[workloadkey]

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
        for _ in range(args.mc_samples):
            model.eval()
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
    pred_mean = preds_stack.mean(dim=0).numpy()
    pred_std = preds_stack.std(dim=0, unbiased=False).numpy()
    labels = labels_all.numpy()
    min_latency = test_loader.min_latency.min().numpy()

    task_pred_dict[workloadkey] = (pred_mean, pred_std, min_latency, labels)
    return task_pred_dict[workloadkey]


def append_remaining(order_prefix, mean_order):
    used = set(order_prefix.tolist())
    rest = [idx for idx in mean_order.tolist() if idx not in used]
    if rest:
        return np.concatenate([order_prefix, np.array(rest, dtype=mean_order.dtype)])
    return order_prefix


def build_strategy_orders(pred_mean, pred_std):
    orders = OrderedDict()
    mean_order = np.argsort(-pred_mean)
    orders["mean"] = mean_order

    for penalty_lambda in args.penalty_lambdas:
        name = "global_penalty_%s" % penalty_lambda
        orders[name] = np.argsort(-(pred_mean - penalty_lambda * pred_std))

    for requested_pool_size in args.pool_sizes:
        pool_size = min(requested_pool_size, len(mean_order))
        pool = mean_order[:pool_size]
        low_unc_pool = pool[np.argsort(pred_std[pool])]
        orders["pool%d_low_unc" % requested_pool_size] = append_remaining(low_unc_pool, mean_order)

        for penalty_lambda in args.pool_lambdas:
            pool_score = pred_mean[pool] - penalty_lambda * pred_std[pool]
            pool_penalty = pool[np.argsort(-pool_score)]
            name = "pool%d_penalty_%s" % (requested_pool_size, penalty_lambda)
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


def eval_model(model_file):
    with open(model_file, "rb") as f:
        loaded_model = pickle.load(f)
    model = loaded_model.module if hasattr(loaded_model, "module") else loaded_model
    model = model.to(device)

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

    task_pred_dict = {}
    strategy_latencies = OrderedDict()
    best_latency_total = 0
    all_abs_errors = []
    all_uncertainties = []

    for network_file in network_files:
        tasks, task_weights = pickle.load(open(network_file, "rb"))
        best_latency = 0

        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                print("error task.workload_key not in pred_a_dataset_dict")
                continue

            pred_mean, pred_std, min_latency, labels = predict_a_dataset(
                pred_a_dataset_dict[task.workload_key],
                task_pred_dict,
                model,
            )

            if not strategy_latencies:
                for strategy_name in build_strategy_orders(pred_mean, pred_std).keys():
                    strategy_latencies[strategy_name] = [0 for _ in top_ks]

            for strategy_name, order in build_strategy_orders(pred_mean, pred_std).items():
                real_values = labels[order]
                real_latency = min_latency / np.maximum(real_values, 1e-5)
                for i, top_k in enumerate(top_ks):
                    strategy_latencies[strategy_name][i] += np.min(real_latency[:top_k]) * weight

            best_latency += min_latency * weight
            all_abs_errors.extend(np.abs(pred_mean - labels).tolist())
            all_uncertainties.extend(pred_std.tolist())

        best_latency_total += best_latency

    print("strategy average top-k scores:")
    print("strategy top1 top5 top10 top20")
    for strategy_name, latencies in strategy_latencies.items():
        scores = [best_latency_total / latency for latency in latencies]
        print(
            "%s %.6f %.6f %.6f %.6f"
            % (strategy_name, scores[0], scores[1], scores[2], scores[3])
        )

    if len(all_abs_errors) > 1:
        uncertainty_arr = np.array(all_uncertainties)
        abs_error_arr = np.array(all_abs_errors)
        corr = np.corrcoef(uncertainty_arr, abs_error_arr)[0, 1]
        print("uncertainty error pearson %.6f" % corr)
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
    parser.add_argument("--test_dataset_name", type=str, default="tlp_dataset_platinum_8272_2308_test.pkl")
    parser.add_argument("--load_name", type=str, default="runs/tlp_mc_dropout_2308/tlp_model_19.pkl")
    parser.add_argument("--platform", type=str, default="llvm")
    parser.add_argument("--mc_samples", type=int, default=8)
    parser.add_argument("--penalty_lambdas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--pool_lambdas", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--pool_sizes", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--drop_uncertain_ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = args.cuda

    with open(args.test_dataset_name, "rb") as f:
        test_datasets = pickle.load(f)

    eval_model(args.load_name)
