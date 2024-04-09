import os
import pickle as pkl
import glob
import itertools
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_ensemble import MLP, modify_save_path, load_logits, create_data, test_loop
from ens_pruning_src.ensemble_methods import voting
import seaborn as sns
import matplotlib.pyplot as plt

CUR_PATH = os.path.dirname(os.path.abspath(__file__))


def plot_ensemble_stats(model_stats):
    model_names = model_stats["model_names"]
    model_comb = []
    for i in range(1, len(model_names) + 1):
        for comb in itertools.combinations(range(3), i):
            model_comb.append("-".join([model_names[c] for c in comb]))

    indv_model_corrects = []
    ens_stats = []
    for mc in model_comb:
        indv_model_corrects.append(model_stats[mc].sum())
        ens_stats.append(model_stats["ens_stats"][mc].shape[0])

    plt.style.use("seaborn")
    fig, ax = plt.subplots()
    x_axis = np.arange(len(model_comb))
    bar1 = ax.bar(x_axis, indv_model_corrects, label="only base model(s) correct", color='tab:blue')
    ax.bar(x_axis, ens_stats, label="ensemble agrees on base model(s)", color='tab:green')
    ax.set_xticks(x_axis)
    ax.set_xticklabels([modify_save_path(mc.replace("_ResNet18", "")) for mc in model_comb], rotation=30)
    for rect in bar1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')

    ax.set_yticks(ens_stats[2:])
    ax.legend()
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.savefig("figures/ens_stats.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_improvement(model_stats, method_names):
    base_errors = model_stats["base_errors"]
    ensemble_errors = model_stats["ensemble_errors"]

    model_names = model_stats["model_names"]
    model_comb_names = []
    model_comb_idx = []
    for i in range(1, len(model_names) + 1):
        for comb in itertools.combinations(range(len(model_names)), i):
            model_comb_names.append("-".join([model_names[c] for c in comb]))
            model_comb_idx.append(comb)

    comb_errors, ens_corrects = [], []
    for comb_idx, comb_n in zip(model_comb_idx, model_comb_names):
        idx = (~base_errors[:, comb_idx].astype(bool)).all(axis=1)
        comb_errors.append(idx.sum())
        ens_corrects.append(ensemble_errors[idx].sum())

    y_tick_labels = []
    for mc in model_comb_names:
        ids = []
        for model_n in mc.split("-"):
            ids.append(str(model_names.index(model_n)))
        y_tick_labels.append("-".join(ids))

    fig, ax = plt.subplots(figsize=(7, 5))
    x_axis = np.arange(0, len(model_comb_names)*3, 3)
    width = 1
    bar1 = ax.bar(x_axis, comb_errors, width, facecolor="none",
                  edgecolor='tomato',  label="Base Model Errors", hatch="x")
    bar2 = ax.bar(x_axis + width, ens_corrects, width, color="seagreen", label="Ensemble Corrects")
    for bar in [bar1, bar2]:
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + width, height, f'{height:.0f}', ha='center', va='bottom', fontsize=12)

    ax.set_ylabel("# Episodes", fontsize=14)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(y_tick_labels)
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.set_ylim(1, 50000)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.legend(loc="lower left")
    ax.set_xlabel("Accurate Model IDs", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    file_name = "-".join(model_names)
    plt.savefig(f"figures/ens_improvements_{file_name}.png", dpi=200, bbox_inches="tight")
    plt.show()


def calc_simple_ens(model_preds, labels, n_query, n_way, model_names):
    import sys
    sys.path.append("..")
    from ens_pruning_src.ensemble_methods import voting

    base_errors = np.stack([(mod_pred == labels).astype(int) for mod_pred in model_preds], axis=1)
    model_preds = np.stack(model_preds, axis=1)

    for i in range(len(model_names)):
        acc = []
        step_size = n_query*n_way
        for j in range(0, len(model_preds), step_size):
            acc.append(np.mean(model_preds[j:j+step_size, i] == labels[j:j+step_size]) * 100)
        acc_mean = np.mean(acc)
        acc_std = np.std(acc)
        print(model_names[i], acc_mean, 1.96 * acc_std / np.sqrt(600))

    voting_preds = voting(model_preds, "plurality", n_query, n_way)
    acc = []
    for i in range(len(voting_preds)):
        y = np.repeat(range(n_way), n_query)
        acc.append(np.mean(voting_preds[i] == y) * 100)
    acc_mean = np.mean(acc)
    acc_std = np.std(acc)
    conf = 1.96 * acc_std / np.sqrt(len(voting_preds))
    print(f"Voting method acc: {acc_mean:.2f} +- {conf:.2f}")
    vot_mean, vot_conf = acc_mean, conf

    return vot_mean, vot_conf


def calc_mean_perf(novel_logits, novel_data):
    from scipy.special import softmax
    mean_pred = np.mean([softmax(novel_logits[i], axis=1) for i in range(len(novel_logits))], axis=0)
    mean_pred = mean_pred.argmax(axis=1)
    y = novel_data[:, -1:].squeeze()
    acc = np.mean(mean_pred == y) * 100
    std = np.std(mean_pred == y)
    conf = 1.96 * std / np.sqrt(len(mean_pred))
    print(f"Simple mean Acc: {acc:.2f} +- {conf:.2f}")
    return acc

def plot_inf_scores(in_scores, model_names):
    model_acc, fusion_mean, pul_acc, mean_acc = in_scores

    fig, ax = plt.subplots()
    x = list(range(len(model_acc)))
    # ax.bar(x, model_acc, label="base models")
    ax.bar(len(x) + 0, pul_acc, label="plurality")
    ax.bar(len(x) + 1, mean_acc, label="average")
    ax.bar(len(x) + 2, fusion_mean, label="fusionshot")
    ax.legend()
    # ax.set_ylim(45, 70)
    plt.show()
    print()


def run(model_names):
    sv_path = f"{CUR_PATH}/ens_checkpoints/{dataset}/{'-'.join(model_names)}_{n_way}way_{n_shot}shot"
    sv_path = modify_save_path(sv_path)
    outfile = os.path.join(sv_path, f'best_model.tar')

    novel_logits = load_logits(model_names, dataset=dataset, class_type="novel", nway=n_way, nshot=n_shot, ep_count=ep_count)
    novel_data = create_data(logits=novel_logits, n_query=n_query, n_way=n_way, shuffle=False, ep_count=ep_count)

    mean_acc = calc_mean_perf(novel_logits, novel_data)

    model_preds = [novel_logits[i].argmax(axis=1) for i in range(len(model_names))]
    labels = novel_data[:, -1]
    model_acc = [np.mean(pred == labels) * 100 for pred in model_preds]
    pul_acc = calc_simple_ens(model_preds, labels, n_query, n_way, model_names)

    model = MLP(len(model_names) * 5, [100, 100], 5)
    tmp = torch.load(outfile)
    model.load_state_dict(tmp["state"])

    model.to(device)
    model.eval()
    novel_loader = DataLoader(novel_data, batch_size=64, shuffle=True)
    fusion_mean, fusion_std = test_loop(model, novel_loader)
    conf = 1.96 * fusion_std / np.sqrt(len(novel_loader))
    print(f"Ensemble acc: {fusion_mean:.2f} +- {conf:.2f}")

    return [model_acc, fusion_mean, pul_acc, mean_acc]


if __name__ == '__main__':
    n_shot = 1
    n_way = 5
    n_query = 15
    dataset = "CUB"
    device = "cuda"
    ep_count=300

    # all_names = ["DeepEMD", 'simpleshot_ResNet18', "protonet_ResNet18", "maml_approx_ResNet18"]
    # all_names = ["maml_approx_ResNet18", 'matchingnet_ResNet18', "protonet_ResNet18", "relationnet_ResNet18"]
    # all_names = ["matchingnet_Conv6", "matchingnet_ResNet18", "protonet_Conv6", "protonet_ResNet18",
    #            "relationnet_Conv6", "relationnet_ResNet18", "maml_approx_Conv6",
    #            "maml_approx_ResNet18", "simpleshot_DenseNet121", "DeepEMD"]
    # all_names = ["DeepEMD", "simpleshot_ResNet18", "easy_ResNet18"]
    all_names = ["simpleshot_WideRes", "simpleshot_DenseNet121", "simpleshot_ResNet18"]
    inference_scores = run(model_names=all_names)
    # plot_inf_scores(inference_scores, all_names)


