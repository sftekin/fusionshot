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


def run(model_names):
    sv_path = f"{CUR_PATH}/ens_checkpoints/{dataset}/{'-'.join(model_names)}_{n_way}way_{n_shot}shot"
    sv_path = modify_save_path(sv_path)
    outfile = os.path.join(sv_path, f'best_model.tar')

    model = MLP(len(model_names) * 5, [100, 100], 5)
    tmp = torch.load(outfile, map_location=device)
    model.load_state_dict(tmp["state"])

    novel_logits = load_logits(model_names, dataset=dataset, class_type="novel", nway=n_way, nshot=n_shot)
    novel_data = create_data(logits=novel_logits, n_query=n_query, n_way=n_way, shuffle=False)

    model.to(device)
    novel_loader = DataLoader(novel_data, batch_size=64, shuffle=False)
    acc_mean, acc_std, logits, labels = test_loop(model, novel_loader, ret_logit=True, device=device)
    ens_logits = logits[:, -5:]
    ens_preds = ens_logits.argmax(axis=1)
    conf = 1.96 * acc_std / np.sqrt(len(logits))
    print(f"Ensemble acc: {acc_mean:.2f} +- {conf:.2f}")


if __name__ == '__main__':
    n_shot = 1
    n_way = 5
    n_query = 15
    dataset = "miniImagenet"
    device = "cpu"

    all_names = ['protonet_ResNet18', "maml_approx_ResNet18", 'simpleshot_ResNet18', 'DeepEMD']
    run(model_names=all_names)
