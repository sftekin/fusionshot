import os
import pickle as pkl
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from helper import load_predictions, calculate_errors
from train_ensemble import modify_save_path


def get_individual_scores(class_name, ds):
    predictions, pred_arr = load_predictions(model_names, n_query, n_way, n_shot=n_shot,
                                             class_name=class_name, dataset=ds)
    _, error_arr = calculate_errors(predictions, pred_arr, n_query=n_query, n_way=n_way)
    scores = error_arr.mean(0)
    return scores


def plot_scatter(idx2names):
    idx2scores = {}
    for idx, name in idx2names.items():
        path = os.path.join("ens_checkpoints", dataset, f"{name}", "results.tar")
        if not os.path.exists(path):
            path = modify_save_path(path)
        result_dict = torch.load(path, map_location=device)
        score = [result_dict['val_acc'], result_dict['test_acc']]
        idx2scores[idx] = score

    ens_size_dict = {len(comb): [] for comb in idx2scores.keys()}
    ens_best_scores = {len(comb): [0, ""] for comb in idx2scores.keys()}
    ens_worst_scores = {len(comb): [99, ""] for comb in idx2scores.keys()}
    for comb, score in idx2scores.items():
        ens_size_dict[len(comb)].append(score[1])
        if score[1] > ens_best_scores[len(comb)][0]:
            ens_best_scores[len(comb)] = [score[1], "".join([str(n) for n in comb])]
        if score[1] < ens_worst_scores[len(comb)][0]:
            ens_worst_scores[len(comb)] = [score[1], "".join([str(n) for n in comb])]

    x_axis, y_axis = [], []
    for size, score_list in ens_size_dict.items():
        x_axis.append(np.repeat(size, len(score_list)))
        y_axis.append(score_list)
    x_axis = np.concatenate(x_axis)
    y_axis = np.concatenate(y_axis)

    fig, ax = plt.subplots()
    ax.scatter(x_axis, y_axis, color="b", s=150, label="Ensemble Set Score")
    x_ticks = np.arange(2, len(model_names)+1)
    ax.set_xticks(x_ticks)
    y_min, y_max = np.floor(y_axis.min()), np.ceil(y_axis.max())

    # novel_scores = get_individual_scores(class_name="novel", ds=dataset)
    # best_novel_id = np.argmax(novel_scores)
    # ax.plot(np.arange(0, 10), np.repeat(novel_scores[best_novel_id] * 100,
    #                           10), "--", c="k", lw=3,
    #         label=f"{model_names[best_novel_id]}")
    ax.legend(loc="lower right", fontsize=16)
    ax.set_xlabel("Ensemble Set Size",  fontsize=16)
    ax.set_ylabel("Accuracy (%)",  fontsize=16)
    # ax.set_yticks(np.linspace(y_min, y_max+1, 10))
    # ax.set_ylim(y_min, y_max+1)
    ax.set_xlim(1.8, len(model_names) + 0.2)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    for x_axis, (y_axis, txt) in ens_best_scores.items():
        ax.text(x_axis, y_axis+0.2, txt, ha='center', va='bottom', color="g", fontweight="bold",  fontsize=14)

    for x_axis, (y_axis, txt) in ens_worst_scores.items():
        if x_axis == len(model_names):
            continue
        ax.text(x_axis, y_axis+0.2, txt, ha='center', va='bottom', color="r", fontweight="bold", fontsize=14)

    # props = dict(boxstyle='round', facecolor='azure', alpha=0.8)
    # textstr = "\n".join([f"{i}. {mn}" for i, mn in enumerate(model_names)])
    # ax.text(0.98, 0.2, textstr, transform=ax.transAxes, fontsize=14,
    #         horizontalalignment="right", bbox=props)

    ax.tick_params(axis="both", labelsize=16)
    file_name = "-".join(model_names)
    plt.savefig(f"figures/scatter_{file_name}_{n_way}way_{n_shot}shot.png", bbox_inches="tight", dpi=200)


def plot_line(idx2names):
    idx2scores = {}
    for idx, name in idx2names.items():
        path = os.path.join("ens_checkpoints", dataset, f"{name}", "results.tar")
        if not os.path.exists(path):
            path = modify_save_path(path)
        result_dict = torch.load(path, map_location=device)
        score = [result_dict['val_acc'], result_dict['test_acc']]
        idx2scores[idx] = score

    scores = np.array(list(idx2scores.values()))
    sorted_idx = np.argsort(scores[:, 1])
    sorted_names = [list(idx2scores.keys())[idx] for idx in sorted_idx]

    # plt.style.use("seaborn")
    fig, ax = plt.subplots(figsize=(30, 5))
    x_axis = np.arange(0, len(sorted_names))
    bar_width = 0.45
    # ax.scatter(x_axis, scores[sorted_idx, 0],  label="Ensemble val")
    ax.plot(x_axis, scores[sorted_idx, 1], "-o", c="r",
            label="Ensemble novel", lw=2, markersize=8)

    # val_scores = get_individual_scores(set_name="val")
    novel_scores = get_individual_scores(class_name="novel", ds=dataset)
    best_novel_id = np.argmax(novel_scores)
    # best_val_id = np.argmax(val_scores)

    # ax.plot(np.repeat(val_scores[best_val_id] * 100, len(x_axis)), "--", c="r", lw=3,
    #         label=f"{model_names[best_val_id]} val")
    ax.plot(np.repeat(novel_scores[best_novel_id] * 100, len(x_axis)), "--", c="k", lw=1.5,
            label=f"{model_names[best_novel_id]} novel")

    if novel_scores[best_novel_id] * 100 < scores.min():
        y_min_val = novel_scores[best_novel_id] * 100
    else:
        y_min_val = np.floor(scores.min())
    y_max_val = np.ceil(scores.max())

    vert_line_idx = scores[sorted_idx, 1] > novel_scores[best_novel_id] * 100
    vert_line_x = np.repeat(x_axis[vert_line_idx][0], len(x_axis))
    vert_line_y = np.linspace(y_min_val, y_max_val, len(x_axis))
    ax.plot(vert_line_x, vert_line_y, "--", c="g", lw=1.5, label="lower bound for Ensemble Sets")

    # ax.set_yticks(np.linspace(np.floor(y_min_val),
    #                           np.ceil(scores.max()), 10), fontsize=18)
    ax.set_ylim(np.floor(y_min_val), np.ceil(scores.max()))
    ax.set_xlim(x_axis.min() - 0.5, x_axis.max() + 0.5)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(["".join([str(n) for n in name_list]) for name_list in sorted_names], rotation=90, fontsize=16)
    ax.set_xlabel("Ensemble sets ", fontsize=18)
    ax.set_ylabel("Accuracy (%)", fontsize=18)
    props = dict(boxstyle='round', facecolor='green', alpha=0.2)
    textstr = "\n".join([f"{i}. {mn}" for i, mn in enumerate(model_names)])
    ax.text(0.95, 0.35, textstr, transform=ax.transAxes, fontsize=18,
            horizontalalignment="right", bbox=props)
    ax.legend(frameon=True, fontsize=18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=18)

    ax.xaxis.grid(color='gray', lw=0.5, alpha=0.5)
    ax.yaxis.grid(color='gray', lw=0.5, alpha=0.5)
    file_name = "-".join(method_names)
    plt.savefig(f"figures/line_{file_name}_score_bars_{n_way}way_{n_shot}shot.png", dpi=200, bbox_inches="tight")


if __name__ == '__main__':
    n_query = 15
    n_way = 5
    n_shot = 5
    dataset = "cross"
    device = "cpu"

    # # *** Simpleshot Exps ***
    # method_names = ["simpleshot"]
    # backbone_names = ["Conv4", "Conv6", "ResNet10", "ResNet18", "ResNet34", "WideRes", "DenseNet121"]
    # model_names = [f"simpleshot_{bb}" for bb in backbone_names]
    #
    # id2n = {}
    # ens_sizes = np.arange(2, len(model_names))
    # for ens_size in ens_sizes:
    #     for comb_idx in itertools.combinations(range(len(model_names)), ens_size):
    #         comb_n = [model_names[i] for i in comb_idx]
    #         id2n[comb_idx] = "-".join(comb_n)

    # # *** Closer Look Exps ***
    # backbone_names = ["Conv4", "Conv6", "ResNet10", "ResNet18", "ResNet34"]
    # method_names = ["matchingnet", "protonet", "maml_approx", "relationnet", "simpleshot"]
    #
    # model_names = []
    # for m in method_names:
    #     model_names += [f"{m}_{bb}" for bb in backbone_names]
    # model_names_arr = np.array(model_names)
    #
    # # # method wise
    # # idx = np.arange(0, len(model_names), 5)
    # # comb_ids = [tuple(idx + i) for i in range(5)]
    #
    # # backbone wise
    # comb_ids = [tuple(np.arange(i * 5, (i * 5) + 5)) for i in range(5)]
    #
    # id2n = {}
    # for comb in comb_ids:
    #     selected_combs = [model_names[i] for i in comb]
    #     id2n[comb] = "-".join(selected_combs)

    # # *** Baseline Comparison model Exps ***
    # model_names = ["matchingnet_ResNet18", "protonet_ResNet18", "maml_approx_ResNet18",
    #                 "relationnet_ResNet18", "simpleshot_ResNet18", "DeepEMD"]
    # method_names = ["matchingnet", "protonet", "maml_approx", "relationnet", "DeepEMD", "simpleshot"]

    # Cross domain comparison

    model_names = ["maml_approx_ResNet18", "matchingnet_ResNet18",
                 "protonet_ResNet18", "relationnet_softmax_ResNet18",
                 "simpleshot_ResNet18"]

    id2n = {}
    ens_sizes = np.arange(2, len(model_names))
    for ens_size in ens_sizes:
        for comb_idx in itertools.combinations(range(len(model_names)), ens_size):
            comb_n = [model_names[i] for i in comb_idx]
            id2n[comb_idx] = "-".join(comb_n)


    # ens_sizes = np.arange(2, len(model_names) + 1)
    # comb_ids = []
    # for ens_size in ens_sizes:
    #     comb_ids += list(itertools.combinations(range(len(model_names)), ens_size))
    #
    # id2n = {}
    # for comb in comb_ids:
    #     selected_combs = [model_names[i] for i in comb]
    #     id2n[comb] = "-".join(selected_combs)

    print(model_names)
    # plot_line(idx2names=id2n)
    plot_scatter(idx2names=id2n)
