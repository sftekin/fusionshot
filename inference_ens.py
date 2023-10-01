import os
import pickle as pkl
import glob
import itertools
import numpy as np
import scipy
import torch.nn
import torch.nn as nn
from torch.utils.data import DataLoader
from train_ensemble import MLP, modify_save_path, load_logits, create_data, test_loop
from ensemble_methods import voting
import seaborn as sns
import matplotlib.pyplot as plt


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

    # method_names = ["simpleshot"]
    # back_bones = ["ResNet18", "WideRes", "DenseNet121"]

    # method_names = ["protonet", "simpleshot", "DeepEMD"]
    # back_bones = ["ResNet18"]
    #
    # model_names = []
    # for method_n in method_names:
    #     for bb in back_bones:
    #         if method_n == "DeepEMD":
    #             model_names.append(method_n)
    #         else:
    #             model_names.append(f"{method_n}_{bb}")

    # sv_path = f"ens_checkpoints/{dataset}/{'-'.join(model_names)}_{n_way}way_{n_shot}shot"
    sv_path = f"ens_checkpoints/{dataset}/one_shot/{'-'.join(model_names)}"
    sv_path = modify_save_path(sv_path)
    outfile = os.path.join(sv_path, f'best_model.tar')

    model = MLP(len(model_names) * 5, [100, 100], 5)
    tmp = torch.load(outfile, map_location=device)
    model.load_state_dict(tmp["state"])

    novel_logits = load_logits(model_names, dataset=dataset, class_type="novel",
                               perform_norm=True, nway=n_way, nshot=n_shot)
    novel_data = create_data(logits=novel_logits, n_query=n_query, n_way=n_way, shuffle=False)

    labels = novel_data[:, -1]
    model_preds = [novel_logits[i].argmax(axis=1) for i in range(len(model_names))]
    base_errors = np.stack([(mod_pred == labels).astype(int) for mod_pred in model_preds], axis=1)
    model_preds = np.stack(model_preds, axis=1)

    model.to(device)
    novel_loader = DataLoader(novel_data, batch_size=64, shuffle=False)
    acc_mean, acc_std, logits, labels = test_loop(model, novel_loader, ret_logit=True, device=device)
    ens_logits = logits[:, -5:]
    ens_preds = ens_logits.argmax(axis=1)
    ens_errors = (ens_preds == labels).astype(int)
    conf = 1.96 * acc_std / np.sqrt(len(logits))
    print(f"Ensemble acc: {acc_mean:.2f} +- {conf:.2f}")
    ens_mean, ens_conf = acc_mean, conf

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

    return [ens_mean, vot_mean]


    # model_stats = {
    #     "model_names": model_names,
    #     "base_errors": base_errors,
    #     "base_logits": np.concatenate(novel_logits, axis=1),
    #     "ensemble_errors": ens_errors,
    #     "ens_logits": ens_logits,
    #     "ens_stats": {}
    # }
    # print(f"Out of {len(base_errors)} episodes:")
    # for i in range(1, len(model_names) + 1):
    #     for comb in itertools.combinations(range(len(model_names)), i):
    #         corr_arr = np.zeros(len(model_names))
    #         for k in comb:
    #             corr_arr[k] = 1
    #         model_correct = (base_errors == corr_arr).all(axis=1)
    #         model_str = "-".join(model_names[k] for k in comb)
    #         print(f"Only '{model_str}' made {model_correct.sum()} correct decisions "
    #               f"and ensemble made {ens_errors[model_correct].sum()} "
    #               f"correct decisions with '{model_str}'")
    #         model_stats[model_str] = model_correct
    #         model_stats["ens_stats"][model_str] = np.argwhere(model_correct & ens_errors.astype(bool))
    #
    # only_ens_made_correct = ens_errors[base_errors.sum(axis=1) == 0]
    # print(f"Only ensemble made {only_ens_made_correct.sum()} correct decisions.")
    # model_stats["ens_stats"]["ensemble"] = np.argwhere((base_errors.sum(axis=1) == 0) & ens_errors.astype(bool))

    # plot_improvement(model_stats, method_names)
    # plot_ensemble_stats(model_stats)

    # with open("results/inference_stats.pkl", "wb") as f:
    #     pkl.dump(model_stats, f)


if __name__ == '__main__':
    n_shot = 1
    n_way = 5
    n_query = 15
    dataset = "miniImagenet"
    device = "cpu"


    # *** Baseline Comparison model Exps ***
    all_names = ["matchingnet_ResNet18", "protonet_ResNet18", "maml_approx_ResNet18",
                 "relationnet_ResNet18", "simpleshot_ResNet18", "DeepEMD"]
    ens_sizes = np.arange(2, len(all_names) + 1)
    comb2scores = {}
    for ens_size in ens_sizes:
        combinations = itertools.combinations(all_names, ens_size)
        for comb in combinations:
            print(comb)
            comb2scores["".join([str(all_names.index(c_name)) for c_name in comb])] = run(model_names=comb)

    comb_names = list(comb2scores.keys())
    comb_scores = np.array(list(comb2scores.values()))
    x_axis = np.arange(len(comb_scores))
    idx = np.argsort(comb_scores[:, 0])
    sorted_names = [comb_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_axis, comb_scores[idx, 0], '->', label="FusionShot", lw=2, markersize=10)
    ax.plot(x_axis, comb_scores[idx, 1], '-<', label="Plurality Voting", lw=2, markersize=10)
    ax.legend(frameon=True, fontsize=18)
    ax.set_xlabel("Ensemble sets ", fontsize=18)
    ax.set_ylabel("Accuracy (%)", fontsize=18)
    ax.set_title("FusionShot vs Plurality Voting", fontsize=18)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=16)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlim(x_axis[0], x_axis[-1])
    ax.xaxis.grid(color='gray', lw=0.5, alpha=0.5)
    ax.yaxis.grid(color='gray', lw=0.5, alpha=0.5)
    plt.savefig("figures/ens_analysis/voting.png", dpi=200, bbox_inches="tight")
    plt.show()

    print()
