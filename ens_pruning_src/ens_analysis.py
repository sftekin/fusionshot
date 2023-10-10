import os
import itertools
import time
import copy
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
import numpy as np
import pandas as pd
from helper import load_predictions, calculate_errors, min_max
from diversity_stats import calc_stat_matrices, calc_generalized_div
from ensemble_methods import ensemble_methods
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import multiprocessing
from tqdm import tqdm


def test_superset(best_combinations, input_comb):
    flag = False
    for comb in best_combinations:
        if set(comb).issubset(input_comb):
            flag = True
    return flag


def get_new_set(best_comb, combinations):
    new_combinations = []
    for comb in combinations:
        if test_superset(best_combinations=best_comb, input_comb=comb):
            new_combinations.append(comb)
    combinations = new_combinations
    return combinations


def get_best_comb(foc_div_dict, prune_per, weights):
    """
    returns the combinations in descending order of scores.

    :param foc_div_dict:
    :param prune_per:
    :param weights:
    :return:
    """
    comb_names = list(foc_div_dict.keys())
    if len(comb_names) == 1:
        return comb_names

    all_ens_arr = np.array(list(foc_div_dict.values()))
    norm_arr = np.apply_along_axis(arr=all_ens_arr, func1d=min_max, axis=0)
    scores = norm_arr[:, 0] * weights[0] + norm_arr[:, 1] * weights[1]

    num_k = int(len(scores) * prune_per)
    if num_k <= 1:
        num_k = 1
    sort_idx = np.argsort(scores)[::-1][:num_k]
    best_combs = [comb_names[i] for i in sort_idx]

    return best_combs


def get_comb_stats(focal_div_dict, combinations, all_error_arr, pred_arr, all_error_dict, ens_size):

    for i in tqdm(range(len(combinations))):
        # select ensemble set
        set_bin_arr = all_error_arr[:, combinations[i]]
        set_preds = pred_arr[:, combinations[i]]

        # calc focal diversity of ensemble
        focal_div = 0
        for focal_idx in combinations[i]:
            focal_arr = all_error_dict[model_names[focal_idx]]
            neg_idx = np.where(focal_arr == 0)[0]
            neg_samp_arr = set_bin_arr[neg_idx]
            focal_div += calc_generalized_div(neg_samp_arr)
        focal_div /= ens_size


        # calculate accuracy of ensemble
        ens_pred = ensemble_methods["voting"](set_preds, method="plurality", n_way=n_way, n_query=n_query)
        y = np.tile(np.repeat(range(n_way), n_query), len(ens_pred))
        ens_pred_flatten = ens_pred.flatten()
        all_acc = np.mean(y == ens_pred_flatten) * 100
        # add to collection
        focal_div_dict[combinations[i]] = [focal_div, all_acc]


def prune_ensemble_sets(model_names, ds_name, prune_per, weights, dataset):
    predictions, pred_arr = load_predictions(model_names=model_names, n_query=n_query, n_shot=n_shot,
                                             n_way=n_way, class_name=ds_name, dataset=dataset)
    all_error_dict, all_error_arr = calculate_errors(predictions, pred_arr, n_query=n_query, n_way=n_way)

    ens_dict = {}
    best_comb = []
    ens_sizes = np.arange(2, len(model_names))
    for j, ens_size in enumerate(ens_sizes):
        print(f"Ens size: {j} / {len(ens_sizes)}")

        # prune the combinations based on best combinations
        combinations = list(itertools.combinations(range(len(model_names)), ens_size))
        # if best_comb:
        #     combinations = get_new_set(best_comb, combinations)

        manager = multiprocessing.Manager()
        focal_div_dict = manager.dict()

        jobs = []
        step_size = len(combinations) // num_cpu
        step_size = 1 if step_size == 0 else step_size
        for i in range(0, len(combinations), step_size):
            comb = combinations[i:i+step_size]
            p = multiprocessing.Process(target=get_comb_stats, args=(focal_div_dict, comb, all_error_arr,
                                                                     pred_arr, all_error_dict, ens_size))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        # add to ens collection
        ens_dict[ens_size] = {k: v for k, v in focal_div_dict.items()}

        # # find the best combination
        # best_comb = get_best_comb(focal_div_dict, prune_per, weights)

    weights_str = '-'.join([str(w) for w in weights])
    with open(f"ens_dict_{ds_name}_{weights_str}.pkl", "wb") as f:
        pkl.dump(ens_dict, f)

    return ens_dict


def plot_acc_div(ens_dict, ds_name, save_extension=""):
    # plot acc vs div
    ens_fig_dir = os.path.join(figures_dir, "ens_analysis")
    if not os.path.exists(ens_fig_dir):
        os.makedirs(ens_fig_dir)

    normalize = mcolors.Normalize(vmin=2, vmax=max(ens_dict.keys()))
    colormap = cm.Spectral
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, focal_div in ens_dict.items():
        stats_arr = np.array(list(focal_div.values()))
        ax.scatter(stats_arr[:, 0], stats_arr[:, 1], s=100, alpha=0.9, color=np.array(colormap(normalize(s))))

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(list(ens_dict.keys()))
    cbar = plt.colorbar(scalarmappaple)
    cbar.ax.tick_params(labelsize=14)

    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.set_xlabel("Focal Diversity", fontsize=16)
    ax.set_ylabel("Plurality Voting Accuracy (%)", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)

    # plt.suptitle(f"Ensemble Analysis, dataset={ds_name}")
    save_path = os.path.join(ens_fig_dir, f"{ds_name}_acc_div_{save_extension}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")


def select_best_sets(ens_dict, weights, prune_per, ds_name):
    all_ens = {}
    for ens_set, foc_div_dicts in ens_dict.items():
        for k, v in foc_div_dicts.items():
            all_ens[k] = v

    all_ens_arr = np.array(list(all_ens.values()))
    norm_arr = np.apply_along_axis(arr=all_ens_arr, func1d=min_max, axis=0)
    scores = norm_arr[:, 0] * weights[0] + norm_arr[:, 1] * weights[1]

    num_k = int(len(scores) * prune_per)
    if num_k <= 1:
        num_k = 1
    sort_idx = np.argsort(scores)[::-1][:num_k]
    decision_arr = all_ens_arr[sort_idx]

    comb_ids = [list(all_ens.keys())[i] for i in sort_idx]
    for comb in comb_ids:
        model_n = [model_names[j] for j in comb]
        print(f"{(all_ens[comb][0] + all_ens[comb][1]/100)/2}\t{all_ens[comb]}\t{model_n}")

    ens_fig_dir = os.path.join(figures_dir, "ens_analysis")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_ens_arr[:, 0], all_ens_arr[:, 1], s=100, c="navy", label="pruned")
    ax.scatter(decision_arr[:, 0], decision_arr[:, 1], s=100, c="r", label="selected")
    ax.set_xlabel("Focal Diversity", fontsize=16)
    ax.set_ylabel("Plurality Voting Accuracy (%)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    # ax.set_title(f"Selecting based on diversity")
    ax.legend(loc=4, fontsize=16)
    ax.grid()

    save_path = os.path.join(ens_fig_dir, f"{ds_name}_pruned_ens_{num_k}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved under {save_path}")

    return comb_ids


def select_best_sets_each_size(ens_dict, weights, prune_per, ds_name):
    all_ens = {}
    for ens_set, foc_div_dicts in ens_dict.items():
        for k, v in foc_div_dicts.items():
            all_ens[k] = v
    all_ens_arr = np.array(list(all_ens.values()))

    decision_arr = []
    comb_ids = []
    for ens_set, foc_div_dict in ens_dict.items():
        comb = get_best_comb(foc_div_dict, prune_per=0.001, weights=[0.5, 0.5])
        for c in comb:
            decision_arr.append(all_ens[c])
            comb_ids.append(c)
            print(c)
    decision_arr = np.stack(decision_arr)

    ens_fig_dir = os.path.join(figures_dir, "ens_analysis")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_ens_arr[:, 0], all_ens_arr[:, 1], c="navy", label="pruned")

    normalize = mcolors.Normalize(vmin=2, vmax=len(ens_dict.keys()))
    colormap = cm.Spectral
    for i in range(len(decision_arr)):
        ax.scatter(decision_arr[i, 0], decision_arr[i, 1], color=np.array(colormap(normalize(i))))

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(range(2, len(ens_dict.keys())))
    plt.colorbar(scalarmappaple)

    ax.set_xlabel("focal diversity")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Selecting based on diversity")
    ax.legend(loc=4)
    ax.grid()

    save_path = os.path.join(ens_fig_dir, f"{ds_name}_pruned_ens_best_each_size.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved under {save_path}")

    return comb_ids


if __name__ == '__main__':
    n_query = 15
    n_way = 5
    n_shot = 1
    figures_dir = "figures"
    ds_name = "val"
    weights = [0.5, 0.5]
    prune_per = 1
    dataset = "miniImagenet"
    num_cpu = multiprocessing.cpu_count()

    model_back_bones = ["Conv4", "ResNet18"]
    method_names = ["matchingnet", "protonet", "maml_approx", "relationnet"]

    model_names = []
    for m in method_names:
        for bb in model_back_bones:
            model_names.append(f"{m}_{bb}")

    # Add simple shot models to the pool
    ss_back_bones = ["DenseNet121", "WideRes"]
    ss_methods = [f"simpleshot_{bb}" for bb in ss_back_bones]
    model_names += ss_methods

    # Add DeepEMD also
    model_names.append("DeepEMD")

    # perform ensemble pruning
    # print(model_names)
    # start_time = time.time()
    # prune_ensemble_sets(model_names=model_names, ds_name=ds_name,
    #                     weights=weights, prune_per=prune_per, dataset=dataset)
    # end_time = time.time()

    weights_str = '-'.join([str(w) for w in weights])
    with open(f"ens_dict_{ds_name}_{weights_str}.pkl", "rb") as f:
        ens_dict = pkl.load(f)

    # print(f"M={len(model_names)}, took {end_time - start_time} seconds")

    # plot results
    weight_str = [str(i) for i in weights]
    plot_acc_div(ens_dict, ds_name, save_extension="-".join(weight_str))

    # select the best from the results
    comb_ids = select_best_sets(ens_dict, weights=[0., 1], prune_per=0.001, ds_name=ds_name)

    # comb_ids = select_best_sets_each_size(ens_dict, weights=[0.5, 0.5], prune_per=0.01, ds_name=ds_name)
    # with open(f"best_comb_ids_{ds_name}.pkl", "wb") as f:
    #     pkl.dump(comb_ids, f)

    # scores_dict = {}
    # acc_list = []
    # foc_list = []
    # for ens_size, comb_dict in ens_scores.items():
    #     best_comb = get_best_comb(comb_dict, prune_per=0.01, weights=[0, 1])
    #     acc_list.append(ens_scores[ens_size][best_comb[0]][1])
    #     foc_list.append(ens_scores[ens_size][best_comb[0]][0])
    #
    # plt.figure()
    # plt.plot(np.arange(2, len(acc_list) + 2), foc_list, label="div")
    # plt.show()
