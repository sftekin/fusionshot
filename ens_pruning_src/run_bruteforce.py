import os
import itertools
import time
import pickle as pkl
import numpy as np
from tqdm import tqdm
import argparse
from helper import load_predictions, calculate_errors
from diversity_stats import calc_stat_matrices, calc_generalized_div
from ensemble_methods import ensemble_methods
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import multiprocessing

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def get_comb_stats(focal_div_dict, combinations, all_error_arr, pred_arr,
                   all_error_dict, ens_size, model_names, n_way, n_query):
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


def prune_ensemble_sets(model_names, class_name, dataset, n_shot, n_way, n_query):
    predictions, pred_arr = load_predictions(model_names=model_names, n_query=n_query, n_shot=n_shot,
                                             n_way=n_way, class_name=class_name, dataset=dataset)
    all_error_dict, all_error_arr = calculate_errors(predictions, pred_arr, n_query=n_query, n_way=n_way)

    ens_dict = {}
    ens_sizes = np.arange(2, len(model_names) + 1)
    for j, ens_size in enumerate(ens_sizes):
        print(f"Ens size: {j} / {len(ens_sizes)}")

        manager = multiprocessing.Manager()
        focal_div_dict = manager.dict()
        jobs = []
        combinations = list(itertools.combinations(range(len(model_names)), ens_size))
        step_size = len(combinations) // num_cpu
        step_size = 1 if step_size == 0 else step_size
        for i in range(0, len(combinations), step_size):
            comb = combinations[i:i + step_size]
            p = multiprocessing.Process(target=get_comb_stats,
                                        args=(focal_div_dict, comb, all_error_arr,
                                              pred_arr, all_error_dict, ens_size,
                                              model_names, n_way, n_query))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        # add to ens collection
        ens_dict[ens_size] = {k: v for k, v in focal_div_dict.items()}

    with open(f"ens_dict_{class_name}.pkl", "wb") as f:
        pkl.dump(ens_dict, f)

    return ens_dict


def plot_acc_div(ens_dict,  best_name, best_val, save_extension=""):
    # plot acc vs div
    ens_fig_dir = f"{CUR_DIR}/figures/ens_analysis"
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
    x1, x2 = ax.get_xlim()
    ax.plot(np.linspace(x1-0.1, x2+0.1, 100), np.repeat(best_val, 100), '--', label=best_name, c='k', lw=2)
    ax.set_xlim(x1, x2)
    ax.set_ylim(48, 68)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.set_xlabel("Focal Diversity", fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16, loc="lower right")
    plt.show()

    # # plt.suptitle(f"Ensemble Analysis, dataset={ds_name}")
    # save_path = os.path.join(ens_fig_dir, f"acc_div_{save_extension}.png")
    # plt.savefig(save_path, dpi=200, bbox_inches="tight")
    # print(f"Figure is saved under {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='focal diversity pruning')
    parser.add_argument('--dataset_name', default="miniImagenet", choices=["CUB", "miniImagenet"])
    parser.add_argument("--n_query", default=15, type=int)
    parser.add_argument("--n_way", default=5, type=int)
    parser.add_argument("--n_shot", default=1, type=int)
    parser.add_argument('--model_names', nargs='+',
                        help='Model name and backbone e.g. protonet_ResNet18', required=True)
    args = parser.parse_args()
    num_cpu = multiprocessing.cpu_count()
    cls_name = "novel"

    # perform ensemble pruning
    model_n = ["matchingnet_ResNet18", "protonet_ResNet18", "relationnet_ResNet18",
               "maml_approx_ResNet18", "protonet_Conv6", "matchingnet_Conv6",
               "maml_approx_Conv6", "simpleshot_ResNet18", "simpleshot_WideRes", "relationnet_Conv6", "DeepEMD"]
    start_time = time.time()
    prune_ensemble_sets(model_names=model_n,
                        class_name=cls_name,
                        dataset=args.dataset_name,
                        n_shot=args.n_shot,
                        n_way=args.n_way,
                        n_query=args.n_query)
    end_time = time.time()

    with open(f"ens_dict_{cls_name}.pkl", "rb") as f:
        ens_dict = pkl.load(f)

    M = len(model_n)
    # print(f"M={M}, took {end_time - start_time} seconds")

    predictions, pred_arr = load_predictions(model_names=model_n, n_query=args.n_query, n_shot=args.n_shot,
                                             n_way=args.n_way, class_name=cls_name, dataset=args.dataset_name)
    all_error_dict, all_error_arr = calculate_errors(predictions, pred_arr, n_query=args.n_query, n_way=args.n_way)

    best_model_n, best_model_v = "", 0
    for k, v in all_error_dict.items():
        acc = np.mean(v) * 100
        if acc >= best_model_v:
            best_model_v = acc
            best_model_n = k
        print(k, acc)

    # plot results
    plot_acc_div(ens_dict, best_model_n, best_model_v, save_extension=f"{M}")
