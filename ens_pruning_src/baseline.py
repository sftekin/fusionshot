from helper import load_predictions, calculate_errors
from ens_pruning_src.diversity_stats import calc_stat_matrices
import argparse
import numpy as np
import itertools
import pickle as pkl


def prune_ensemble_sets(model_names, class_name, dataset, n_shot, n_way, n_query):
    predictions, pred_arr = load_predictions(model_names=model_names, n_query=n_query, n_shot=n_shot,
                                             n_way=n_way, class_name=class_name, dataset=dataset)
    all_error_dict, all_error_arr = calculate_errors(predictions, pred_arr, n_query=n_query, n_way=n_way)
    stat_dict = calc_stat_matrices(all_error_dict)
    kappa_val = stat_dict["kappa_statistics"]
    kappa_dict = {}
    for col in kappa_val:
        for row, val in kappa_val[col].items():
            key = (model_names.index(col), model_names.index(row))
            kappa_dict[key] = val

    ens_sizes = np.arange(2, len(model_names) + 1)
    div_dict = {}
    for j, ens_size in enumerate(ens_sizes):
        combinations = list(itertools.combinations(range(len(model_names)), ens_size))
        for comb in combinations:
            mean_kappa = 0
            if len(comb) == 2:
                mean_kappa = kappa_dict[comb]
            else:
                for i, sub_comb in enumerate(itertools.combinations(comb, 2)):
                    mean_kappa += kappa_dict[sub_comb]
                mean_kappa /= (i + 1)
            div_dict[comb] = mean_kappa

    div_val = list(div_dict.values())
    div_key = list(div_dict.keys())
    idx = np.argsort(list(div_dict.values()))
    comb_names = []
    comb_scores = []
    for i in idx:
        # print([model_names[j] for j in div_key[i]], div_val[i])
        comb_names.append(div_key[i])
        comb_scores.append(div_val[i])
        print(div_key[i], div_val[i])
    print()

    return comb_names, comb_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='focal diversity pruning')
    parser.add_argument('--dataset_name', default="miniImagenet", choices=["CUB", "miniImagenet"])
    parser.add_argument("--n_query", default=15, type=int)
    parser.add_argument("--n_way", default=5, type=int)
    parser.add_argument("--n_shot", default=1, type=int)
    parser.add_argument('--model_names', nargs='+',
                        help='Model name and backbone e.g. protonet_ResNet18', required=True)
    args = parser.parse_args()
    cls_name = "novel"

    m_names = ["matchingnet_Conv6", "matchingnet_ResNet18", "protonet_Conv6", "protonet_ResNet18",
               "relationnet_Conv6", "relationnet_ResNet18", "maml_approx_Conv6",
               "maml_approx_ResNet18", "simpleshot_DenseNet121", "DeepEMD"]

    kappa_list = prune_ensemble_sets(
        model_names=m_names,
        class_name=cls_name,
        dataset=args.dataset_name,
        n_shot=args.n_shot,
        n_way=args.n_way,
        n_query=args.n_query)

    with open(f"ens_dict_{cls_name}.pkl", "rb") as f:
        ens_dict = pkl.load(f)

    comb_names = []
    scores = []
    weights = np.array([0.5, 0.5])
    for ens_size, comb_dict in ens_dict.items():
        for comb, val in comb_dict.items():
            comb_names.append(comb)
            s = np.array([val[0], val[1] / 100])
            scores.append(np.sum(s * weights))
    comb_names = np.array(comb_names)
    scores = np.array(scores)
    idx = np.argsort(scores)
    comb_names = comb_names[idx]
    scores = scores[idx]

    for comb, score in zip(comb_names[-10:], scores[-10:]):
        print([m_names[k] for k in comb])
        kp_idx = kappa_list[0].index(comb)
        print(comb, score, f"kappa rank {kp_idx} kappa score {kappa_list[1][kp_idx]}")

    print()
