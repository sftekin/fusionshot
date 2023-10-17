import pygad
import numpy as np
import os
from helper import load_predictions, calculate_errors
from diversity_stats import calc_generalized_div
from ensemble_methods import voting
import time
import argparse


def run(model_names, n_query, n_shot, n_way, dataset, weights, size_penalty):
    ds_name = "val"

    # prepare the model paths
    predictions, pred_arr = load_predictions(model_names=model_names, n_query=n_query, n_shot=n_shot,
                                             n_way=n_way, class_name=ds_name, dataset=dataset)
    all_error_dict, all_error_arr = calculate_errors(predictions, pred_arr, n_query=n_query, n_way=n_way)

    def calc_div_acc(solution):
        comb_idx = solution.astype(bool)

        # select ensemble set
        set_bin_arr = all_error_arr[:, comb_idx]
        set_preds = pred_arr[:, comb_idx]

        # calc focal diversity of ensemble
        focal_div = 0
        ens_size = sum(solution)
        for focal_idx in range(ens_size):
            focal_arr = set_bin_arr[:, focal_idx]
            neg_idx = np.where(focal_arr == 0)[0]
            neg_samp_arr = set_bin_arr[neg_idx]
            focal_div += calc_generalized_div(neg_samp_arr)
        focal_div /= ens_size

        # calculate accuracy of ensemble
        ens_pred = voting(set_preds, method="plurality", n_way=n_way, n_query=n_query)
        y = np.tile(np.repeat(range(n_way), n_query), len(ens_pred))
        ens_pred_flatten = ens_pred.flatten()
        acc_score = np.mean(y == ens_pred_flatten)

        return focal_div, acc_score

    def fitness_function(ga_instance, solution, solution_idx):
        if sum(solution) < 2:
            score = -99
        else:
            focal_div, acc_score = calc_div_acc(solution)
            score = focal_div * weights[0] + acc_score * weights[1]
            if size_penalty:
                score -= 0.1 * sum(solution)/len(solution)
        return score

    ga_params = {
        "num_generations": 1000,
        "num_parents_mating": 50,
        "sol_per_pop": 100,
        "num_genes": len(model_names),
        "fitness_func": fitness_function,
        "gene_space": [0, 1],
        "parent_selection_type": "sss",
        "crossover_type": "two_points",
        "gene_type": int,
        "mutation_by_replacement": False,
        "mutation_probability": 0.,
        # mutation_type="adaptive",
        # mutation_probability=[0.25, 0.01],
        # "stop_criteria": ["reach_0.475"],
        "stop_criteria": ["saturate_100"],
    }

    ga_instance = pygad.GA(**ga_params)
    print("Genetic algorithm has started")
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    ga_instance.plot_fitness(ylabel="Score", title="", font_size=16)

    # solution, solution_fitness, solution_idx = ga_instance.best_solution()

    pop_fitness = ga_instance.cal_pop_fitness()
    top_idx = pop_fitness.argsort()[-5:]
    for i in range(5):
        sol = ga_instance.population[top_idx[i]]
        sol_div, sol_acc = calc_div_acc(sol)
        selected_models = [model_names[i] for i in range(len(model_names)) if sol[i]]
        print(f"Selected models in the top {i} solution : {selected_models} with "
              f"Focal Diversity, Accuracy, and Fitness value = {sol_div}, {sol_acc}, {pop_fitness[top_idx[i]]}")

    print(f"Lasted {(end_time - start_time)}seconds")
    print(ga_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='focal diversity pruning')
    parser.add_argument('--dataset_name', default="miniImagenet", choices=["CUB", "miniImagenet"])
    parser.add_argument("--n_query", default=15, type=int)
    parser.add_argument("--n_way", default=5, type=int)
    parser.add_argument("--n_shot", default=1, type=int)
    parser.add_argument("--focal_div_weight", default=0.6, type=float)
    parser.add_argument("--acc_weight", default=0.4, type=float)
    parser.add_argument("--size_penalty", action="store_true")
    parser.add_argument('--model_names', nargs='+',
                        help='Model name and backbone e.g. protonet_ResNet18', required=True)
    args = parser.parse_args()

    wgh = [args.focal_div_weight, args.acc_weight]  # div_weight, acc_weight

    run(model_names=args.model_names, weights=wgh, dataset=args.dataset_name,
        n_shot=args.n_shot, n_way=args.n_way, n_query=args.n_query, size_penalty=args.size_penalty)
