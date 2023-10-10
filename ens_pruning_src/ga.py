import pygad
import numpy as np
import os
from helper import load_predictions, calculate_errors, calc_generalized_div, voting
import time


def run():
    n_query = 15
    n_way = 5
    n_shot = 1
    ds_name = "val"
    dataset = "miniImagenet"
    working_dir = os.path.dirname(os.path.abspath(__file__))

    model_back_bones = ["ResNet34"]
    # model_back_bones = ["ResNet34"]
    model_back_bones = ["Conv4", "Conv6", "ResNet10", "ResNet18", "ResNet34"]
    method_names = ["matchingnet", "protonet", "maml_approx", "relationnet"]
    model_names = []
    for m in method_names:
        for bb in model_back_bones:
            model_names.append(f"{m}_{bb}")

    # Add simple shot models to the pool
    # ss_back_bones = ["ResNet18"]
    ss_back_bones = ["Conv4", "Conv6", "ResNet10", "ResNet18", "ResNet34", "DenseNet121", "WideRes"]
    ss_methods = [f"simpleshot_{bb}" for bb in ss_back_bones]
    model_names += ss_methods

    # Add DeepEMD also
    model_names.append("DeepEMD")

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
            score -= 0.1 * sum(solution)/len(solution)
        return score

    # create GA params
    weights = [0.3, 0.7]  # div_weight, acc_weight

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
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    ga_instance.plot_fitness(ylabel="Score", title="", font_size=16)

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    sol_div, sol_acc = calc_div_acc(solution)
    selected_models = [model_names[i] for i in range(len(model_names)) if solution[i]]
    print(f"Selected models in the best solution : {selected_models}")
    print(f"Focal Diversity, Accuracy, and Fitness values of the best solution = {sol_div}, {sol_acc}, {solution_fitness}")
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    print(f"Lasted {(end_time - start_time)}seconds")
    print(ga_params)
    # fun_inputs = [4, -2, 3.5, 5, -11, -4.7]
    # desired_output = 44
    #
    # def fitness_function(ga_instance, solution, solution_idx):
    #     output = np.sum(solution * fun_inputs)
    #     fitness = 1.0 / np.abs(output - desired_output)
    #     return fitness
    #
    # # PyGAD parameters
    # fitness_func = fitness_function
    #
    # num_generations = 50
    # num_parents_mating = 4
    #
    # sol_per_pop = 8
    # num_genes = len(fun_inputs)
    #
    # init_range_low = -2
    # init_range_high = 5
    #
    # parent_selection_type = "sss"
    # keep_parents = 1
    #
    # crossover_type = "single_point"
    #
    # mutation_type = "random"
    # mutation_percent_genes = 10
    #
    # ga_instance = pygad.GA(num_generations=num_generations,
    #                        num_parents_mating=num_parents_mating,
    #                        fitness_func=fitness_func,
    #                        sol_per_pop=sol_per_pop,
    #                        num_genes=num_genes,
    #                        init_range_low=init_range_low,
    #                        init_range_high=init_range_high,
    #                        parent_selection_type=parent_selection_type,
    #                        keep_parents=keep_parents,
    #                        crossover_type=crossover_type,
    #                        mutation_type=mutation_type,
    #                        gene_type=int,
    #                        mutation_percent_genes=mutation_percent_genes)
    #
    # ga_instance.run()
    #
    # solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # print("Parameters of the best solution : {solution}".format(solution=solution))
    # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    #
    # prediction = np.sum(np.array(fun_inputs) * solution)
    # print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))


if __name__ == '__main__':
    run()
