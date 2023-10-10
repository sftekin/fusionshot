import numpy as np
from helper import load_predictions, calculate_errors, calc_perf, load_all_degree_preds


def run():
    n_query = 15
    n_way = 5
    n_shot = 5
    dataset = "cross"
    cls_names = ["novel"]

    # model_back_bones = ["Conv4", "Conv6", "ResNet10", "ResNet18", "ResNet34"]
    # method_names = ["matchingnet", "protonet", "maml_approx", "relationnet"]

    # model_back_bones = ["Conv4", "Conv6", "ResNet10", "ResNet18", "ResNet34", "DenseNet121", "WideRes"]
    model_back_bones = ["ResNet18"]
    method_names = ["matchingnet", "protonet", "maml_approx", "relationnet_softmax", "simpleshot"]

    # model_back_bones = [""]
    # method_names = ["DeepEMD"]

    for cls in cls_names:
        print(f"Dataset : {cls}")
        for m in method_names:
            model_names = [f"{m}_{bb}" for bb in model_back_bones] if m != "DeepEMD" else [m]
            predictions, pred_arr = load_predictions(model_names=model_names, n_query=n_query, n_way=n_way,
                                                     n_shot=n_shot, class_name=cls, dataset=dataset)
            all_error_dict, all_error_arr = calculate_errors(predictions, pred_arr, n_query=n_query, n_way=n_way)

            acc_per_episode = all_error_arr.reshape(600, 75, -1).mean(axis=1) * 100
            acc = acc_per_episode.mean(axis=0)
            std = acc_per_episode.std(axis=0)
            conf_inv = 1.96 * std / np.sqrt(600)

            print_str = [f"{acc[j]:.2f} +- {conf_inv[j]:.2f}" for j in range(len(acc))]
            print("\t".join(print_str))


if __name__ == '__main__':
    run()

