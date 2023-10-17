import os
import numpy as np
import pickle as pkl


MODEL_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "base_model_src", "inference_out")


def load_predictions(model_names, n_query, n_way, n_shot, class_name, dataset):
    predictions = {}
    pred_arr = []
    for model_n in model_names:
        if model_n.split("_")[0] == "maml":
            method_name = "maml_approx" if "approx" == model_n.split("_")[1] else "maml"
        elif model_n.split("_")[0] == "relationnet":
            method_name = "relationnet_softmax" if "softmax" == model_n.split("_")[1] else "relationnet"
        else:
            method_name = model_n.split("_")[0]

        with open(f"{MODEL_OUT_DIR}/{dataset}/model_outs/{method_name}/{model_n}_{class_name}_{n_way}way_{n_shot}shot.pkl", "rb") as f:
            results = pkl.load(f)
        pred = results["predicts"]

        if model_n == "DeepEMD":
            order = lambda x: x.reshape(n_query, n_way).T.flatten()
            pred = np.apply_along_axis(order, axis=1, arr=pred)
        predictions[model_n] = pred
        pred_arr.append(pred.flatten())
    pred_arr = np.stack(pred_arr, axis=0).T

    return predictions, pred_arr


def load_all_degree_preds(model_names, n_query, n_way, set_name):
    predictions = {}
    for model_n in model_names:
        logits = np.load(f"model_outs/{model_n}_{set_name}_logits.npy")
        sorted_b = np.argsort(logits, axis=2)
        pred_arr = []
        for i in range(n_way):
            pred_ = sorted_b[..., i]
            if model_n == "DeepEMD":
                order = lambda x: x.reshape(n_query, n_way).T.flatten()
                pred_ = np.apply_along_axis(order, axis=1, arr=pred_)
            pred_arr.append(pred_.flatten())
        predictions[model_n] = np.stack(pred_arr, axis=0).T
    return predictions


def calculate_errors(predictions, pred_arr, n_query, n_way):
    errors = {}
    for i, model_name in enumerate(predictions.keys()):
        bin_arr = calc_perf(pred_arr[:, i], n_query, n_way)
        errors[model_name] = bin_arr
    err_arr = np.stack(list(errors.values())).T
    return errors, err_arr


def calc_perf(pred, n_query, n_way):
    num_samples = len(pred) // (n_query * n_way)
    y = np.repeat(range(n_way), n_query)
    y = np.tile(y, num_samples)
    return (pred == y).astype(int)


def min_max(in_arr):
    min_ = in_arr.min()
    max_ = in_arr.max()
    return (in_arr - min_) / (max_ - min_)
