import os
import glob
import itertools
import numpy as np
import pandas as pd
import scipy
import torch.nn
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_ensemble import MLP, test_loop, load_logits, create_data, modify_save_path


def run(model_names):
    dataset = "miniImagenet"
    novel_logits = load_logits(model_names, dataset=dataset, class_type="novel",
                               perform_norm=True, nway=n_way, nshot=n_shot)
    novel_data = create_data(novel_logits, n_way, n_query)
    novel_loader = DataLoader(novel_data, batch_size=64, shuffle=True)

    model = MLP(len(model_names) * 5, [100, 100], 5)
    model = model.to("cuda")

    path = f"ens_checkpoints/{dataset}/one_shot/{'-'.join(model_names)}"
    path = modify_save_path(path)

    best_dict = torch.load(f"{path}/best_model.tar")
    model.load_state_dict(best_dict["state"])
    model.eval()

    acc_mean, acc_std, ens_logits, labels = test_loop(model, novel_loader, ret_logit=True)
    ens_logits = ens_logits.reshape(-1, len(model_names) + 1, 5)

    episode_count = 30
    fig, ax = plt.subplots(figsize=(10, 5))
    x_axis = np.arange(episode_count * 5)
    data = []
    for i in range(len(model_names)):
        m_logit = ens_logits[:episode_count, i]
        data.append(m_logit.flatten())
    df = pd.DataFrame(np.stack(data).T, columns=model_names)
    df.plot.bar(stacked=True, ax=ax, alpha=0.4)
    ax.plot(x_axis, ens_logits[:episode_count, -1].flatten(), c='k', label="ensemble")
    y = np.zeros((episode_count, 5))
    y[:] = np.nan
    y[np.arange(episode_count), labels[:episode_count].astype(int)] = 0.5
    ax.scatter(x_axis, y.flatten(), marker='x', c='r', label="ground truth")
    ax.legend()
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(0, episode_count * 5 + 1, 5))
    ax.set_xticks(x_axis, minor=True)
    ax.set_xticklabels(np.arange(0, episode_count + 1, 1))
    ax.xaxis.grid(which='both')
    ax.xaxis.grid(which='minor', alpha=0.3, linestyle='dashed')
    ax.xaxis.grid(which='major', alpha=1)
    ax.set_xlim(-1, episode_count * 5)
    ax.set_xlabel("Episode numbers")
    ax.set_ylabel("Probability")
    plt.savefig(f"figures/visualized_weights_{episode_count}.png", dpi=200, bbox_inches="tight")

    print(acc_mean)


if __name__ == '__main__':
    import pickle as pkl
    n_shot = 1
    n_query = 15
    n_way = 5
    n_epochs = 300
    # lambda_1 = 0.01
    # temperatures = [1, 1, 1]
    # temp_flag = True
    normalize = True

    # *** Simple Shot Exps ***
    all_names = ["protonet_ResNet18", "simpleshot_ResNet18", "DeepEMD"]
    run(model_names=all_names)

    # ep_num = 2
    # plt.figure()
    # x_axis = np.arange(5)
    # splitted = np.split(model_preds[ep_num], 4)
    # for i, w in enumerate(splitted):
    #     plt.plot(x_axis, w, '-o', c='navy', label="base_model")
    # plt.plot(x_axis, scores.detach().cpu().numpy()[ep_num], '-o', c="r", label="ensemble_model")
    # plt.xticks(x_axis)
    # plt.legend()
    # plt.savefig(f"{ep_num}.png")