import matplotlib.pyplot as plt
import numpy as np


def main():
    ss_models = ["SimpleS_Conv4", "SimpleS_Conv6", "SimpleS_RN10", "SimpleS_RN18",
                 "SimpleS_RN34", "SimpleS_DN121", "SimpleS_WR", "Ens-All", "Ens-Best"]
    ss_scores = [50.58, 53.20, 58.66, 62.93, 65.80, 66.03, 65.16, 68.71, 69.62]

    fig, ax = plt.subplots(figsize=(8, 6))
    x_axis = np.arange(len(ss_models))
    y_axis = np.linspace(40, 70, 21)
    ax.bar(x_axis[:-2], ss_scores[:-2], color="tab:blue")
    ax.bar(x_axis[-2], ss_scores[-2], color="r")
    ax.bar(x_axis[-1], ss_scores[-1], color="tab:orange")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(ss_models, rotation=40)
    ax.set_yticks(y_axis)
    ax.set_ylim(40, 70)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.savefig("figures/ss_hist.png", bbox_inches="tight", dpi=200)

    sota_models = ["MatchN_RN18", "ProtoN_RN18", "MAML_RN18",
                   "RelationN_RN18", "SimpleS_RN18", "DeepEMD", "Ens-All", "Ens-Best"]
    sota_scores = [49.46, 49.76, 47.57, 47.39, 62.93, 63.92, 63.457, 66.315]
    fig, ax = plt.subplots(figsize=(8, 6))
    x_axis = np.arange(len(sota_models))
    y_axis = np.linspace(40, 70, 21)
    ax.bar(x_axis[:-2], sota_scores[:-2], color="tab:blue")
    ax.bar(x_axis[-2], sota_scores[-2], color="r")
    ax.bar(x_axis[-1], sota_scores[-1], color="tab:orange")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(sota_models, rotation=40)
    ax.set_yticks(y_axis)
    ax.set_ylim(40, 70)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.savefig("figures/sota_hist.png", bbox_inches="tight", dpi=200)

if __name__ == '__main__':
    main()
