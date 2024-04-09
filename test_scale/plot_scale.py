import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import glob


time_taken = []
acc_all = []
for dir_path in glob.glob("scale_stats_*.npy"):
    arr = np.load(dir_path)
    acc_all.append(arr[0])
    time_taken.append(arr[2])
time_taken_mean, time_taken_std = np.mean(time_taken, axis=0), 2 * np.std(time_taken, axis=0)
acc_all_mean, acc_std = np.mean(acc_all, axis=0), 2 * np.std(acc_all, axis=0)

print()

# plt.style.use("seaborn")
fig, ax = plt.subplots(figsize=(7, 4))
x_axis = np.arange(len(time_taken_mean))
ax.plot(x_axis, time_taken_mean, '-o', lw=3, c="tab:blue", label="Pruning+Fitting")
ax.fill_between(x_axis, time_taken_mean + time_taken_std, time_taken_mean - time_taken_std, alpha=0.3, color='tab:blue')
ax.set_ylim(0, 110)
ax.set_xlim(0, len(time_taken_mean)-1)
ax.set_ylabel("Time to Converge (s)", fontsize=18)
ax.set_xlabel("Number of Models in the Pool", fontsize=18)
ax.tick_params(axis='both', labelsize=18)
ax.set_title("Time Complexity", fontsize=18)
ax.legend(fontsize=16, loc="lower right", ncol=4,
          fancybox=True, shadow=True)
ax.yaxis.grid(alpha=0.9)  # horizontal lines
ax.xaxis.grid(alpha=0.9)  # horizontal lines
plt.savefig("model_scaling.png", dpi=200, bbox_inches="tight")
# plt.show()
