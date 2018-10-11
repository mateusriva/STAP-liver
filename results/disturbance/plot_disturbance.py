"""Script for plotting disturbance results as produced by `explorations.disturbance`"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Plot function
def plot_mean_and_CI(mean, lb, ub, ax, color_mean=None, color_shading=None, label=None):
    # plot the shaded range of the confidence intervals
    ax.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    ax.plot(mean, color_mean, label=label)

# Experiment to plot
n = (8,8,6)

# Loading experiment files
costs_rep = np.load("results/disturbance/{}-{}-{}-costs.npy".format(n[0],n[1],n[2]))
dices_liver_rep = np.load("results/disturbance/{}-{}-{}-dices_liver.npy".format(n[0],n[1],n[2]))
dices_average_rep = np.load("results/disturbance/{}-{}-{}-dices_average.npy".format(n[0],n[1],n[2]))

# obtaining means and sems
reps = costs_rep.shape[1]
costs_mean = np.mean(costs_rep,axis=0)
costs_sem = sem(costs_rep,axis=0)
costs_ci = 1.96 * costs_sem
dices_liver_mean = np.mean(dices_liver_rep,axis=0)
dices_liver_sem = sem(dices_liver_rep,axis=0)
dices_liver_ci = 1.96 * dices_liver_sem
dices_average_mean = np.mean(dices_average_rep,axis=0)
dices_average_sem = sem(dices_average_rep,axis=0)
dices_average_ci = 1.96 * dices_average_sem

#Preparing plot
fig, ax1 = plt.subplots()
# Plotting costs, axis on left
plot_mean_and_CI(costs_mean, costs_mean-costs_ci, costs_mean+costs_sem, ax1, label="Cost", color_mean="blue", color_shading="blue")
ax1.set_ylabel("Matching Cost")
ax1.grid(True, linestyle="--")

# Plotting liver dice, axis on right
ax2 = ax1.twinx()
plot_mean_and_CI(dices_liver_mean, dices_liver_mean-dices_liver_ci, dices_liver_mean+dices_liver_sem, ax2, label="Liver SI", color_mean="red", color_shading="red")
plot_mean_and_CI(dices_average_mean, dices_average_mean-dices_average_ci, dices_average_mean+dices_average_sem, ax2, label="Avg. SI", color_mean="green", color_shading="green")
ax2.set_ylabel("Similarity Index")

ax1.set_xlabel("Repetitions")
plt.title("Correlation Experiment ${} \\times {} \\times {}$".format(n[0],n[1],n[2]))

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, loc=7)


plt.show()
