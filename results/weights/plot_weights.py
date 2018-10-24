"""Script for plotting weights results as produced by `explorations.weights_calibration`"""

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

# Experiments to plot
ws = range(11)
fs = [0,1,2,4]
f=1

for f in fs:
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.set_ylabel("Matching Cost")
    ax2 = ax1.twinx()
    ax2.grid(True, linestyle="--", axis="y")
    ax2.set_ylabel("Similarity Index")
    ax1.set_xlabel("Weight Set")
    ax1.set_xticks(ws)

    costs_rep, dices_liver_rep, dices_average_rep = [], [], []
    for w in ws:
        costs = np.load("results/weights/{}_{}-costs.npy".format(w,f))
        dices_liver = np.load("results/weights/{}_{}-dices_liver.npy".format(w,f))
        dices_average = np.load("results/weights/{}_{}-dices_average.npy".format(w,f))

        # Removing "NaNs"
        costs = costs[~np.isnan(costs)]
        dices_liver = dices_liver[~np.isnan(dices_liver)]
        dices_average = dices_average[~np.isnan(dices_average)]

        # Removing corrupted outliers (over 2 stdevs)
        if f > 0:
            costs = costs[abs(costs - np.mean(costs)) < 2 * np.std(costs)]
        if f > 0:
            dices_liver = dices_liver[abs(dices_liver - np.mean(dices_liver)) < 2 * np.std(dices_liver)]
        if f > 0:
            dices_average = dices_average[abs(dices_average - np.mean(dices_average)) < 2 * np.std(dices_average)]

        # obtaining means and sems
        costs_mean = np.mean(costs,axis=0)
        costs_sem = sem(costs,axis=0)
        costs_ci = 1.96 * costs_sem
        dices_liver_mean = np.mean(dices_liver,axis=0)
        dices_liver_sem = sem(dices_liver,axis=0)
        dices_liver_ci = 1.96 * dices_liver_sem
        dices_average_mean = np.mean(dices_average,axis=0)
        dices_average_sem = sem(dices_average,axis=0)
        dices_average_ci = 1.96 * dices_average_sem

        costs_rep.append((costs_mean,costs_ci))
        dices_liver_rep.append((dices_liver_mean,dices_liver_ci))
        dices_average_rep.append((dices_average_mean,dices_average_ci))

    costs_means = [x[0] for x in costs_rep]
    costs_cis = [x[1] for x in costs_rep]
    dices_liver_means = [x[0] for x in dices_liver_rep]
    dices_liver_cis = [x[1] for x in dices_liver_rep]
    dices_average_means = [x[0] for x in dices_average_rep]
    dices_average_cis = [x[1] for x in dices_average_rep]

    ax1.bar(np.array(range(11))-0.2, costs_means, width=0.25, align="center", label="Cost", color="b")
    ax1.errorbar(np.array(range(11))-0.2, costs_means, yerr=costs_cis, fmt="none", capsize=2, ecolor="k")
    ax2.bar(np.array(range(11))+0, dices_liver_means, width=0.25, align="center", label="Liver SI", color="r")
    ax2.errorbar(np.array(range(11))+0, dices_liver_means, yerr=dices_liver_cis, fmt="none", capsize=2, ecolor="k")
    ax2.bar(np.array(range(11))+0.2, dices_average_means, width=0.25, align="center", label="Avg SI", color="g")
    ax2.errorbar(np.array(range(11))+0.2, dices_average_means, yerr=dices_average_cis, fmt="none", capsize=2, ecolor="k")

    plt.title("Noise Profile $f_{}$".format(f))

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc=0)
    
    ax1.set_ylim(bottom=0.0, top=None)
    ax2.set_ylim((0.0,1.0))

    plt.show()
