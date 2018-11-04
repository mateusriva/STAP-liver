"""Script for tabling weights results as produced by `explorations.weights_calibration`"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Experiments to plot
ws = range(11)
fs = [0,1,2,4]
f=1

print("Noise & Weight Set $W$ & Cost & Average SI & Liver SI \\\\")
for f in fs:
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

    max_liver = np.argmax(dices_liver_means)
    max_average = np.argmax(dices_average_means)
    # Weight sets
    for w in ws:
        line = ""
        if w == 0:
            line = "\multirow{10}{*}{" + str(f) + "}"
        line += " & "
        line += str(w) + " & "
        line += "${:.4f}".format(costs_means[w]) + " \pm " + "{:.4f}".format(costs_cis[w]) + "$ & "
        if w == max_average:
            line += "$\mathbf{{{:.4f}".format(dices_average_means[w]) + " \pm " + "{:.4f}".format(dices_average_cis[w]) + "}$ & "
        else:
            line += "${:.4f}".format(dices_average_means[w]) + " \pm " + "{:.4f}".format(dices_average_cis[w]) + "$ &  "
        if w == max_liver:
            line += "$\mathbf{{{:.4f}".format(dices_liver_means[w]) + " \pm " + "{:.4f}".format(dices_liver_cis[w]) + "}$ \\\\ "
        else:
            line += "${:.4f}".format(dices_liver_means[w]) + " \pm " + "{:.4f}".format(dices_liver_cis[w]) + "$ \\\\ "
        print(line)
