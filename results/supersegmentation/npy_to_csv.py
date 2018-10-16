"""Converts supersegmentation experiment npy results to a csv."""

import numpy as np
import csv

all_as = ["traditional_watershed", "compact_watershed_886", "compact_watershed_10108", "slic400", "slic600"]
all_fs = ["0","1","2","4"]
all_stats = ["costs", "dices_average", "dices_liver", "regions", "times"]

stats = {}

for a in all_as:
    if a not in stats.keys():
        stats[a] = {}
    for f in all_fs:
        if f not in stats.keys():
            stats[a][f] = {}
        for stat in all_stats:
            stats[a][f][stat] = np.load("results/supersegmentation/{}_{}-{}.npy".format(a,f,stat))

with open("supersegmentation.csv", "w") as f:
    writer = csv.writer(f)

    columns = []
    for a in all_as:
        for f in all_fs:
            for stat in all_stats:
                column = [a, f, stat]
                for value in stats[a][f][stat]:
                    column.append(value)
                columns.append(column)

    # transpose columns to rows
    rows = list(map(list, zip(*columns)))

    for row in rows:
        writer.writerow(row)
