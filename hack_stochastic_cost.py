"""Script for assessing fidelity of stochastic solution cost."""


import numpy as np
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt
from itertools import permutations

from lic_srg import SRG
from lic_solution import Matching
from lic_patient import Patient, Volume, LabelMap

# Importing patient
model_patient = Patient.build_from_folder("data/4")
# Building model graph
model_graph = SRG.build_from_patient(model_patient)

# We will be cutting the patient's volume and labelmap, just for speeding up the test
model_patient.volumes["t2"].data = model_patient.volumes["t2"].data[:,:,20:30]
model_patient.labelmaps["t2"].data = model_patient.labelmaps["t2"].data[:,:,20:30]

# Splitting the background into 3 labels
model_patient.labelmaps["t2"].data += 2 # Adding space for the extra labels at the start
model_patient.labelmaps["t2"].data[np.logical_and(model_patient.volumes["t2"].data < 10, model_patient.labelmaps["t2"].data == 2)] = 0 # posterior background is 0
model_patient.labelmaps["t2"].data[model_patient.labelmaps["t2"].data.shape[1]//2:,:,:][model_patient.labelmaps["t2"].data[model_patient.labelmaps["t2"].data.shape[1]//2:,:,:] == 0] = 1 # anterior background is 1


print("Running watershed... ", end="", flush=True)
t0 = time()
watershed_labelmap = model_patient.volumes['t2'].watershed_volume()
print("Done. {} labels found. {:.4f}s".format(watershed_labelmap.header["num_labels"], time()-t0))

print("Building observation graph... ", end="", flush=True)
t0 = time()
observed_patient = deepcopy(model_patient)
observed_patient.labelmaps['t2'] = watershed_labelmap
observation_graph = SRG.build_from_patient(observed_patient)
print("Done. {:.4f}s".format(time()-t0))

print("Generating greedy solution... ", end="", flush=True)
t0 = time()
# creating empty match dict
match_dict = {}
# for each vertex in the observation graph, find the closest matched model vertex (ignore edge info)
for i, obs_vertex in enumerate(observation_graph.vertexes):
    best_model_vertex = np.argmin([obs_vertex.cost_to(model_vertex) for model_vertex in model_graph.vertexes])
    match_dict[i] = best_model_vertex
solution = Matching(match_dict, model_graph, observation_graph)
print("Done. {:.4f}s".format(time()-t0))

# Start base truth computing
print("\n==BASE TRUTH==")

base_vertex_cost= np.mean(list(observation_graph.vertexes[key].cost_to(model_graph.vertexes[value]) for key, value in match_dict.items()))
vertex_stdev = np.std(list(observation_graph.vertexes[key].cost_to(model_graph.vertexes[value]) for key, value in match_dict.items()))

base_edge_cost = np.mean(list(
    observation_graph.adjacency_matrix[pair1[0],pair2[0]]
    .cost_to(model_graph.adjacency_matrix[pair1[1],pair2[1]]) 
    for pair1, pair2 in permutations(match_dict.items(), 2) 
        if pair1[0] < pair2[0]))
edge_stdev = np.std(list(
    observation_graph.adjacency_matrix[pair1[0],pair2[0]]
    .cost_to(model_graph.adjacency_matrix[pair1[1],pair2[1]]) 
    for pair1, pair2 in permutations(match_dict.items(), 2) 
        if pair1[0] < pair2[0]))
base_cost = (base_vertex_cost + base_edge_cost)/2

print("Base cost: {:.2f}. Base vertex cost and std: {:.2f} +/- {:.2f}. Base edge cost and std: {:.2f} +/- {:.2f}".format(base_cost, base_vertex_cost, vertex_stdev, base_edge_cost, edge_stdev))

t0 = time()
solution.cost()
base_time = time()-t0
print("Time to compute cost: {:.4f}s".format(base_time))


# Start stochastic cost experiments
# Parameters
# ----------
#seeds = [1]
seeds = range(100) # seeds for experiments
percentages = np.linspace(1,0.1,10) # disturbance count

# Results
# -------
all_edge_costs = [] # List of lists of edge costs
all_vertex_costs = [] # List of lists of vertex costs
all_costs = [] # List of lists of costs
all_times = [] # List of lists of times
for seed in seeds:
    print("On seed {}".format(seed))
    # Seeding the numpy's random generator
    np.random.seed(seed)

    # Initializing cost and accuracy list
    costs, edge_costs, vertex_costs, times = [], [], [], []

    print("0%      50%      100%")
    # computing stochastic costs
    for i, percentage in enumerate(percentages):
        if i % (len(percentages)/10) == 0:
            print("##", end="", flush=True)
        
        # computing costs
        t0 = time()
        cost = solution.cost(vertex_percentage=percentage, edge_percentage=percentage)
        current_time = time()-t0
        edge_cost = solution.edge_cost(edge_percentage=percentage)
        vertex_cost = solution.vertex_cost(vertex_percentage=percentage)

        # appending
        costs.append(cost)
        edge_costs.append(edge_cost)
        vertex_costs.append(vertex_cost)
        times.append(current_time)
    print("")

    all_edge_costs.append(edge_costs)
    all_vertex_costs.append(vertex_costs)
    all_costs.append(costs)
    all_times.append(times)

# Computing and displaying final results
all_edge_costs = np.array(all_edge_costs)
all_vertex_costs = np.array(all_vertex_costs)
all_costs = np.array(all_costs)
all_times = np.array(all_times)
mean_edge_costs = np.mean(all_edge_costs, 0)
mean_vertex_costs = np.mean(all_vertex_costs, 0)
mean_costs = np.mean(all_costs, 0)
mean_times = np.mean(all_times, 0)
std_edge_costs = np.std(all_edge_costs, 0)
std_vertex_costs = np.std(all_vertex_costs, 0)
std_costs = np.std(all_costs, 0)
std_times = np.std(all_times, 0)

fig, ax1 = plt.subplots()
# Plotting a line representing real cost
ax1.plot(percentages, [base_cost]*len(percentages),"r-")
ax1.plot(percentages, [base_vertex_cost]*len(percentages),"b-")
ax1.plot(percentages, [base_edge_cost]*len(percentages),"g-")
# Plotting computed costs per percentage
ax1.errorbar(percentages, mean_costs, std_costs, None, "r--", capsize=3, label="Global Costs")
ax1.errorbar(percentages, mean_vertex_costs, std_vertex_costs, None, "b--", capsize=3, label="Vertex Costs")
ax1.errorbar(percentages, mean_edge_costs, std_edge_costs, None, "g--", capsize=3, label="Edge Costs")
ax1.set_ylabel("Cost")

ax2 = ax1.twinx()
ax2.plot(percentages, [base_time]*len(percentages),"c-")
ax2.errorbar(percentages, mean_times, std_times, None, "c--", capsize=3, label="Time")
ax2.set_ylabel("Seconds")

ax1.set_xlabel("Percentage")


plt.title("Cost per percentage")
ax1.legend()
ax1.grid()
ax2.legend()
plt.savefig("plots/stochastic-cost.png")
