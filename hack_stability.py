"""Script for assessing graph stability under disturbances.
"""


import numpy as np
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt

from lic_srg import SRG
from lic_solution import Matching
from lic_patient import Patient, Volume, LabelMap

# Importing patient
model_patient = Patient.build_from_folder("data/4")
# Building model graph
model_graph = SRG.build_from_patient(model_patient)

# Start disturbance experiments
# Parameters
# ----------
#seeds = [1]
seeds = range(100) # seeds for experiments
disturbances = 1000 # disturbance count

# Results
# -------
all_edge_costs = [] # List of lists of costs
all_vertex_costs = [] # List of lists of costs
all_costs = [] # List of lists of costs
all_accuracies = [] # List of lists of prediction accuracy
for seed in seeds:
    print("On seed {}".format(seed))
    # Seeding the numpy's random generator
    np.random.seed(seed)

    # Copying model graph
    observation_graph = deepcopy(model_graph)

    # Initializing cost and accuracy list
    costs, edge_costs, vertex_costs, accuracies = [], [], [], []

    print("0%      50%      100%")
    # applying disturbances
    for i in range(disturbances):
        if i % (disturbances/20) == 0:
            print("#", end="", flush=True)
        # choosing a random vertex to disturb
        vertex_to_disturb_index = np.random.randint(len(observation_graph.vertexes))
        vertex_to_disturb = observation_graph.vertexes[vertex_to_disturb_index]
        # choosing a random attribute to disturb
        attribute_to_disturb_key = np.random.choice(list(vertex_to_disturb.attributes.keys()))
        # applying between -10% and +10% disturbance
        if attribute_to_disturb_key == "centroid":
            observation_graph.vertexes[vertex_to_disturb_index].attributes[attribute_to_disturb_key] = [x*((np.random.rand() * 0.2) + 0.9) for x in observation_graph.vertexes[vertex_to_disturb_index].attributes[attribute_to_disturb_key]]
        elif attribute_to_disturb_key == "mean_intensity":
            observation_graph.vertexes[vertex_to_disturb_index].attributes[attribute_to_disturb_key] *= ((np.random.rand() * 0.2) + 0.9)
        # recomputing edges
        if attribute_to_disturb_key == "centroid":
            for other_vertex_index, other_vertex in enumerate(observation_graph.vertexes):
                if other_vertex_index == vertex_to_disturb_index:
                    continue
                observation_graph.adjacency_matrix[other_vertex_index,vertex_to_disturb_index].attributes["distance"] = [vertex_to_disturb.attributes["centroid"][i] - other_vertex.attributes["centroid"][i] for i in range(3)]
                observation_graph.adjacency_matrix[vertex_to_disturb_index,other_vertex_index].attributes["distance"] = [other_vertex.attributes["centroid"][i] - vertex_to_disturb.attributes["centroid"][i] for i in range(3)]

        # Done disturbing, now computing solution, cost and accuracy
        # creating empty match dict
        match_dict = {}
        # for each vertex in the observation graph, find the closest matched model vertex (ignore edge info)
        for i, obs_vertex in enumerate(observation_graph.vertexes):
            best_model_vertex = np.argmin([obs_vertex.cost_to(model_vertex) for model_vertex in model_graph.vertexes])
            match_dict[i] = best_model_vertex
        solution = Matching(match_dict, model_graph, observation_graph)
        
        # computing cost and accuracy
        cost = solution.cost()
        edge_cost = solution.edge_cost()
        vertex_cost = solution.vertex_cost()
        accuracy = sum([key == value for key,value in match_dict.items()])/len(match_dict)

        # appending
        costs.append(cost)
        edge_costs.append(edge_cost)
        vertex_costs.append(vertex_cost)
        accuracies.append(accuracy)
    print("")

    all_edge_costs.append(edge_costs)
    all_vertex_costs.append(vertex_costs)
    all_costs.append(costs)
    all_accuracies.append(accuracies)

# Computing and displaying final results
all_edge_costs = np.array(all_edge_costs)
all_vertex_costs = np.array(all_vertex_costs)
all_costs = np.array(all_costs)
mean_edge_costs = np.mean(all_edge_costs, 0)
mean_vertex_costs = np.mean(all_vertex_costs, 0)
mean_costs = np.mean(all_costs, 0)
all_accuracies = np.array(all_accuracies)
mean_accuracies = np.mean(all_accuracies, 0)

plt.scatter(range(disturbances), mean_costs, c=mean_accuracies)
plt.plot(range(disturbances), mean_edge_costs, "r-", label="Edge Costs")
plt.plot(range(disturbances), mean_vertex_costs, "b-", label="Vertex Costs")
plt.legend()
plt.title("Cost and Accuracy (as color) per disturbance")
plt.ylabel("Cost")
plt.xlabel("Disturbances")
plt.colorbar()
plt.grid()
plt.savefig("plots/disturbance-cost.png")
plt.clf()
plt.scatter(range(disturbances), mean_accuracies, c=mean_costs)
plt.title("Accuracy and Cost (as color) per disturbance")
plt.ylabel("Accuracy")
plt.xlabel("Disturbances")
plt.colorbar()
plt.grid()
plt.savefig("plots/disturbance-acc.png")
plt.clf()

# Start mislabel experiments
# Parameters
# ----------
#seeds = [1]
seeds = range(100) # seeds for experiments
mislabels = 150 # mislabel count

# Results
# -------
all_edge_costs = [] # List of lists of costs
all_vertex_costs = [] # List of lists of costs
all_costs = [] # List of lists of costs
all_accuracies = [] # List of lists of prediction accuracy
for seed in seeds:
    print("On seed {}".format(seed))
    # Seeding the numpy's random generator
    np.random.seed(seed)

    # Copying model graph
    observation_graph = deepcopy(model_graph)

    # Initializing cost and accuracy list
    costs, edge_costs, vertex_costs, accuracies = [], [], [], []

    # Computing solution
    # creating empty match dict
    match_dict = {}
    # for each vertex in the observation graph, find the closest matched model vertex (ignore edge info)
    for i, obs_vertex in enumerate(observation_graph.vertexes):
        best_model_vertex = np.argmin([obs_vertex.cost_to(model_vertex) for model_vertex in model_graph.vertexes])
        match_dict[i] = best_model_vertex
    solution = Matching(match_dict, model_graph, observation_graph)

    print("0%      50%      100%")
    # applying mislabels
    for i in range(mislabels):
        if i % (mislabels/20) == 0:
            print("#", end="", flush=True)
        # choosing a random vertex to mislabel
        vertex_to_mislabel_index = np.random.choice(list(match_dict.keys()))
        match_dict[vertex_to_mislabel_index] = np.random.randint(len(match_dict))

        # reassembling the solution
        solution = Matching(match_dict, model_graph, observation_graph)
        
        # computing cost and accuracy
        cost = solution.cost()
        edge_cost = solution.edge_cost()
        vertex_cost = solution.vertex_cost()
        accuracy = sum([key == value for key,value in match_dict.items()])/len(match_dict)

        # appending
        costs.append(cost)
        edge_costs.append(edge_cost)
        vertex_costs.append(vertex_cost)
        accuracies.append(accuracy)
    print("")

    all_edge_costs.append(edge_costs)
    all_vertex_costs.append(vertex_costs)
    all_costs.append(costs)
    all_accuracies.append(accuracies)

# Computing and displaying final results
all_edge_costs = np.array(all_edge_costs)
all_vertex_costs = np.array(all_vertex_costs)
all_costs = np.array(all_costs)
mean_edge_costs = np.mean(all_edge_costs, 0)
mean_vertex_costs = np.mean(all_vertex_costs, 0)
mean_costs = np.mean(all_costs, 0)
all_accuracies = np.array(all_accuracies)
mean_accuracies = np.mean(all_accuracies, 0)

plt.scatter(range(mislabels), mean_costs, c=mean_accuracies)
plt.plot(range(mislabels), mean_edge_costs, "r-", label="Edge Costs")
plt.plot(range(mislabels), mean_vertex_costs, "b-", label="Vertex Costs")
plt.legend()
plt.title("Cost and Accuracy (as color) per mislabel")
plt.ylabel("Cost")
plt.xlabel("Mislabels")
plt.colorbar()
plt.grid()
plt.savefig("plots/mislabel-cost.png")
plt.clf()
plt.scatter(range(mislabels), mean_accuracies, c=mean_costs)
plt.title("Accuracy and Cost (as color) per mislabel")
plt.ylabel("Accuracy")
plt.xlabel("Mislabels")
plt.colorbar()
plt.grid()
plt.savefig("plots/mislabel-acc.png")
plt.clf()
