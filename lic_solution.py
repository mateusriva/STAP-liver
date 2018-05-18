"""This module contain the SRG Solution class.

This module includes functions for generating and improving
SRG solutions, along with global cost functions.
"""

import numpy as np
from itertools import permutations

import lic_patient, lic_srg

class Matching:
    """Contains a matching solution between SRGs.

    A matching solution is an injective mapping
    between the vertexes of an observation graph
    and the vertexes of the model graph.

    This is represented as a dictionary, where
    each key is a vertex in the observation graph
    and each value is the matched model vertex.

    Attributes
    ----------
    match_dict : `dict`
        Matching dictionary between vertexes.
    model_graph : `SRG`
        Model graph to match against.
    observation_graph : `SRG`
        Observation graph to be matched.
    """
    def __init__(self, match_dict, model_graph, observation_graph):
        self.match_dict = match_dict
        self.model_graph = model_graph
        self.observation_graph = observation_graph

    def cost(self, weights=None, vertex_weights=None, edge_weights=None):
        """Computes the global cost of this solution.

        Two weights may be provided: respectively,
        the weight of the vertex total distance and
        the weight of the edge total distance.

        Arguments
        ---------
        weights : `tuple` of two `floats`
            Weights for vertex distance sum and
            edge distance sum, respectively. If
            `None`, weights are equal.
        vertex_weights : `list` of `float`
            Weight of each Vertex attribute. If None, weights are equal.
        edge_weights : `list` of `float`
            Weight of each Edge attribute. If None, weights are equal.

        Returns
        -------
        cost : `float`
            Global cost of the solution, weighted.
        """
        if weights is None:
            weights = (1,1)

        # Computing all vertex distances
        vertex_distances = self.vertex_cost(vertex_weights)
        # Computing all edge distances
        edge_distances = self.edge_cost(vertex_weights)
        return (weights[0]*(vertex_distances) + weights[1]*(edge_distances))/np.sum(weights)

    def vertex_cost(self, weights=None):
        """Computes the average vertex cost of this solution.

        Arguments
        ---------
        weights : `tuple` of two `floats`
            Weight of each Vertex attribute. If None, weights are equal.

        Returns
        -------
        cost : `float`
            Average vertex cost of the solution, weighted.
        """
        if weights is None:
            weights = (1,1)
        return np.mean(list(self.observation_graph.vertexes[key].cost_to(self.model_graph.vertexes[value], weights) for key, value in self.match_dict.items()))

    def edge_cost(self, weights=None):
        """Computes the average edge cost of this solution.

        Arguments
        ---------
        weights : `tuple` of two `floats`
            Weight of each Edge attribute. If None, weights are equal.

        Returns
        -------
        cost : `float`
            Average edge cost of the solution, weighted.
        """
        if weights is None:
            weights = (1,1)
        return np.mean(list(
            self.observation_graph.adjacency_matrix[pair1[0],pair2[0]]
            .cost_to(self.model_graph.adjacency_matrix[pair1[1],pair2[1]], weights) 
            for pair1, pair2 in permutations(self.match_dict.items(), 2) 
                if pair1[0] < pair2[0]))



if __name__ == '__main__':
    from time import time

    print("Loading a single patient... ", end="", flush=True)
    t0 = time()
    model_patient = lic_patient.Patient.build_from_folder("data/4")
    print("Done. {:.4f}s".format(time()-t0))

    # We will be cutting the patient's volume and labelmap, just for speeding up the test
    model_patient.volumes["t2"].data = model_patient.volumes["t2"].data[:,:,20:]
    model_patient.labelmaps["t2"].data = model_patient.labelmaps["t2"].data[:,:,20:]

    # Splitting the background into 3 labels
    model_patient.labelmaps["t2"].data += 2 # Adding space for the extra labels at the start
    model_patient.labelmaps["t2"].data[np.logical_and(model_patient.volumes["t2"].data < 10, model_patient.labelmaps["t2"].data == 2)] = 0 # posterior background is 0
    model_patient.labelmaps["t2"].data[model_patient.labelmaps["t2"].data.shape[1]//2:,:,:][model_patient.labelmaps["t2"].data[model_patient.labelmaps["t2"].data.shape[1]//2:,:,:] == 0] = 1 # anterior background is 1

    print("Building model graph... ", end="", flush=True)
    t0 = time()
    model_graph = lic_srg.SRG.build_from_patient(model_patient)
    print("Done. {:.4f}s".format(time()-t0))

    print("Running watershed... ", end="", flush=True)
    t0 = time()
    watershed_labelmap = model_patient.volumes['t2'].watershed_volume()
    print("Done. {} labels found. {:.4f}s".format(watershed_labelmap.header["num_labels"], time()-t0))


    print("Building observation graph... ", end="", flush=True)
    t0 = time()
    from copy import deepcopy
    observed_patient = deepcopy(model_patient)
    observed_patient.labelmaps['t2'] = watershed_labelmap
    observation_graph = lic_srg.SRG.build_from_patient(observed_patient)
    print("Done. {:.4f}s".format(time()-t0))

    # displaying greedy solution
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches, matplotlib.colors as mcolors
    from lic_display import IndexTracker, display_volume, bg3label_color_map as label_color_map, bg3label_text_map as label_text_map
    
    # # Assembling and displaying predicted labelmap
    # predicted_labelmap = deepcopy(watershed_labelmap.data)
    # for observation, prediction in match_dict.items():
    #     predicted_labelmap[predicted_labelmap==observation] = -prediction
    # predicted_labelmap *= -1

    # fig,axes=plt.subplots(1,2)
    # trackers = [IndexTracker(ax, X, title=title, cmap=cmap) 
    #             for ax,X,title,cmap in zip(axes,[predicted_labelmap, model_patient.labelmaps["t2"].data],["Predictions","Truth"],[mcolors.ListedColormap(list(label_color_map.values())),mcolors.ListedColormap(list(label_color_map.values()))])]
    # for tracker in trackers:
    #     fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    # patches = [mpatches.Patch(color=value, label=label_text_map[key]) for key,value in label_color_map.items()]
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    # plt.show()
    # Testing multiple vertex cost weights
    for weights in [(0,1),(0.02,0.98),(0.03,0.97)]:
        # generating greedy solution
        # TODO: move this to a "lic_solution" module?
        print("Generating greedy solution for weights {}... ".format(weights), end="", flush=True)
        t0 = time()
        # creating empty match dict
        match_dict = {}
        # for each vertex in the observation graph, find the closest matched model vertex (ignore edge info)
        for i, obs_vertex in enumerate(observation_graph.vertexes):
            best_model_vertex = np.argmin([obs_vertex.cost_to(model_vertex, weights=weights) for model_vertex in model_graph.vertexes])
            match_dict[i] = best_model_vertex
        print("Done. {:.4f}s".format(time()-t0))

        print("Computing cost... ", end="", flush=True)
        solution = Matching(match_dict, model_graph, observation_graph)
        cost = solution.cost()
        print("Done. Cost is {}. {:.4f}s".format(cost, time()-t0))

        # Assembling and displaying predicted labelmap
        predicted_labelmap = deepcopy(watershed_labelmap.data)
        for observation, prediction in match_dict.items():
            predicted_labelmap[predicted_labelmap==observation] = -prediction
        predicted_labelmap *= -1

        #fig,axes=plt.subplots(1,2)
        #trackers = [IndexTracker(ax, X, title=title, cmap=cmap) 
        #            for ax,X,title,cmap in zip(axes,[predicted_labelmap, model_patient.labelmaps["t2"].data],["Predictions {}".format(weights),"Truth"],[mcolors.ListedColormap(list(label_color_map.values())),mcolors.ListedColormap(list(label_color_map.values()))])]
        #for tracker in trackers:
        #    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        display_volume(predicted_labelmap, title="Predictions - weights {}".format(weights), cmap=mcolors.ListedColormap(list(label_color_map.values())))
        plt.show()
