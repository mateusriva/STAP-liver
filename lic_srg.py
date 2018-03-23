"""
This module contains the Statistical-Relational Graph class.

AS the implementation of the SRG is expected to vary wildly
during development, this docstring will remain empty until
the implementation is consolidated.
"""

import numpy as np
from itertools import permutations

import lic_patient, lic_attributes

class SRG:
    """A Statistical-Relational Graph.

    This class represents a SRG, which may be either a model
    SRG (that is, the 'trained' SRG or the 'template') or an
    observed SRG (that is, acquired from observation of an
    unlabeled image).

    Attributes
    ----------
    vertexes : list of :obj:`Vertex`
        List of vertexes. Each is labeled by their index
        (that is, the vertex for label '0' is at [0]).
    adjacency_matrix : ndarray of :obj:`Edge`
        Adjacency matrix, connecting vertexes with edge 
        information.
    statistical_attributes : `list` of str
        List of statistical attributes' keys.
    relational_attributes : `list` of str
        List of relational attributes' keys.
    """
    def __init__(self, vertexes, adjacency_matrix):
        self.vertexes = vertexes
        self.adjacency_matrix = adjacency_matrix

        self.statistical_attributes = list(vertexes[0].attributes.keys())
        self.relational_attributes = list(adjacency_matrix[0][0].attributes.keys())

    def __repr__(self):
        return "SRG with {} vertexes and {} edges\nStat attrs: {}; Rel attrs: {}".format(len(self.vertexes), self.adjacency_matrix.shape, self.statistical_attributes, self.relational_attributes)

    def dump(self):
        """Dump this SRG to string."""
        dump_string = "'SRG':{{'statistical_attributes':{},'relational_attributes':{},".format(self.statistical_attributes, self.relational_attributes)
        dump_string += "'vertexes':{{"
        for i, vertex in enumerate(self.vertexes):
            dump_string += "{{{} : {}}}".format(i, vertex.dump())
            if i < len(self.vertexes)-1:
                dump_string += ","
        dump_string += "}}, 'edges':{{"
        for i, line in enumerate(self.adjacency_matrix):
            for j, edge in enumerate(line):
                dump_string += "{{{},{} : {}}}".format(i,j, edge.dump())
                if i < self.adjacency_matrix.shape[0]-1 or j < self.adjacency_matrix.shape[1]-1:
                    dump_string += ","
        dump_string += "}} }}"
        return dump_string

    @classmethod
    def build_from_patient(cls, patient):
        """Builds a SRG from a set of annotated patients.

        (TODO: multiple patients, currently only one)

        Attributes currently being considered:
        > Statistical: voxel centroid, real intensity
        > Relational: vectorial distance

        Arguments
        ---------
        patient : :obj:`Patient`
            Annotated Patient object
        """
        # Computing statistical attributes
        #volumetry = lic_attributes.compute_volumetry(patient)
        centroids = lic_attributes.compute_centroids(patient)
        mean_intensities = lic_attributes.compute_mean_intensities(patient)

        # Assembling the list of vertexes
        vertexes = []
        for label in centroids.keys():
            new_vertex = Vertex(label, {"centroid":centroids[label]["voxel"], "mean_intensity": mean_intensities[label]["real"]})
            vertexes.append(new_vertex)

        # Assembling the adjacency matrix
        adjacency_matrix = np.empty((len(vertexes), len(vertexes)), dtype=Edge)
        for label1, vertex1 in enumerate(vertexes):
            for label2, vertex2 in enumerate(vertexes):
                # Assembling edge attributes
                # Computing vectorial distance between label1 and label2
                distance = [vertex2.attributes["centroid"][i] - vertex1.attributes["centroid"][i] for i in range(3)]
                # Building Edge object 
                edge = Edge((label1,label2), {"distance": distance})
                
                adjacency_matrix[label1,label2] = edge

        return cls(vertexes, adjacency_matrix)

class Vertex:
    """A vertex of the :class:`SRG`.

    Each vertex contains a list of statistical attributes,
    which may confuse with the OOP usage of the word.
    But oh well.

    Attributes
    ----------
    id : int
        Label of this vertex.
    attributes : dict
        Dictionary of vertex attributes. Each attribute is
        keyed by an ID.
    """
    def __init__(self, id, attributes):
        self.id = id
        self.attributes = attributes

    def __repr__(self):
        return "Vertex {} with attributes {}".format(self.id, self.attributes)

    def dump(self):
        """Dump this Vertex to string."""
        return "'Vertex':{{'id':{},'attributes':{}}}".format(self.id, self.attributes)

    def cost_to(self, other, weights=None):
        """Computes the cost between this `Vertex` and another.

        This functions computes the matching cost (as the
        Euclidean distance between attributes) of two
        vertexes.

        Arguments
        ---------
        other : :obj:`Vertex`
            `Vertex` to compute distance to. Must have the same
            attributes as `self`.
        weights : `list` of `float`
            Weight of each attribute. If None, weights are equal.

        Returns
        -------
        cost : `float`
            Matching cost between vertexes. 
        """
        # Asserting same attributes
        assert self.attributes.keys() == other.attributes.keys(), "Vertexes must have same attributes for matching cost"

        # Equal weights if default
        if weights is None:
            weights = np.ones((len(self.attributes),))

        # Computing euclidean distances per attributes
        distances = np.empty((len(self.attributes)),)
        for i, attribute_key in enumerate(self.attributes.keys()):
            distances[i] = np.linalg.norm(np.array(self.attributes[attribute_key]) - np.array(other.attributes[attribute_key]))

        # weighting distances
        cost = np.sum(weights * distances)/np.sum(weights)
        return cost

class Edge:
    """A Edge of the :class:`SRG`.

    Each Edge contains a list of relational attributes,
    which may confuse with the OOP usage of the word.
    But oh well.

    Attributes
    ----------
    id : tuple
        Labels connected by this edge.
    attributes : dict
        Dictionary of Edge attributes. Each attribute is
        keyed by an ID.
    """
    def __init__(self, id, attributes):
        self.id = id
        self.attributes = attributes

    def __repr__(self):
        return "Edge {} with attributes {}".format(self.id, self.attributes)

    def dump(self):
        """Dump this Edge to string."""
        return "'Edge':{{'id':{},'attributes':{}}}".format(self.id, self.attributes)

    def cost_to(self, other, weights=None):
        """Computes the cost between this `Edge` and another.

        This functions computes the matching cost (as the
        Euclidean distance between attributes) of two
        Edges.

        Arguments
        ---------
        other : :obj:`Edge`
            `Edge` to compute distance to. Must have the same
            attributes as `self`.
        weights : `list` of `float`
            Weight of each attribute. If None, weights are equal.

        Returns
        -------
        cost : `float`
            Matching cost between Edges. 
        """
        # Asserting same attributes
        assert self.attributes.keys() == other.attributes.keys(), "Edges must have same attributes for matching cost"

        # Equal weights if default
        if weights is None:
            weights = np.ones((len(self.attributes),))

        # Computing euclidean distances per attributes
        distances = np.empty((len(self.attributes)),)
        for i, attribute_key in enumerate(self.attributes.keys()):
            distances[i] = np.linalg.norm(np.array(self.attributes[attribute_key]) - np.array(other.attributes[attribute_key]))

        # weighting distances
        cost = np.sum(weights * distances)/np.sum(weights)
        return cost

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

    def cost(self, weights=None):
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

        Returns
        -------
        cost : `float`
            Global cost of the solution, weighted.
        """
        if weights is None:
            weights = (1,1)

        # Computing all vertex distances
        vertex_distances = sum(self.observation_graph.vertexes[key].cost_to(self.model_graph.vertexes[value]) for key, value in self.match_dict.items())
        # Computing all edge distances
        edge_distances = sum(
            self.observation_graph.adjacency_matrix[pair1[0],pair2[0]]
            .cost_to(self.model_graph.adjacency_matrix[pair1[1],pair2[1]]) 
            for pair1, pair2 in permutations(self.match_dict.items(), 2) 
                if pair1[0] < pair2[0])
        return (weights[0]*(vertex_distances) + weights[1]*(edge_distances))/np.sum(weights)

if __name__ == '__main__':
    from time import time

    print("Loading a single patient... ", end="", flush=True)
    t0 = time()
    model_patient = lic_patient.Patient.build_from_folder("data/4")
    print("Done. {:.4f}s".format(time()-t0))


    print("Building model graph... ", end="", flush=True)
    t0 = time()
    model_graph = SRG.build_from_patient(model_patient)
    print("Done. {:.4f}s".format(time()-t0))


    print(model_graph)
    #for i, vertex in enumerate(model_graph.vertexes):
    #    print("Reporting on vertex {}\n---------------------".format(i))
    #    print(vertex)
    #    for edge in model_graph.adjacency_matrix[i,:]:
    #        print(edge)
    #    print("")


    print("Running watershed... ", end="", flush=True)
    t0 = time()
    watershed_labelmap = model_patient.volumes['t2'].watershed_volume()
    print("Done. {} labels found. {:.4f}s".format(watershed_labelmap.header["num_labels"], time()-t0))


    print("Building observation graph... ", end="", flush=True)
    t0 = time()
    observed_patient = model_patient
    observed_patient.labelmaps['t2'] = watershed_labelmap
    observation_graph = SRG.build_from_patient(observed_patient)
    print("Done. {:.4f}s".format(time()-t0))

    print(observation_graph)
    #for i, vertex in enumerate(observation_graph.vertexes[:2]):
    #    print("Reporting on vertex {}\n---------------------".format(i))
    #    print(vertex)
    #    for edge in observation_graph.adjacency_matrix[i,:2]:
    #        print(edge)
    #    print("")

    #print(model_graph.dump(), file=open("model_graph.srg", "w"))
    #print(observation_graph.dump(), file=open("observation_graph.srg", "w"))

    # Generate random matching solution
    import random

    print("Generating random solution... ", end="", flush=True)
    t0 = time()
    possible_options = len(model_graph.vertexes)
    match_dict = {}
    for i, vertex in enumerate(observation_graph.vertexes):
        match_dict[i] = random.randint(0, possible_options-1)
    print("Done. {:.4f}s".format(time()-t0))

    print("Computing cost... ", end="", flush=True)
    solution = Matching(match_dict, model_graph, observation_graph)
    cost = solution.cost()
    print("Done. Cost is {}. {:.4f}s".format(cost, time()-t0))

    # displaying random solution
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from lic_display import display_solution, label_color_map, label_text_map
    plt.title("Random solution. Cost is {:.2f}".format(cost))
    plt.imshow(display_solution(observed_patient.volumes['t2'].data[:,:,36], observed_patient.labelmaps['t2'].data[:,:,36], solution.match_dict, window_wl=(700,300)))
    patches = [mpatches.Patch(color=value, label=label_text_map[key]) for key,value in label_color_map.items()]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.show()

    # generating greedy solution
    # TODO: move this to a "lic_solution" module?
    print("Generating greedy solution... ", end="", flush=True)
    t0 = time()
    # creating empty match dict
    match_dict = {}
    # for each vertex in the observation graph, find the closest matched model vertex (ignore edge info)
    for i, obs_vertex in enumerate(observation_graph.vertexes):
        best_model_vertex = np.argmin([obs_vertex.cost_to(model_vertex) for model_vertex in model_graph.vertexes])
        match_dict[i] = best_model_vertex
    print("Done. {:.4f}s".format(time()-t0))

    print("Computing cost... ", end="", flush=True)
    solution = Matching(match_dict, model_graph, observation_graph)
    cost = solution.cost()
    print("Done. Cost is {}. {:.4f}s".format(cost, time()-t0))

    # displaying greedy solution
    plt.title("Greedy solution. Cost is {:.2f}".format(cost))
    plt.imshow(display_solution(observed_patient.volumes['t2'].data[:,:,36], observed_patient.labelmaps['t2'].data[:,:,36], solution.match_dict, window_wl=(700,300)))
    patches = [mpatches.Patch(color=value, label=label_text_map[key]) for key,value in label_color_map.items()]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.show()