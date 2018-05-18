"""
This module contains the Statistical-Relational Graph class.

AS the implementation of the SRG is expected to vary wildly
during development, this docstring will remain empty until
the implementation is consolidated.
"""

import numpy as np

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
        > Statistical: voxel centroid, normalized intensity
        > Relational: voxel vectorial distance

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
            new_vertex = Vertex(label, {"centroid":centroids[label]["voxel"], "mean_intensity": mean_intensities[label]["relative"]})
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
    from copy import deepcopy
    observed_patient = deepcopy(model_patient)
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