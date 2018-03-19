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
    """
    def __init__(self, vertexes, adjacency_matrix):
        self.vertexes = vertexes
        self.adjacency_matrix = adjacency_matrix

    def __repr__(self):
        return "SRG with {} vertexes and {} edges".format(len(self.vertexes), self.adjacency_matrix.shape)

    @classmethod
    def build_from_labelmap(cls, patient):
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





if __name__ == '__main__':
    from time import time

    print("Loading a single patient... ", end="", flush=True)
    t0 = time()
    model_patient = lic_patient.Patient.build_from_folder("data/4")
    print("Done. {:.4f}s".format(time()-t0))


    print("Building model graph... ", end="", flush=True)
    t0 = time()
    model_graph = SRG.build_from_labelmap(model_patient)
    print("Done. {:.4f}s".format(time()-t0))


    print(model_graph)
    for i, vertex in enumerate(model_graph.vertexes):
        print("Reporting on vertex {}\n---------------------".format(i))
        print(vertex)
        for edge in model_graph.adjacency_matrix[i,:]:
            print(edge)

        print("")


    print("Running watershed... ", end="", flush=True)
    t0 = time()
    watershed_labelmap = model_patient.volumes['t2'].watershed_volume()
    print("Done. {} labels found. {:.4f}s".format(watershed_labelmap.header["num_labels"], time()-t0))


    print("Building observation graph... ", end="", flush=True)
    t0 = time()
    observed_patient = model_patient
    observed_patient.labelmaps['t2'] = watershed_labelmap
    watershed_graph = SRG.build_from_labelmap(observed_patient)
    print("Done. {:.4f}s".format(time()-t0))


    print(watershed_graph)
    for i, vertex in enumerate(watershed_graph.vertexes[:2]):
        print("Reporting on vertex {}\n---------------------".format(i))
        print(vertex)
        for edge in watershed_graph.adjacency_matrix[i,:2]:
            print(edge)

        print("")