"""
This module contains the Statistical-Relational Graph class.

AS the implementation of the SRG is expected to vary wildly
during development, this docstring will remain empty until
the implementation is consolidated.
"""

import numpy as np

class SRG:
    """A Statistical-Relational Graph.

    This class represents a SRG, which may be either a model
    SRG (that is, the 'trained' SRG or the 'template') or an
    observed SRG (that is, acquired from observation of an
    unlabeled image).

    Note: this implementation is lean, for ease of computing.
    An alternative implementation, with multiple classes and
    named dictionaries, can be found at branch `descriptive_srg`.

    Attributes
    ----------
    vertices : `2d-array`
        Graph vertices and corresponding attributes.
        Each line corresponds to a single vertex, keyed
        by the index of the line. Each column corresponds
        to an attribute of the vertex.
    edges : `2d-array`
        Graph edges and correponding attributes.
        Each line corresponds to a single edge, keyed
        by the index of the line `x` as the edge
        between vertices `x/|V|` and `x%|V|`.
    vertex_attributes : `list` of str
        List of vertex attributes' keys, as strings.
    edge_attributes : `list` of str
        List of edge attributes' keys, as strings.
    """
    def __init__(self, vertices, edges, vertex_attributes, edge_attributes):
        self.vertices = vertices
        self.edges = edges

        self.vertex_attributes = vertex_attributes
        self.edge_attributes = edge_attributes

    def __repr__(self):
        return "SRG with {} vertices and {} edges\nStat attrs: {}; Rel attrs: {}".format(self.vertices.shape[0], self.edges.shape[0], self.vertex_attributes, self.edge_attributes)
