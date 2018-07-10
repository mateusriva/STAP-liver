"""A collection of utilities for data visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes_lewiner
from skimage.segmentation import mark_boundaries

class IndexTracker(object):
    """This class creates a 3D plot split by slices
    which can be scrolled.
    """
    def __init__(self, ax, X, title="", **kwargs):
        self.ax = ax

        ax.set_title(title)

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind], **kwargs)
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def display_volume(X, **kwargs):
    """Displays a volume X with a scroller"""
    fig,ax=plt.subplots(1,1)
    tracker = IndexTracker(ax, X, **kwargs)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

class ColorIndexTracker(object):
    """This class creates a 3D plot split by slices
    which can be scrolled.
    """
    def __init__(self, ax, X, title="", **kwargs):
        self.ax = ax

        ax.set_title(title)

        self.X = X
        rows, cols, self.slices = X.shape[:-1]
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind,:], **kwargs)
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind,:])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def display_color_volume(X, **kwargs):
    """Displays a volume X with a scroller"""
    fig,ax=plt.subplots(1,1)
    tracker = ColorIndexTracker(ax, X, **kwargs)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

def display_segments_as_lines(volume, labelmap, width=700, level=300, **kwargs):
    """Overlays a labelmap on a volume using red border lines."""
    # window leveling
    leveled = volume.astype(float)
    leveled = (leveled-(level-(width/2)))/(width)
    leveled[leveled < 0.0] = 0.0
    leveled[leveled > 1.0] = 1.0

    # building line overlay from labelmap
    overlayed = np.empty_like(leveled).repeat(3,-1).reshape((leveled.shape[0], leveled.shape[1], leveled.shape[2], 3))
    for i, pack in enumerate(zip(np.rollaxis(leveled,2),np.rollaxis(labelmap,2))):
        leveled_slice, label_slice = pack
        overlayed[...,i,:] = mark_boundaries(leveled_slice, label_slice)

    display_color_volume(overlayed, **kwargs)

def display_solution(observed_labelmap, solution, **kwargs):
    """Displays a predicted solution"""
    # Assembling prediction labelmap
    predicted_labelmap = np.zeros_like(observed_labelmap)
    for element, prediction in enumerate(solution):
        predicted_labelmap[observed_labelmap==element]=prediction
    display_volume(predicted_labelmap, **kwargs)

def represent_srg(object_graph,class_names=None,vertex_range=None):
    """Returns a SRG as an human readable string.
    """
    representation = str(object_graph)
    if vertex_range is None:
        represent_vertices = object_graph.vertices
    else:
        represent_vertices = object_graph.vertices[vertex_range[0]:vertex_range[1]]

    representation += "\n#\tClasses\t"
    for attr in object_graph.vertex_attributes:
        representation += "\t|{}".format(attr)
    representation += "\n"
    for i, vertex in enumerate(represent_vertices):
        if class_names is not None:
            representation += "{}\t{}".format(i,class_names[i])
        else:
            representation += "{}\t\t".format(i)
        for j in range(len(object_graph.vertex_attributes)):
            representation += "\t|{:.3f}\t".format(vertex[j])
        representation += "\n"

    return representation
