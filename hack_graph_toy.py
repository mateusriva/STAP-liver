"""Script for checking if the SRGs are working with simple, toy images."""

import numpy as np
import matplotlib.pyplot as plt, matplotlib.colors as mcolors, matplotlib.patches as mpatches
from skimage.morphology import ball
import scipy.ndimage as ndi
from itertools import permutations

from lic_srg import SRG, Matching
from lic_patient import Patient, Volume, LabelMap
from lic_display import display_solution, label_text_map, label_color_map

np.random.seed(0)

# Slice scroll function!
class IndexTracker(object):
    def __init__(self, ax, X, **kwargs):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

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

def show_slice_plot(X, **kwargs):
    fig,ax=plt.subplots(1,1)
    tracker = IndexTracker(ax, X, **kwargs)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

# generating dummy volume with three spheres of different intensities
dummy_volume_data = np.zeros((400,400,400))
gray1, gray2, gray3 = 100, 200, 400
center1, center2, center3 = (150,150,150),(200,200,200),(290,290,290)
radius1, radius2, radius3 = 100, 50, 100
sphere1 = ball(radius1) * gray1
sphere2 = ball(radius2) * gray2
sphere3 = ball(radius3) * gray3

dummy_volume_data[center1[0]-(radius1):center1[0]+(radius1)+1,
            center1[1]-(radius1):center1[1]+(radius1)+1,
            center1[2]-(radius1):center1[2]+(radius1)+1] += sphere1
dummy_volume_data[center2[0]-(radius2):center2[0]+(radius2)+1,
            center2[1]-(radius2):center2[1]+(radius2)+1,
            center2[2]-(radius2):center2[2]+(radius2)+1] += sphere2
dummy_volume_data[center3[0]-(radius3):center3[0]+(radius3)+1,
            center3[1]-(radius3):center3[1]+(radius3)+1,
            center3[2]-(radius3):center3[2]+(radius3)+1] += sphere3
# hacking overlap
dummy_volume_data[dummy_volume_data==300] = 200

# subsampling dummy volume on the 3rd dimension
dummy_volume_data = dummy_volume_data[:,:,::10]

# building dummy labelmap
dummy_labelmap_data = np.zeros_like(dummy_volume_data, dtype=int)
dummy_labelmap_data[dummy_volume_data == 100] = 1
dummy_labelmap_data[dummy_volume_data == 200] = 2
dummy_labelmap_data[dummy_volume_data == 400] = 3

# splitting background into separate labels
dummy_labelmap_data[:200,:,:][dummy_labelmap_data[:200,:,:]==0] = 4
dummy_labelmap_data[200:,:,:][dummy_labelmap_data[200:,:,:]==0] = 0

# Displaying dummy map
#show_slice_plot(dummy_labelmap_data)

# Blurring volume
dummy_volume_data = ndi.gaussian_filter(dummy_volume_data, (5,5,1))
# Noising volume
dummy_volume_data = dummy_volume_data * (0.1*(np.random.random_sample(dummy_volume_data.shape)+0.95))
dummy_volume_data = ndi.gaussian_filter(dummy_volume_data, (2,2,1))
dummy_volume_data = dummy_volume_data.astype(int)

# assembling dummy patient
dummy_volume = Volume("dummy",header={"dimensions":dummy_volume_data.shape,"spacings":(1,1,2),"initial_position":(0,0,0)},data=dummy_volume_data)
dummy_labelmap = LabelMap("dummy",header={"num_labels":3},data=dummy_labelmap_data)
dummy_patient = Patient("dummy", volumes={"dummy":dummy_volume},labelmaps={"dummy":dummy_labelmap})

# displaying dummy patient volume
#show_slice_plot(dummy_patient.volumes["dummy"].data, cmap="gray")

# Building model SRG
model_graph = SRG.build_from_patient(dummy_patient)
print("built model SRG:")
print(model_graph)
for i, vertex in enumerate(model_graph.vertexes):
    print("Reporting on vertex {}\n---------------------".format(i))
    print(vertex)
    for edge in model_graph.adjacency_matrix[i,:]:
        print(edge)
    print("")

# Building observational SRG
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    watershed_labelmap = dummy_patient.volumes['dummy'].watershed_volume()

#show_slice_plot(watershed_labelmap.data, cmap=mcolors.ListedColormap ( np.random.rand ( 256,3)))

observed_patient = dummy_patient
observed_patient.labelmaps['dummy'] = watershed_labelmap
observation_graph = SRG.build_from_patient(observed_patient)
print("built observation SRG:")
print(observation_graph)
for i, vertex in enumerate(observation_graph.vertexes[:3]):
    print("Reporting on vertex {}\n---------------------".format(i))
    print(vertex)
    for edge in observation_graph.adjacency_matrix[i,:3]:
        print(edge)
    print("")

# generating greedy solution
print("Generating greedy solution...")
# creating empty match dict
match_dict = {}
# for each vertex in the observation graph, find the closest matched model vertex (ignore edge info)
for i, obs_vertex in enumerate(observation_graph.vertexes):
    #best_model_vertex = np.argmin([obs_vertex.cost_to(model_vertex) for model_vertex in model_graph.vertexes])
    best_model_vertex, best_model_cost = None, float("inf")
    print("Finding best for {}".format(obs_vertex))
    # hack: backgrounds are 0
    if np.mean(obs_vertex.attributes["mean_intensity"]) < 0.01:
        best_model_vertex = 0
        print("\tIs background!")
    else:
        for j,model_vertex in enumerate(model_graph.vertexes):
            cost = obs_vertex.cost_to(model_vertex, weights=(0.2,0.8))
            print("\tCost to {} is {:.2f}".format(model_vertex, cost))
            if (cost < best_model_cost):
                best_model_vertex = j
                best_model_cost = cost
    match_dict[i] = best_model_vertex
    print("\t\tbest for {} is {}\n".format(i,best_model_vertex))


# displaying greedy solution
# solution_cube = np.copy(watershed_labelmap.data) #display_solution(observed_patient.volumes["dummy"].data, observed_patient.labelmaps["dummy"].data, match_dict)
# for key, value in match_dict.items():
#     solution_cube[solution_cube==key] = value
# #show_slice_plot(solution_cube)
# fig,axes=plt.subplots(1,3)
# ax1, ax2, ax3=axes
# tracker1 = IndexTracker(ax1, solution_cube)
# fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)
# tracker2 = IndexTracker(ax2, dummy_patient.volumes["dummy"].data, cmap="gray")
# fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)
# tracker3 = IndexTracker(ax3, watershed_labelmap.data, cmap=mcolors.ListedColormap ( np.random.rand ( 256,3)))
# fig.canvas.mpl_connect('scroll_event', tracker3.onscroll)
# plt.show()

def cost(self, weights=None, vertex_weights=None, edge_weights=None):
    # global cost of a solution
    if weights is None:
        weights = (1,1)

    # Computing all vertex distances
    #vertex_distances = sum(self.observation_graph.vertexes[key].cost_to(self.model_graph.vertexes[value], vertex_weights) for key, value in self.match_dict.items())
    vertex_distances = []
    for key, value in self.match_dict.items():
        vertex_cost = self.observation_graph.vertexes[key].cost_to(self.model_graph.vertexes[value], vertex_weights)
        print("cost from ObsV {} to ModV {}: {:.2f}".format(key, value, vertex_cost))
        vertex_distances.append(vertex_cost)
    # Computing all edge distances
    #edge_distances = sum(
    #    self.observation_graph.adjacency_matrix[pair1[0],pair2[0]]
    #    .cost_to(self.model_graph.adjacency_matrix[pair1[1],pair2[1]], edge_weights) 
    #    for pair1, pair2 in permutations(self.match_dict.items(), 2) 
    #        if pair1[0] < pair2[0])
    edge_distances = []
    for pair1, pair2 in permutations(self.match_dict.items(),2):
        if pair1[0] < pair2[0]:
            edge_cost = self.observation_graph.adjacency_matrix[pair1[0],pair2[0]].cost_to(self.model_graph.adjacency_matrix[pair1[1],pair2[1]], edge_weights) 
            if pair1[0] % 3 == 0 and pair2[0] % 3 == 0:
                print("cost from ObsE {}-{} to ModE {}-{}: {:.2f}".format(pair1[0],pair2[0], pair1[1],pair2[1], edge_cost))
            edge_distances.append(edge_cost)

    print("sum of vertex distances: {:.2f}; sum of edge distances: {:.2f}".format(np.sum(vertex_distances), np.sum(edge_distances)))
    return (weights[0]*(np.sum(vertex_distances)) + weights[1]*(np.sum(edge_distances)))/np.sum(weights)

print("Computing cost... ", end="", flush=True)
solution = Matching(match_dict, model_graph, observation_graph)
cost = cost(solution)
print("Done. Cost is {}".format(cost))