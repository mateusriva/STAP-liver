"""This script contains multiple visual and quantitative explorations
of different supersegmentation algorithms in the context of real MRI.
"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from liver_full_functions import *

import warnings
warnings.filterwarnings("ignore")

initial_weights = (1,1,1,1)
vertex_weights = (1,1,1,1,10)
edge_weights = (1,1,1,1,1)
graph_weights = (1,1)

def compact_watershed(observation_volume_data):
    """Compact watershed with 500 markers.
    """
    # Applying gradient
    smoothed = ndi.gaussian_filter(observation_volume_data, (5,5,1))
    smoothed = smoothed/np.max(smoothed) # normalization for magnitude
    #display_volume(smoothed, cmap="gray")
    magnitude = np.sqrt(ndi.filters.sobel(smoothed, axis=0)**2 + ndi.filters.sobel(smoothed, axis=1)**2 + ndi.filters.sobel(smoothed, axis=2)**2)
    #display_volume(magnitude, cmap="gray", title="Magnitude")
    observed_labelmap_data = watershed(magnitude, markers=1000, compactness=0.001)-1
    return observed_labelmap_data

def local_minima_watershed(observation_volume_data):
    """Watershed with all gradient local minima.
    """
    # Applying gradient
    smoothed = ndi.gaussian_filter(observation_volume_data, (5,5,1))
    smoothed = smoothed/np.max(smoothed) # normalization for magnitude
    #display_volume(smoothed, cmap="gray")
    magnitude = np.sqrt(ndi.filters.sobel(smoothed, axis=0)**2 + ndi.filters.sobel(smoothed, axis=1)**2 + ndi.filters.sobel(smoothed, axis=2)**2)
    #display_volume(magnitude, cmap="gray", title="Magnitude")
    # Getting local minima of the volume
    volume_local_minima = local_minima(magnitude)
    #display_volume(volume_local_minima)
    # Labeling local_minima
    markers, total_markers = ndi.label(volume_local_minima)
    observed_labelmap_data = watershed(magnitude,markers=markers)-1
    return observed_labelmap_data

def h_minima_watershed(observation_volume_data, h):
    """Watershed with h-high gradient local minima.
    """
    # Applying gradient
    smoothed = ndi.gaussian_filter(observation_volume_data, (5,5,1))
    smoothed = smoothed/np.max(smoothed) # normalization for magnitude
    #display_volume(smoothed, cmap="gray")
    magnitude = np.sqrt(ndi.filters.sobel(smoothed, axis=0)**2 + ndi.filters.sobel(smoothed, axis=1)**2 + ndi.filters.sobel(smoothed, axis=2)**2)
    #display_volume(magnitude, cmap="gray", title="Magnitude")
    # Getting local minima of the volume with a height of 0.1
    volume_local_minima = h_minima(magnitude, h=h)
    #display_volume(volume_local_minima)
    # Labeling local_minima
    markers, total_markers = ndi.label(volume_local_minima)
    observed_labelmap_data = watershed(magnitude,markers=markers)-1
    return observed_labelmap_data

def h_01_minima_watershed(observation_volume_data):
    return h_minima_watershed(observation_volume_data, 0.1)

def h_001_minima_watershed(observation_volume_data):
    return h_minima_watershed(observation_volume_data, 0.01)

def h_02_minima_watershed(observation_volume_data):
    return h_minima_watershed(observation_volume_data, 0.2)

def large_minima_watershed(observation_volume_data):
    """Watershed with all gradient local minima., using a structural element of 5x5x3
    """
    # Applying gradient
    smoothed = ndi.gaussian_filter(observation_volume_data, (5,5,1))
    smoothed = smoothed/np.max(smoothed) # normalization for magnitude
    #display_volume(smoothed, cmap="gray")
    magnitude = np.sqrt(ndi.filters.sobel(smoothed, axis=0)**2 + ndi.filters.sobel(smoothed, axis=1)**2 + ndi.filters.sobel(smoothed, axis=2)**2)
    #display_volume(magnitude, cmap="gray", title="Magnitude")
    # Getting local minima of the volume with a structural element 5x5x3
    volume_local_minima = local_minima(magnitude, selem=np.ones((5,5,3)))
    #display_volume(volume_local_minima)
    # Labeling local_minima
    markers, total_markers = ndi.label(volume_local_minima)
    observed_labelmap_data = watershed(magnitude,markers=markers)-1
    return observed_labelmap_data

def slic(observation_volume_data):
    """SLIC superpixel with 500 regions"""
    observed_labelmap_data = skis.slic(observation_volume_data, n_segments=1000,
                        compactness=0.0001, multichannel=False, sigma=(5,5,1))
    return observed_labelmap_data

supersegmentation_functions = [compact_watershed, h_01_minima_watershed,h_001_minima_watershed,h_02_minima_watershed, large_minima_watershed, slic, local_minima_watershed]

# Step 1: Loading data
# -----------------------
model_patient = Patient.build_from_folder("data/4")
model_volume, model_labelmap = model_patient.volumes['t2'], model_patient.labelmaps['t2']
# Reconfiguring model_labelmap with extra backgrounds and unified liver
model_labelmap.data += 1 # Adding space for the automatic "body" label
model_labelmap.data[np.logical_and(model_volume.data < 10, model_labelmap.data == 1)] = 0 # automatic body
model_labelmap.data += 1 # Adding space for the split background
model_labelmap.data[:model_labelmap.data.shape[1]//2,:,:][model_labelmap.data[:model_labelmap.data.shape[1]//2,:,:] == 1] = 0 # splitting background
model_labelmap.data[model_labelmap.data == 3] = 2 # vena cava is body
model_labelmap.data[model_labelmap.data >= 4] = 3 # portal, hepatic veins are 'liver'
# getting center of body
body_center = [int(x) for x in measure_center_of_mass(np.ones_like(model_labelmap.data), labels=model_labelmap.data, index=range(4))[2]]
model_labelmap.data = model_labelmap.data + 7 # adding space for the body divisions
model_labelmap.data[model_labelmap.data == 7] = 0
model_labelmap.data[model_labelmap.data == 8] = 1
# Splitting the body into 8 cubes, based on centroid
model_labelmap.data[:body_center[0],:body_center[1],:body_center[2]][model_labelmap.data[:body_center[0],:body_center[1],:body_center[2]] == 9] = 2
model_labelmap.data[:body_center[0],:body_center[1],body_center[2]:][model_labelmap.data[:body_center[0],:body_center[1],body_center[2]:] == 9] = 3
model_labelmap.data[:body_center[0],body_center[1]:,:body_center[2]][model_labelmap.data[:body_center[0],body_center[1]:,:body_center[2]] == 9] = 4
model_labelmap.data[:body_center[0],body_center[1]:,body_center[2]:][model_labelmap.data[:body_center[0],body_center[1]:,body_center[2]:] == 9] = 5
model_labelmap.data[body_center[0]:,:body_center[1],:body_center[2]][model_labelmap.data[body_center[0]:,:body_center[1],:body_center[2]] == 9] = 6
model_labelmap.data[body_center[0]:,:body_center[1],body_center[2]:][model_labelmap.data[body_center[0]:,:body_center[1],body_center[2]:] == 9] = 7
model_labelmap.data[body_center[0]:,body_center[1]:,:body_center[2]][model_labelmap.data[body_center[0]:,body_center[1]:,:body_center[2]] == 9] = 8

#display_volume(model_labelmap.data, cmap=class_colors)
# display_overlayed_volume(model_volume.data, model_labelmap.data, label_colors=[(0,0,0),(0.1,0.1,0.1),(0.40,0.40,0.40),(0.42,0.42,0.42),(0.44,0.44,0.44),(0.46,0.46,0.46),(0.48,0.48,0.48),(0.50,0.50,0.50),(0.52,0.52,0.52),(0.54,0.54,0.54),(1,0,0),(0,1,0)], title="Model")

observation_volume = deepcopy(model_volume)

# Step 2: Generating model graph
# -----------------------
model_graph = build_graph(model_volume.data, model_labelmap.data)
model_graph, mean_vertex, std_vertex, mean_edge, std_edge = normalize_graph(model_graph)
# print("Model:",represent_srg(model_graph, class_names=class_names))

for supersegmentation_function in supersegmentation_functions:
    print("On", supersegmentation_function.__name__)
    t0 = time()
    # Step 3: Generating observation
    # -----------------------
    observed_labelmap_data = supersegmentation_function(observation_volume.data)

    # display_segments_as_lines(observation_volume.data, observed_labelmap_data, title="{}".format(supersegmentation_function.__name__))
    #display_volume(observed_labelmap_data,cmap=ListedColormap(np.random.rand(255,3)))
    #display_overlayed_volume(observation_volume.data, observed_labelmap_data, label_colors=np.random.rand(255,3),width=1,level=0.5)

    # Step 4: Generating super-observation graph
    # -----------------------
    super_graph = build_graph(observation_volume.data, observed_labelmap_data, add_edges=False)
    super_graph = normalize_graph(super_graph,mean_vertex, std_vertex, mean_edge, std_edge)
    super_adjacency = rag.RAG(observed_labelmap_data)
    # print("Superobservation:",represent_srg(super_graph))
    print("\t{} vertices generated".format(len(super_graph.vertices)))

    # Step 5: Generating optimal solution
    # -----------------------
    solution = np.empty(super_graph.vertices.shape[0])
    for i, super_vertex in enumerate(super_graph.vertices):
        # Computing cost to all model vertices
        labels, counts = np.unique(model_labelmap.data[np.where(observed_labelmap_data == i)], return_counts=True)
        solution[i] = labels[np.argmax(counts)]
    # print("Initial solution:")
    # for i, prediction in enumerate(solution):
    #     print("\t{}: {}".format(i, prediction))


    #print("End of epoch #{}: solution = {}".format(epoch,solution))
    joined_labelmap_data = np.zeros_like(observed_labelmap_data)
    for label, model_vertex in enumerate(model_graph.vertices):
        joined_labelmap_data[np.isin(observed_labelmap_data, np.where(solution==label))]=label
    observation_graph = build_graph(observation_volume.data, joined_labelmap_data, target_vertices=model_graph.vertices.shape[0])
    observation_graph = normalize_graph(observation_graph, mean_vertex, std_vertex, mean_edge, std_edge)
    vertex_costs = compute_vertex_cost(observation_graph.vertices, model_graph.vertices, weights=vertex_weights)
    edge_costs = compute_edge_cost(observation_graph.edges, model_graph.edges, weights=edge_weights)
    dice = (2. * np.logical_and(joined_labelmap_data==10, model_labelmap.data == 10)).sum()/((joined_labelmap_data==10).sum() + (model_labelmap.data == 10).sum())
    print("\tOptimal Solution (Costs: {:.3f},{:.3f}), Dice: {:.4f}".format(np.mean(vertex_costs),np.mean(edge_costs), dice))
    #print("Observation:",represent_srg(observation_graph, class_names=class_names))

    print("\tTotal time taken: {:.2f}s".format(time()-t0))

    # display_volume(joined_labelmap_data, cmap=class_colors, title="Optimal Solution (Costs: {:.3f},{:.3f})".format(np.mean(vertex_costs),np.mean(edge_costs)))
