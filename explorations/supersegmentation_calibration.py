"""This script performs supersegmentation experiments, measuring the capabilities of distint algorithms.

For more information, see Riva's dissertation, Section 4.1.2

Authors
-------
 * Mateus Riva (mriva@ime.usp.br)
"""

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from calibration_functions import *

# Experimental parameters
vertex_weights = (1,1,1,1,1)
edge_weights = (1,1,1,1,1)
graph_weights = (1,1)

all_fs=[0,1,2,4]
repetitions=100

def traditional_watershed(observation_dummy):
    """Compact watershed with local minima initial markers"""
    # Smoothing
    smoothed = ndi.gaussian_filter(observation_dummy, (5,5,1))
    # Gradient (magnitude of sobel)
    magnitude = np.sqrt(ndi.filters.sobel(smoothed, axis=0)**2 + ndi.filters.sobel(smoothed, axis=1)**2 + ndi.filters.sobel(smoothed, axis=2)**2)
    # Seeding
    markers = local_minima(magnitude)
    markers = np.logical_or(markers, anisotropic_seeds(observation_dummy.shape, (4,4,4)))
    markers, total_markers = ndi.label(markers)
    # Computing
    return watershed(magnitude, markers=markers, compactness=0.001)-1

def compact_watershed(observation_dummy, n):
    """Compact watershed with evenly-spaced initial markers"""
    # Smoothing
    smoothed = ndi.gaussian_filter(observation_dummy, (5,5,1))
    # Gradient (magnitude of sobel)
    magnitude = np.sqrt(ndi.filters.sobel(smoothed, axis=0)**2 + ndi.filters.sobel(smoothed, axis=1)**2 + ndi.filters.sobel(smoothed, axis=2)**2)
    # Seeding
    markers=anisotropic_seeds(observation_dummy.shape, n)
    markers = ndi.label(markers)[0]
    # Computing
    return watershed(magnitude, markers=markers, compactness=0.001)-1

def compact_watershed_886(observation_dummy):
    return compact_watershed(observation_dummy, (8,8,6))

def compact_watershed_10108(observation_dummy):
    return compact_watershed(observation_dummy, (10,10,8))

def slic(observation_dummy, n):
    """SLIC superpixel"""
    return skis.slic(observation_dummy, n_segments=n,
                        compactness=0.0001, multichannel=False, sigma=(5,5,1), spacing=(6,6,1))

def slic400(observation_dummy):
    return slic(observation_dummy, 400)

def slic600(observation_dummy):
    return slic(observation_dummy, 600)

all_as = [traditional_watershed, compact_watershed_886, compact_watershed_10108, slic400, slic600]

# Step 1
model_dummy, model_labelmap = generate_liver_phantom_dummy()
# Splitting the model: body into 8, bg into 2
body_center = [int(x) for x in measure_center_of_mass(np.ones_like(model_labelmap), labels=model_labelmap, index=range(4))[1]]
model_labelmap = model_labelmap + 8
model_labelmap[model_labelmap.shape[1]//2:,:,:][model_labelmap[model_labelmap.shape[1]//2:,:,:] == 8] = 1
model_labelmap[model_labelmap == 8] = 0
# Splitting the body into 8 cubes, based on centroid
model_labelmap[:body_center[0],:body_center[1],:body_center[2]][model_labelmap[:body_center[0],:body_center[1],:body_center[2]] == 9] = 2
model_labelmap[:body_center[0],:body_center[1],body_center[2]:][model_labelmap[:body_center[0],:body_center[1],body_center[2]:] == 9] = 3
model_labelmap[:body_center[0],body_center[1]:,:body_center[2]][model_labelmap[:body_center[0],body_center[1]:,:body_center[2]] == 9] = 4
model_labelmap[:body_center[0],body_center[1]:,body_center[2]:][model_labelmap[:body_center[0],body_center[1]:,body_center[2]:] == 9] = 5
model_labelmap[body_center[0]:,:body_center[1],:body_center[2]][model_labelmap[body_center[0]:,:body_center[1],:body_center[2]] == 9] = 6
model_labelmap[body_center[0]:,:body_center[1],body_center[2]:][model_labelmap[body_center[0]:,:body_center[1],body_center[2]:] == 9] = 7
model_labelmap[body_center[0]:,body_center[1]:,:body_center[2]][model_labelmap[body_center[0]:,body_center[1]:,:body_center[2]] == 9] = 8
#display_volume(model_dummy, cmap="gray", title="Model Input")
#display_volume(model_labelmap, cmap=color_map, title="Model Input")
#print("Model:",represent_srg(model_graph, class_names=class_names))

# Step 2: Generating model graph
# -----------------------
model_graph = build_graph(model_dummy, model_labelmap)
model_graph, mean_vertex, std_vertex, mean_edge, std_edge = normalize_graph(model_graph)

for a in all_as:
    for f in all_fs:
        amount = 0.00001*f

        costs_rep,dices_liver_rep,dices_average_rep, regions_rep, times_rep=[],[],[],[],[]
        for rep in range(repetitions):
            print("On repetition {}/{} of a={}, f={}".format(rep, repetitions, a.__name__, f))
            t0 = time()
            observation_dummy = generate_fat_salt_and_pepper_noise(model_dummy, radius=7,amount=amount)
            observation_dummy[np.logical_or(model_labelmap == 0, model_labelmap == 1)] = 0
            # display_volume(observation_dummy, cmap="gray", title="Observation Input")

            # Step 3: Generating observation
            # -----------------------
            # Applying gradient
            observed_labelmap = a(observation_dummy)
            regions_count = len(np.unique(observed_labelmap))
            #display_segments_as_lines(observation_dummy, observed_labelmap, width=1, level=0.5)
            #display_segments_as_lines(np.rollaxis(observation_dummy, 2).transpose([0,2,1]), np.rollaxis(observed_labelmap, 2).transpose([0,2,1]), width=1, level=0.5)
            #display_volume(observed_labelmap,cmap=ListedColormap(np.random.rand(255,3)))
            #display_overlayed_volume(observation_dummy, observed_labelmap, label_colors=np.random.rand(255,3),width=1,level=0.5)

            # Step 4: Generating solution
            # -----------------------
            solution = np.empty(regions_count)
            for i in range(regions_count):
                # Computing cost to all model vertices
                labels, counts = np.unique(model_labelmap[np.where(observed_labelmap == i)], return_counts=True)
                solution[i] = labels[np.argmax(counts)]

            # Step 5: Compute cost/accuracy
            # -----------------------

            joined_labelmap = np.zeros_like(observed_labelmap)
            for label, model_vertex in enumerate(model_graph.vertices):
                joined_labelmap[np.isin(observed_labelmap, np.where(solution==label))]=label
            observation_graph = build_graph(observation_dummy, joined_labelmap, target_vertices=model_graph.vertices.shape[0])
            observation_graph = normalize_graph(observation_graph, mean_vertex, std_vertex, mean_edge, std_edge)

            vertex_costs = compute_vertex_cost(observation_graph.vertices, model_graph.vertices, weights=vertex_weights)
            edge_costs = compute_edge_cost(observation_graph.edges, model_graph.edges, weights=edge_weights)

            dice_liver = dice_coefficient(joined_labelmap==11, model_labelmap == 11)
            dice_average = np.mean([dice_coefficient(joined_labelmap==label, model_labelmap == label) for label in range(len(model_graph.vertices))])

            #print("Initial Solution (Costs: {:.3f},{:.3f}, liver Dice: {:.3f}, avg Dice: {:.3f})".format(np.mean(vertex_costs),np.mean(edge_costs), dice_liver, dice_average))
            #print("Observation:",represent_srg(observation_graph, class_names=class_names))
            #display_volume(joined_labelmap, cmap=color_map, title="Contiguous Solution (Costs: {:.3f},{:.3f})".format(np.mean(vertex_costs),np.mean(edge_costs)))

            costs_rep.append((np.mean(vertex_costs) + np.mean(edge_costs))/2)
            dices_liver_rep.append(dice_liver)
            dices_average_rep.append(dice_average)
            regions_rep.append(regions_count)
            times_rep.append(time()-t0)
        np.save("results/supersegmentation/{}_{}-costs.npy".format(a, f), costs_rep)
        np.save("results/supersegmentation/{}_{}-regions.npy".format(a, f), regions_rep)
        np.save("results/supersegmentation/{}_{}-times.npy".format(a, f), times_rep)
        np.save("results/supersegmentation/{}_{}-dices_liver.npy".format(a, f), dices_liver_rep)
        np.save("results/supersegmentation/{}_{}-dices_average.npy".format(a, f), dices_average_rep)
