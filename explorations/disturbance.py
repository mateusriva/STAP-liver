"""This script performs disturbance experiments, correlating cost with accuracy.

For more information, see Riva's dissertation, Section 4.1.1

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

all_ns=[(6,6,4),(7,7,5),(8,8,6)]
repetitions=100

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
observation_dummy = deepcopy(model_dummy)
# display_volume(observation_dummy, cmap="gray", title="Observation Input")

# Step 2: Generating model graph
# -----------------------
model_graph = build_graph(model_dummy, model_labelmap)
model_graph, mean_vertex, std_vertex, mean_edge, std_edge = normalize_graph(model_graph)
#print("Model:",represent_srg(model_graph, class_names=class_names))

for n in all_ns:
    p = math.floor(np.prod(n)/20)
    k = math.floor(np.prod(n)/10)

    costs_rep,dices_liver_rep,dices_average_rep=[],[],[]
    for rep in range(repetitions):
        print("On repetition {} of n={}".format(rep, n))
        # Step 3: Generating observation
        # -----------------------
        # Applying gradient
        smoothed = ndi.gaussian_filter(observation_dummy, (5,5,1))
        #display_volume(smoothed, cmap="gray")
        magnitude = np.sqrt(ndi.filters.sobel(smoothed, axis=0)**2 + ndi.filters.sobel(smoothed, axis=1)**2 + ndi.filters.sobel(smoothed, axis=2)**2)
        #display_volume(magnitude, cmap="gray", title="Magnitude")
        markers=anisotropic_seeds(observation_dummy.shape, n)
        # Forcing vena cava
        for i in [0,10,20,30,40]:
            markers[148,180,i] = 1
        markers = ndi.label(markers)[0]
        observed_labelmap = watershed(magnitude, markers=markers, compactness=0.001)-1
        regions_count = len(np.unique(observed_labelmap))
        #display_segments_as_lines(observation_dummy, observed_labelmap, width=1, level=0.5)
        #display_segments_as_lines(np.rollaxis(observation_dummy, -1), np.rollaxis(observed_labelmap,-1), width=1, level=0.5)
        #display_volume(observed_labelmap,cmap=ListedColormap(np.random.rand(255,3)))
        #display_overlayed_volume(observation_dummy, observed_labelmap, label_colors=np.random.rand(255,3),width=1,level=0.5)

        # Step 4: Generating solution
        # -----------------------
        solution = np.empty(regions_count)
        for i in range(regions_count):
            # Computing cost to all model vertices
            labels, counts = np.unique(model_labelmap[np.where(observed_labelmap == i)], return_counts=True)
            solution[i] = labels[np.argmax(counts)]

        # Step 5: Compute initial cost/accuracy
        # -----------------------
        costs,dices_liver,dices_average=[],[],[]

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
        costs.append((np.mean(vertex_costs) + np.mean(edge_costs))/2)
        dices_liver.append(dice_liver)
        dices_average.append(dice_average)
        #display_volume(joined_labelmap, cmap=color_map, title="Contiguous Solution (Costs: {:.3f},{:.3f})".format(np.mean(vertex_costs),np.mean(edge_costs)))

        # Step 6: disturbances
        # -----------------------
        for epoch in range(k):
            #print("In epoch {}/{}".format(epoch,k))
            # Choosing p elements for disturbance
            disturbed = np.random.choice(range(len(solution)), p)
            # Randomly replacing
            for element in disturbed:
                solution[element] = np.random.randint(len(model_graph.vertices))

            # Computing and storing
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
            costs.append((np.mean(vertex_costs) + np.mean(edge_costs))/2)
            dices_liver.append(dice_liver)
            dices_average.append(dice_average)

        costs_rep.append(costs)
        dices_liver_rep.append(dices_liver)
        dices_average_rep.append(dices_average)
    np.save("results/disturbance/{}-{}-{}-costs.npy".format(n[0],n[1],n[2]), costs_rep)
    np.save("results/disturbance/{}-{}-{}-dices_liver.npy".format(n[0],n[1],n[2]), dices_liver_rep)
    np.save("results/disturbance/{}-{}-{}-dices_average.npy".format(n[0],n[1],n[2]), dices_average_rep)
