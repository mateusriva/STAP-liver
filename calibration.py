"""Functions for calibrating the pipeline with dummies.

This module contains both functions for generating dummies,
and for putting them through the segmentation pipeline."""

from calibration_functions import *

#if __name__ == '__main__':
# Step 1: Loading data (generating dummies)
# -----------------------
#model_dummy, model_labelmap = generate_dummy(3), generate_dummy_label(3)
#model_dummy, model_labelmap = generate_checkerboard_dummy((4,4,2), (30,30,30), np.arange(4*4*2)*50)
#model_dummy, model_labelmap = generate_moons_dummy(20,20,3,1,0.5)
model_dummy, model_labelmap = generate_liver_phantom_dummy()
# TODO automatic model split
model_labelmap[model_labelmap.shape[1]//2:,:,:][model_labelmap[model_labelmap.shape[1]//2:,:,:] == 0] = 4
display_volume(model_dummy, cmap="gray", title="Model Input")
display_volume(model_labelmap, cmap=color_map, title="Model Input")
#observation_dummy = generate_dummy(2)
#observation_dummy = np.random.normal(model_dummy, 20)
#observation_dummy = deepcopy(model_dummy)
observation_dummy = generate_fat_salt_and_pepper_noise(model_dummy, radius=7,amount=0.00001)
observation_dummy[np.logical_or(model_labelmap == 0, model_labelmap == 4)] = 0
#observation_dummy = random_noise(model_dummy, "speckle", seed=10) #amount=0.05)
display_volume(observation_dummy, cmap="gray", title="Observation Input")

# Step 2: Generating model graph
# -----------------------
model_graph = build_graph(model_dummy, model_labelmap)
model_graph, mean_vertex, std_vertex, mean_edge, std_edge = normalize_graph(model_graph)
print("Model:",represent_srg(model_graph))

# Step 3: Generating observation
# -----------------------
# Applying gradient
# smoothed = ndi.gaussian_filter(observation_dummy, (5,5,1))
# magnitude = ndi.morphology.morphological_gradient(smoothed, structure=np.ones((15,15,1)))
# display_volume(magnitude, cmap="gray", title="Magnitude")
# # Getting local minima of the volume with a structural element 5x5x1
# volume_local_minima = local_minima(magnitude,selem=np.ones((11,11,3)))
# #volume_local_minima = h_minima(magnitude, h=threshold_otsu(magnitude))
# # Labeling local_minima
# markers, total_markers = ndi.label(volume_local_minima)
# observed_labelmap = watershed(magnitude,markers=markers)-1
observed_labelmap = skis.slic(observation_dummy, n_segments=400,
                    compactness=0.0001, multichannel=False, sigma=(5,5,1))
display_segments_as_lines(observation_dummy, observed_labelmap, width=1, level=0.5)
#display_volume(observed_labelmap)

# Step 4: Generating super-observation graph
# -----------------------
super_graph = build_graph(observation_dummy, observed_labelmap, add_edges=False)
super_graph = normalize_graph(super_graph,mean_vertex, std_vertex, mean_edge, std_edge)
print("Superobservation:",represent_srg(super_graph))

# Step 5: Generating initial solution
# -----------------------
solution = np.empty(super_graph.vertices.shape[0])
for i, super_vertex in enumerate(super_graph.vertices):
    # Computing cost to all model vertices
    super_vertex_matrix = np.vstack([super_vertex]*model_graph.vertices.shape[0])
    costs = compute_initial_vertex_cost(super_vertex_matrix, model_graph.vertices, weights=(0.1,0.1,0.1,0.7))
    solution[i] = np.argmin(costs)
print("Inital solution:")
for i, prediction in enumerate(solution):
    print("\t{}: {}".format(i, prediction))

# Step 6: Region Joining
# -----------------------
joined_labelmap = np.zeros_like(observed_labelmap)
for element, prediction in enumerate(solution):
    joined_labelmap[observed_labelmap==element]=prediction
observation_graph = build_graph(observation_dummy, joined_labelmap, target_vertices=model_graph.vertices.shape[0])
observation_graph = normalize_graph(observation_graph, mean_vertex, std_vertex, mean_edge, std_edge)
vertex_costs = np.mean(np.linalg.norm(observation_graph.vertices - model_graph.vertices, axis=-1))
edge_costs = np.mean(np.linalg.norm(observation_graph.edges - model_graph.edges, axis=-1))
print("Joined Initial Solution (Costs: {:.3f},{:.3f})".format(vertex_costs,edge_costs))
display_volume(joined_labelmap, cmap=color_map, title="Joined Initial Solution (Costs: {:.3f},{:.3f})".format(vertex_costs,edge_costs))
print("Observation:",represent_srg(observation_graph))

# Step 7: Improvement
# -----------------------
for epoch in range(1):
    for i, super_vertex in enumerate(super_graph.vertices):
        current_prediction = solution[i]
        current_vertex_costs = np.mean(np.linalg.norm(observation_graph.vertices - model_graph.vertices, axis=-1))
        current_edge_costs = np.mean(np.linalg.norm(observation_graph.edges - model_graph.edges, axis=-1))
        current_cost = current_vertex_costs + current_edge_costs
        # sanity check
        if math.isnan(current_cost): current_cost = math.inf

        for j, potential_prediction in enumerate(model_graph.vertices):
            # Skipping same replacements
            if j == current_prediction: continue
            # Replacing the supervertex's labels
            working_labelmap = deepcopy(joined_labelmap)
            working_labelmap[observed_labelmap==i] = j
            #display_volume(working_labelmap,cmap=color_map, title="Replacing {}'s label (curr: {}) with {}".format(i, solution[i],j))
            # Updating graph
            working_graph = build_graph(observation_dummy, working_labelmap, target_vertices=model_graph.vertices.shape[0])
            #print(represent_srg(working_graph))

            # Computing costs
            potential_vertex_costs = np.mean(np.linalg.norm(working_graph.vertices - model_graph.vertices, axis=-1))
            potential_edge_costs = np.mean(np.linalg.norm(working_graph.edges - model_graph.edges, axis=-1))
            potential_cost = potential_vertex_costs + potential_edge_costs
            print("Replacing {}'s label (curr: {}) with {}".format(i, solution[i],j))
            print("\t cost is {:.2f} (current best: {:.2f})".format(potential_cost,current_cost))
            # Improving if better
            if potential_cost < current_cost:
                current_prediction = j
                current_vertex_costs = potential_vertex_costs
                current_edge_costs = potential_edge_costs
                current_cost = potential_cost

        # Replacing best in solution
        solution[i] = current_prediction

    # End of an epoch, rebuilding the joined graph
    print("End of epoch #{}: solution = {}".format(epoch,solution))
    joined_labelmap = np.zeros_like(observed_labelmap)
    for element, prediction in enumerate(solution):
        joined_labelmap[observed_labelmap==element]=prediction
    observation_graph = build_graph(observation_dummy, joined_labelmap)
    vertex_costs = np.mean(np.linalg.norm(observation_graph.vertices - model_graph.vertices, axis=-1))
    edge_costs = np.mean(np.linalg.norm(observation_graph.edges - model_graph.edges, axis=-1))
    print("Joined Epoch #{} Solution (Costs: {:.3f},{:.3f})".format(epoch, vertex_costs,edge_costs))
    display_volume(joined_labelmap, cmap=color_map, title="Joined Epoch #{} Solution (Costs: {:.3f},{:.3f})".format(epoch, vertex_costs,edge_costs))
    print("Observation:",represent_srg(observation_graph))
