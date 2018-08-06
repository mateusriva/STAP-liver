"""Full Liver segmentation module for the SRG.

This module contains specific configurations
for the SRG, in order to make it segment livers.

Authors:
 * Mateus Riva (mriva@ime.usp.br)
"""

from liver_full_functions import *

#if __name__ == '__main__':
# Step 1: Loading data
# -----------------------
model_patient = Patient.build_from_folder("data/4")
model_volume, model_labelmap = model_patient.volumes['t2'], model_patient.labelmaps['t2']
# Reconfiguring model_labelmap with extra backgrounds and unified liver
model_labelmap.data += 2 # Adding space for the extra labels at the start
model_labelmap.data[np.logical_and(model_volume.data < 10, model_labelmap.data == 2)] = 0 # posterior background is 0
model_labelmap.data[model_labelmap.data.shape[1]//2:,:,:][model_labelmap.data[model_labelmap.data.shape[1]//2:,:,:] == 0] = 1 # anterior background is 1
model_labelmap.data[model_labelmap.data >= 4] = 4
model_labelmap.header["num_labels"] = 5
#display_overlayed_volume(model_volume.data, model_labelmap.data, label_colors=[(0,0,0),(0.5,0.5,0.5),(1,1,1),(0,0,1),(1,0,0)], title="Model")

observation_volume = deepcopy(model_volume)

# Step 2: Generating model graph
# -----------------------
model_graph = build_graph(model_volume.data, model_labelmap.data)
model_graph, mean_vertex, std_vertex, mean_edge, std_edge = normalize_graph(model_graph)
print("Model:",represent_srg(model_graph, class_names=class_names))

# Step 3: Generating observation
# -----------------------
# Applying gradient
smoothed = ndi.gaussian_filter(observation_volume.data, (5,5,1))
smoothed = smoothed/np.max(smoothed)
display_volume(smoothed, cmap="gray")
magnitude = np.sqrt(ndi.filters.sobel(smoothed, axis=0)**2 + ndi.filters.sobel(smoothed, axis=1)**2 + ndi.filters.sobel(smoothed, axis=2)**2)
display_volume(magnitude, cmap="gray", title="Magnitude")
# Getting local minima of the volume with a structural element 5x5x1
#volume_local_minima = local_minima(magnitude)
volume_local_minima = h_minima(magnitude, h=0.1, selem=np.ones((5,5,3)))
display_volume(volume_local_minima)
# Labeling local_minima
markers, total_markers = ndi.label(volume_local_minima)
observed_labelmap_data = watershed(magnitude,markers=markers)-1
# observed_labelmap_data = skis.slic(observation_volume.data, n_segments=400,
#                    compactness=0.0001, multichannel=False, sigma=(5,5,1))
display_segments_as_lines(observation_volume.data, observed_labelmap_data)
#display_volume(observed_labelmap_data,cmap=ListedColormap(np.random.rand(255,3)))
#display_overlayed_volume(observation_volume.data, observed_labelmap_data, label_colors=np.random.rand(255,3),width=1,level=0.5)
boundary_adjacency = rag.rag_boundary(observed_labelmap_data, magnitude)
def weight_boundary(graph, src, dst, n):
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass
merged_labelmap_data = rag.merge_hierarchical(observed_labelmap_data, boundary_adjacency, 1, True, True, merge_func=merge_boundary, weight_func=weight_boundary)
display_segments_as_lines(observation_volume.data, merged_labelmap_data)
observed_labelmap_data = merged_labelmap_data

# Step 4: Generating super-observation graph
# -----------------------
super_graph = build_graph(observation_volume.data, observed_labelmap_data, add_edges=False)
super_graph = normalize_graph(super_graph,mean_vertex, std_vertex, mean_edge, std_edge)
super_adjacency = rag.RAG(observed_labelmap_data)
#print("Superobservation:",represent_srg(super_graph))

# Step 5: Generating initial solution
# -----------------------
solution = np.empty(super_graph.vertices.shape[0])
solution_costs = np.empty_like(solution)
for i, super_vertex in enumerate(super_graph.vertices):
    # Computing cost to all model vertices
    super_vertex_matrix = np.vstack([super_vertex]*model_graph.vertices.shape[0])
    costs = compute_initial_vertex_cost(super_vertex_matrix, model_graph.vertices, weights=(1,1,1,1))
    solution[i] = np.argmin(costs)
    solution_costs[i] = np.min(costs)
# print("Initial solution:")
# for i, prediction in enumerate(solution):
#     print("\t{}: {}".format(i, prediction))

#solution = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,       1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.,       1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 0.,       1., 1., 1., 0., 0., 0., 4., 1., 1., 1., 0., 0., 0., 1., 1., 4., 1.,       0., 1., 0., 3., 1., 1., 3., 4., 4., 3., 1., 1., 0., 1., 1., 1., 3.,       1., 1., 4., 4., 0., 3., 1., 4., 2., 4., 2., 1., 4., 4., 1., 1., 1.,       1., 1., 4., 1., 1., 1., 1., 1., 1., 4., 1., 1., 1., 1., 1., 1., 1.,       1., 1., 4., 1., 1., 4., 1., 1., 4., 4., 4., 4., 1., 4., 4., 4., 1.,       4., 0., 1., 1., 1., 4., 4., 4., 4., 1., 4., 4., 4., 4., 4., 4., 4.,       4., 1., 1., 4., 4., 4., 4., 1., 4., 4., 4., 4., 4., 4., 4., 4., 4.,       4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,       4., 4., 4., 4., 4., 4., 4., 4., 4., 4.])

# Step 6: Region Joining
# -----------------------
vertex_weights = (10,10,10,1,50)
edge_weights = (10,10,10)
graph_weights = (1,1)
joined_labelmap_data = np.zeros_like(observed_labelmap_data)
for element, prediction in enumerate(solution):
    joined_labelmap_data[observed_labelmap_data==element]=prediction
observation_graph = build_graph(observation_volume.data, joined_labelmap_data, target_vertices=model_graph.vertices.shape[0])
observation_graph = normalize_graph(observation_graph, mean_vertex, std_vertex, mean_edge, std_edge)
vertex_costs = compute_vertex_cost(observation_graph.vertices, model_graph.vertices, weights=vertex_weights)
edge_costs = compute_edge_cost(observation_graph.edges, model_graph.edges, weights=edge_weights)
print("Joined Initial Solution (Costs: {:.3f},{:.3f})".format(np.mean(vertex_costs),np.mean(edge_costs)))
display_volume(joined_labelmap_data, cmap=class_colors, title="Joined Initial Solution (Costs: {:.3f},{:.3f})".format(np.mean(vertex_costs),np.mean(edge_costs)))
print("Observation:",represent_srg(observation_graph, class_names=class_names))

# Step 7: Improvement
# -----------------------
total_epochs = len(solution)//2
improvement_cutoff = 1#len(solution) # TODO: convergence? cutoff by cost difference?
for epoch in range(total_epochs):
    # attempting to improve each vertex, starting from the most expensive
    #for super_vertex_index, _ in sorted(enumerate(solution_costs), key=lambda x: x[1], reverse=True)[:improvement_cutoff]:
    for super_vertex_index in [np.argmax(solution_costs)]:
        current_prediction_index = solution[super_vertex_index]
        current_vertex_costs = compute_vertex_cost(observation_graph.vertices, model_graph.vertices, weights=vertex_weights)
        current_edge_costs = compute_edge_cost(observation_graph.edges, model_graph.edges, weights=edge_weights)
        current_cost = graph_weights[0]*np.mean(current_vertex_costs) + graph_weights[1]*np.mean(current_edge_costs)

        print("Modifying supervertex {} (curr: {}, cost: {:.6f})".format(super_vertex_index, current_prediction_index, current_cost))

        # Soft contiguity: potential predictions may only be neighboring labels
        potential_predictions = set([solution[neighbour] for neighbour in super_adjacency.adj[super_vertex_index].keys()])
        for potential_prediction_index in potential_predictions:
            # Skipping same replacements
            if potential_prediction_index == current_prediction_index: continue

            # Replacing the current prediction with the potential
            working_labelmap_data = deepcopy(joined_labelmap_data)
            working_labelmap_data[observed_labelmap_data==super_vertex_index] = potential_prediction_index

            # Updating graph
            working_graph = build_graph(observation_volume.data, working_labelmap_data, target_vertices=model_graph.vertices.shape[0])
            working_graph = normalize_graph(working_graph, mean_vertex, std_vertex, mean_edge, std_edge)

            # Computing costs
            potential_vertex_costs = compute_vertex_cost(working_graph.vertices, model_graph.vertices, weights=vertex_weights)
            potential_edge_costs = compute_edge_cost(working_graph.edges, model_graph.edges)
            potential_cost = graph_weights[0]*np.mean(potential_vertex_costs) + graph_weights[1]*np.mean(potential_edge_costs)
            print("\tAttempting replace with {}, cost: {:.6f}".format(potential_prediction_index, potential_cost))
            # Improving if better
            if potential_cost < current_cost:
                current_prediction_index = potential_prediction_index
                current_vertex_costs = potential_vertex_costs
                current_edge_costs = potential_edge_costs
                current_cost = potential_cost

        solution[super_vertex_index] = current_prediction_index
        solution_costs[super_vertex_index] = 0#np.mean(current_vertex_costs)

        print("\t* Replaced with {}".format(current_prediction_index))

    # End of an epoch, rebuilding the joined graph
    print("End of epoch #{}".format(epoch))
    #print("End of epoch #{}: solution = {}".format(epoch,solution))
    joined_labelmap_data = np.zeros_like(observed_labelmap_data)
    for element, prediction in enumerate(solution):
        joined_labelmap_data[observed_labelmap_data==element]=prediction
    observation_graph = build_graph(observation_volume.data, joined_labelmap_data, target_vertices=model_graph.vertices.shape[0])
    observation_graph = normalize_graph(observation_graph, mean_vertex, std_vertex, mean_edge, std_edge)
    vertex_costs = compute_vertex_cost(observation_graph.vertices, model_graph.vertices, weights=vertex_weights)
    edge_costs = compute_edge_cost(observation_graph.edges, model_graph.edges, weights=edge_weights)
    print("Epoch {} Solution (Costs: {:.3f},{:.3f})".format(epoch,np.mean(vertex_costs),np.mean(edge_costs)))
    #display_volume(joined_labelmap_data, cmap=class_colors, title="Epoch {} Solution (Costs: {:.3f},{:.3f})".format(epoch, np.mean(vertex_costs),np.mean(edge_costs)))
    print("Observation:",represent_srg(observation_graph, class_names=class_names))

display_volume(joined_labelmap_data, cmap=class_colors, title="Epoch {} Solution (Costs: {:.3f},{:.3f})".format(epoch, np.mean(vertex_costs),np.mean(edge_costs)))

# TODO: histogramas dos atributos
