import numpy as np
import pandas as pd
import networkx as nx


def nx_greedy_weighted_maximum_matching(G, normalize_edge_costs=False):
    """
    Computes a greedy maximum weight matching for the given graph G.

    :param G: A NetworkX graph with weighted edges
    :param normalize_edge_costs: If True, normalize edge costs using vertex degrees
    :return: A set of matched edges forming a greedy maximum weight matching
    """
    # Get the edges with weights
    edges = list(G.edges(data=True))

    # Sort edges by weight (non-increasing order)
    if normalize_edge_costs:
        # Normalize edge weights: weight / (degree(u) + degree(v))
        edges.sort(key=lambda e: e[2]['weight'] / (G.degree(e[0]) + G.degree(e[1])), reverse=True)
    else:
        edges.sort(key=lambda e: e[2]['weight'], reverse=True)

    matching = set()
    matched_vertices = set()

    # Iterate through sorted edges and apply greedy matching
    matching_weight = 0
    for u, v, data in edges:
        weight = data['weight']
        #if weight > 0 and u != v:
        if u not in matched_vertices and v not in matched_vertices :
            matching.add((u, v))
            matched_vertices.add(u)
            matched_vertices.add(v)
            matching_weight += weight

    return matching

def greedy_weighted_maximum_matching(costs, normalize=False):
    """
    Greedy algorithm for maximum weighted matching based on a cost matrix (Pandas DataFrame).

    :param costs: A pandas DataFrame where costs.iloc[i, j] is the weight of the edge between index[i] and columns[j].
                  Diagonal elements (i==j) should be set to np.inf (infinity).
    :param normalize: If True, normalize the edge costs by the degrees of the vertices.
    :return: A list of tuples representing the matched pairs (based on DataFrame index/columns) and the total weight.
    """
    num_vertices = costs.shape[0]

    # Get the row and column labels
    row_labels = costs.index
    col_labels = costs.columns

    # Initialize the degree of each vertex for normalization if needed
    if normalize:
        degrees = np.sum(np.isfinite(costs.values), axis=1)  # Only count non-inf edges (finite values)

    # Flatten the matrix into a list of (row_index, col_index, weight) tuples using the DataFrame's index/columns
    # Ignore self-loops (infinity values)
    # Normalize the cost
    # Use DataFrame labels
    edges = [
        (row_labels[i], col_labels[j],
         costs.iloc[i, j] / (degrees[i] + degrees[j]) if normalize else costs.iloc[i, j])
        for i in range(num_vertices) for j in range(i + 1, num_vertices) if np.isfinite(costs.iloc[i, j])]

    # Sort edges by weight in descending order
    edges.sort(key=lambda x: x[2], reverse=True)

    matched_vertices = set()
    matching = []
    total_weight = 0

    # Iterate over sorted edges and apply greedy matching
    for row, col, weight in edges:
        if row not in matched_vertices and col not in matched_vertices:
            matching.append((row, col))
            matched_vertices.add(row)
            matched_vertices.add(col)
            total_weight += costs.loc[row, col]  # Use original weight from the DataFrame

    return matching, total_weight


def calculate_features(costs, k, greedy=500):

    size = len(costs)
    iterations = []
    nodes = list(range(size))
    costs = pd.DataFrame(costs, columns=nodes, index=nodes)
    selected_nodes = nodes

    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node, weight=costs.loc[node, node], iteration=0)

    round = 1
    total_cost = 0

    while len(selected_nodes) > 2:
        current_costs = costs.loc[selected_nodes, selected_nodes]
        matrix_size = len(current_costs)
        is_even_size = matrix_size % 2 != 0

        if len(selected_nodes) >= greedy:
            indices, total_weight = greedy_weighted_maximum_matching(costs.loc[selected_nodes, selected_nodes])
        else:
            Gn = nx.Graph()
            Gn.add_nodes_from(selected_nodes)
            # upper triangular matrix
            Gn.add_weighted_edges_from(
                (selected_nodes[i], selected_nodes[j], costs.loc[selected_nodes[i], selected_nodes[j]])
                for i in range(len(selected_nodes)) for j in range(i+1, len(selected_nodes))

            )
            indices = sorted(nx.max_weight_matching(Gn))
        match_with_fake_node = None
        if is_even_size:
            matched_nodes = {i for i, j in indices} | {j for i, j in indices}
            match_with_fake_node = list(set(selected_nodes) - matched_nodes)[0]

        selected_nodes = []

        for a, b in indices:
            if costs.loc[a, a] >= costs.loc[b, b]:
                G.add_edge(a, b, weight=costs.loc[a, b])
                selected_nodes.append(a)
                G.nodes[a]['iteration'] += 1
            else:
                G.add_edge(b, a, weight=costs.loc[b, a])
                selected_nodes.append(b)
                G.nodes[b]['iteration'] += 1

        if match_with_fake_node is not None:
            G.nodes[match_with_fake_node]['iteration'] += 1
            selected_nodes.append(match_with_fake_node)

        iterations.append({"G": G.to_undirected(), "round": round})
        round = round + 1

    if len(selected_nodes) == 2:
        a, b = selected_nodes
        if costs.loc[a, a] >= costs.loc[b, b]:
            G.add_edge(a, b, weight=costs.loc[a, b])
            G.nodes[a]['iteration'] += 1
            root = a
        else:
            G.add_edge(b, a, weight=costs.loc[b, a])
            G.nodes[b]['iteration'] += 1
            root = b
    else:
        G.nodes[selected_nodes[0]]['iteration'] += 1
        root = selected_nodes[0]

    iterations.append({"G": G.to_undirected(), "round": round})

    T = G.copy()

    node_iteration = nx.get_node_attributes(G, 'iteration')
    node_costs = nx.get_node_attributes(G, 'weight')
    sorted_nodes = sorted(node_iteration.keys(), key=lambda k: (node_iteration[k], node_costs[k]),
                          reverse=True)[:k][::-1]

    # break into k clusters
    # Collect edges to be removed
    edges_to_remove = []
    for node in sorted_nodes:
        for u, v in G.in_edges(node):
            edges_to_remove.append((u, v))

    # Remove edges
    for u, v in edges_to_remove:
        G.remove_edge(u, v)

    best_features = np.sort(sorted_nodes)
    model = {"root": root, "components": G, "sorted_nodes": sorted_nodes,
             "tree": T.to_undirected(), "iterations": iterations}
    return best_features, total_cost, model


