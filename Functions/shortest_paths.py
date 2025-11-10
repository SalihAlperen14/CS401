
def shortest_paths(source, weight_by_edge):
    weight_by_node = {
        v: float('inf') for u, v in weight_by_edge
    }
    weight_by_node[source] = 0

    for i in range(len(weight_by_node) - 1):
        for (u, v), weight in weight_by_edge.items():
            weight_by_node[v] = min(
                weight_by_node[u] + weight,
                weight_by_node[v]
            )

    return weight_by_node


# This function takes in a source node and a dictionary of edge weights, and calculates the shortest paths from the source node to all other nodes in the graph. It does this by using dynamic programming to update the weight of each node with the minimum weight path from the source node to that node.