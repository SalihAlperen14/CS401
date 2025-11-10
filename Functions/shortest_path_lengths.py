
from collections import defaultdict

def shortest_path_lengths(n, length_by_edge):
    length_by_path = defaultdict(lambda: float('inf'))
    length_by_path.update({(i, i): 0 for i in range(n)})
    length_by_path.update(length_by_edge)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                length_by_path[i, j] = min(
                    length_by_path[i, j],
                    length_by_path[i, k] + length_by_path[k, j]
                )

    return length_by_path


# The function "shortest_path_lengths" takes in two parameters, the number of vertices and a dictionary representing the length of each edge, and returns a dictionary with the shortest path length between each pair of vertices. It uses nested for loops to iterate through all possible paths and continuously updates the shortest path length for each pair of vertices.