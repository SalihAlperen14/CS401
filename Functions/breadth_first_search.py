
from collections import deque as Queue

def breadth_first_search(startnode, goalnode):
    queue = Queue()
    queue.append(startnode)

    nodesseen = set()
    nodesseen.add(startnode)

    while queue:
        node = queue.popleft()

        if node is goalnode:
            return True
        else:
            queue.extend(node for node in node.successors if node not in nodesseen)
            nodesseen.update(node.successors)

    return False


# This function implements the Breadth-First Search algorithm to find a path between a start node and a goal node by maintaining a queue of nodes to be visited and a set of nodes that have already been visited. It explores all possible paths from the start node until the goal node is found or all nodes have been visited, returning a boolean value to indicate whether a path has been found.