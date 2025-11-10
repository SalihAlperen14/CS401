
def depth_first_search(startnode, goalnode):
    nodesvisited = set()

    def search_from(node):
        if node in nodesvisited:
            return False
        elif node is goalnode:
            return True
        else:
            nodesvisited.add(node)
            return any(
                search_from(nextnode) for nextnode in node.successors
            )

    return search_from(startnode)

# This function performs a depth-first search from the given startnode, checking if the goalnode is reached. It uses a set to keep track of visited nodes and recursively calls itself on each successor node until the goalnode is found.