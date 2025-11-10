
def reverse_linked_list(node):
    prevnode = None
    while node:
        nextnode = node.successor
        node.successor = prevnode
        prevnode = node
        node = nextnode
    return prevnode



# This function takes a linked list as an input and reverses the order of the nodes, returning the new head node of the reversed list. It uses a while loop and temporary variables to reassign the pointers of each node to point to the previous node, effectively reversing the direction of the linked list.