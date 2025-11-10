def detect_cycle(node):
    hare = tortoise = node

    while True:
        if hare is None or hare.successor is None:
            return False

        tortoise = tortoise.successor
        hare = hare.successor.successor

        if hare is tortoise:
            return True



# The function takes a node as an argument and sets two variables, hare and tortoise, equal to that node. It then iterates through a loop, checking if the hare and tortoise are at the same node or if they have reached the end of the list. If they meet at the same node at any point, the function returns True, indicating the presence of a cycle in the linked list. Otherwise, it returns False.