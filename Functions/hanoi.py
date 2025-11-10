
def hanoi(height, start=1, end=3):
    steps = []
    if height > 0:
        helper = ({1, 2, 3} - {start} - {end}).pop()
        steps.extend(hanoi(height - 1, start, helper))
        steps.append((start, end))
        steps.extend(hanoi(height - 1, helper, end))

    return steps

# The hanoi function uses recursion to solve the Towers of Hanoi problem, where it moves disks from one peg to another using three pegs. It takes in the height (number of disks), start peg, and end peg as parameters and returns a list of steps to move the disks.