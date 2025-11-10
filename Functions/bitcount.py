
def bitcount(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count

# This function takes in an integer as its input and counts the number of ones present in the binary representation of that number by using a bitwise AND operation and a loop. It then returns the count as the output.