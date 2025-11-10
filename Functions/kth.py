
def kth(arr, k):
    pivot = arr[0]
    below = [x for x in arr if x < pivot]
    above = [x for x in arr if x > pivot]

    num_less = len(below)
    num_lessoreq = len(arr) - len(above)

    if k < num_less:
        return kth(below, k)
    elif k >= num_lessoreq:
        return kth(above, k - num_lessoreq)
    else:
        return pivot

# This function recursively sorts the given array by grouping elements smaller than the first element into a "below" list and elements larger than the first element into an "above" list. It then compares the number of elements in the "below" and "above" lists to the given kth position and returns either the kth element from the "below" list, the kth element from the "above" list, or the first element (if k is within the range of the number of elements in the original array).