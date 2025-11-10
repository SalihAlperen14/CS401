
def find_in_sorted(arr, x):
    def binsearch(start, end):
        if start == end:
            return -1
        mid = start + (end - start) // 2
        if x < arr[mid]:
            return binsearch(start, mid)
        elif x > arr[mid]:
            return binsearch(mid + 1, end)
        else:
            return mid

    return binsearch(0, len(arr))

# This function, "find_in_sorted", takes in a sorted array and a target value as parameters. It uses recursion to perform a binary search on the array, returning the index of the target value if found, and -1 if it is not present in the array.