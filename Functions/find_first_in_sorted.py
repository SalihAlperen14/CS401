
def find_first_in_sorted(arr, x):
    lo = 0
    hi = len(arr)

    while lo < hi:
        mid = (lo + hi) // 2

        if x == arr[mid] and (mid == 0 or x != arr[mid - 1]):
            return mid

        elif x <= arr[mid]:
            hi = mid

        else:
            lo = mid + 1

    return -1

# This function takes in an array and a target value and uses binary search to find the index of the first occurrence of the target value in the sorted array, or returns -1 if the target value is not found in the array. The function achieves this by repeatedly dividing the search space in half until the target value is found or the search space is exhausted.