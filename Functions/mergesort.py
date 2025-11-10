
def mergesort(arr):
    def merge(left, right):
        result = []
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:] or right[j:])
        return result

    if len(arr) <= 1:
        return arr
    else:
        middle = len(arr) // 2
        left = mergesort(arr[:middle])
        right = mergesort(arr[middle:])
        return merge(left, right)

# This function implements the merge sort algorithm, which uses recursion to divide an array into smaller sub-arrays and then merges them back together in sorted order. The merge function takes in two sorted sub-arrays and combines them into a single sorted array, while the outer mergesort function recursively calls itself to continue dividing and merging the array until the entire array is sorted.