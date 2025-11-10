
def bucketsort(arr, k):
    counts = [0] * k
    for x in arr:
        counts[x] += 1

    sorted_arr = []
    for i, count in enumerate(counts):
        sorted_arr.extend([i] * count)

    return sorted_arr



# This function implements the bucket sort algorithm to sort an array by creating an auxiliary list of counts and then iterating through that list to create a new, sorted array based on the counts of each element. It takes in two parameters: the array to be sorted and the maximum value in the array, which determines the number of buckets to use in sorting.