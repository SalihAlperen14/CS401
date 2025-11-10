
def kheapsort(arr, k):
    import heapq

    heap = arr[:k]
    heapq.heapify(heap)

    for x in arr[k:]:
        yield heapq.heappushpop(heap, x)

    while heap:
        yield heapq.heappop(heap)


# The function performs a k-heap sort on an array by creating a min heap of the first k elements and then using the heapq module to push the remaining elements onto the heap and then pop them out in sorted order. This allows for sorting an array with a large amount of elements while only using a fixed amount of memory for the heap.