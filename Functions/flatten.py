
def flatten(arr):
    for x in arr:
        if isinstance(x, list):
            for y in flatten(x):
                yield y
        else:
            yield x


# The flatten function takes in an array and yields each element of the array one by one. If the element is a list, the function recursively calls itself on that list to yield its elements, thus flattening the nested lists and yielding all elements at once.