
def powerset(arr):
    if arr:
        first, *rest = arr
        rest_subsets = powerset(rest)
        return rest_subsets + [[first] + subset for subset in rest_subsets]
    else:
        return [[]]

# This function takes in an array and recursively generates all possible subsets of that array. It does this by taking the first element, then finding the subsets of the rest of the array and combining them with the first element. It continues this process until all possible subsets are found, ultimately returning a list of all the subsets.