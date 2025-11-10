
def lis(arr):
    ends = {}
    longest = 0

    for i, val in enumerate(arr):

        prefix_lengths = [j for j in range(1, longest + 1) if arr[ends[j]] < val]

        length = max(prefix_lengths) if prefix_lengths else 0

        if length == longest or val < arr[ends[length + 1]]:
            ends[length + 1] = i
            longest = max(longest, length + 1)

    return longest

# The function 'lis' takes in an array and finds the length of the longest increasing subsequence within the array. It does this by keeping track of the indices of the end elements of possible subsequence lengths and updating the longest length whenever a new subsequence is found.