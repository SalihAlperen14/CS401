
def longest_common_subsequence(a, b):
    if not a or not b:
        return ''

    elif a[0] == b[0]:
        return a[0] + longest_common_subsequence(a[1:], b[1:])

    else:
        return max(
            longest_common_subsequence(a, b[1:]),
            longest_common_subsequence(a[1:], b),
            key=len
        )


# This function takes in two strings and finds the longest common subsequence between them by recursively removing the first character from each string and comparing them to find the longest sequence. If either string is empty, an empty string is returned, otherwise the function compares all possible subsequences and returns the longest one.