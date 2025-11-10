
def subsequences(a, b, k):
    if k == 0:
        return [[]]

    ret = []
    for i in range(a, b + 1 - k):
        ret.extend(
            [i] + rest for rest in subsequences(i + 1, b, k - 1)
        )

    return ret


# This function takes in three parameters, "a", "b", and "k", and returns a list of all subsequences of length "k" taken from the range of "a" to "b". It does this by recursively calling itself and using a "for" loop to generate all possible combinations.