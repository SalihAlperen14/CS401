
def lcs_length(s, t):
    from collections import Counter

    dp = Counter()

    for i in range(len(s)):
        for j in range(len(t)):
            if s[i] == t[j]:
                dp[i, j] = dp[i - 1, j - 1] + 1

    return max(dp.values()) if dp else 0

# This function takes in two strings and calculates the length of the longest common subsequence between them using a dynamic programming approach, which iteratively compares characters in the strings and fills a Counter with the lengths of longest common subsequences for each character index pair. The last line of code returns the maximum value from the Counter (which represents the longest common subsequence's length) of the given strings, or 0 if the Counter is empty.