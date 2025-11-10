
def knapsack(capacity, items):
    from collections import defaultdict
    memo = defaultdict(int)

    for i in range(1, len(items) + 1):
        weight, value = items[i - 1]

        for j in range(1, capacity + 1):
            memo[i, j] = memo[i - 1, j]

            if weight <= j:
                memo[i, j] = max(
                    memo[i, j],
                    value + memo[i - 1, j - weight]
                )

    return memo[len(items), capacity]


# This function uses the Dynamic Programming approach to solve the Knapsack problem, where it calculates and stores the maximum value that can be obtained with a fixed capacity and a given list of items. This helps reduce the time complexity of the algorithm by avoiding repeated calculations.