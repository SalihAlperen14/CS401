
def possible_change(coins, total):
    if total == 0:
        return 1
    if total < 0 or not coins:
        return 0

    first, *rest = coins
    return possible_change(coins, total - first) + possible_change(rest, total)

# Possible_change is a recursive function that calculates the number of ways to make change for a given total amount using specified coins. It checks the base cases of a total of 0 (returning 1) and a total less than 0 or no coins available (returning 0), and otherwise repeatedly calls itself, subtracting the first coin from the total and adding that result to the result of the function for the remaining coins.