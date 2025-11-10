
def get_factors(n):
    if n == 1:
        return []

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return [i] + get_factors(n // i)

    return [n]


# This function takes in an integer as a parameter and recursively finds all the prime factors of that number. It checks every number from 2 to the square root of the input number and returns a list containing all the prime factors found.