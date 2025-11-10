
def sieve(max):
    primes = []
    for n in range(2, max + 1):
        if all(n % p > 0 for p in primes):
            primes.append(n)
    return primes

# This function implements the Sieve of Eratosthenes algorithm to generate a list of prime numbers up to a specified maximum value. It iterates through all numbers up to the maximum, and only adds them to the list if they are not divisible by any of the previously generated primes.