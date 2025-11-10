
def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


# The gcd function is used to find the greatest common divisor of two numbers by recursively finding the remainder of a divided by b until the remainder is zero, at which point the function returns the value of a.