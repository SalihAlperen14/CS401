
import string
def to_base(num, b):
    result = ''
    alphabet = string.digits + string.ascii_uppercase
    while num > 0:
        i = num % b
        num = num // b
        result = alphabet[i] + result
    return result

# The to_base function converts the given number into a string representation using the specified base. The while loop divides the number by the base and adds the corresponding character from a predefined alphabet to the result, until the number is reduced to 0.