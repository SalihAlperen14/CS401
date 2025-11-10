
def is_valid_parenthesization(parens):
    depth = 0
    for paren in parens:
        if paren == '(':
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                return False

    return depth == 0

# This function takes in a string of parentheses and checks if it is a valid parenthesization. It does this by keeping track of the depth of the parentheses, increasing it for every opening parenthesis and decreasing it for every closing parenthesis, and returning False if the depth ever goes below 0 (indicating an invalid parenthesization).