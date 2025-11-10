
def shunting_yard(tokens):
    precedence = {
        '+': 1,
        '-': 1,
        '*': 2,
        '/': 2
    }

    rpntokens = []
    opstack = []
    for token in tokens:
        if isinstance(token, int):
            rpntokens.append(token)
        else:
            while opstack and precedence[token] <= precedence[opstack[-1]]:
                rpntokens.append(opstack.pop())
            opstack.append(token)

    while opstack:
        rpntokens.append(opstack.pop())

    return rpntokens


# This function takes in a list of tokens and uses the shunting yard algorithm to convert it into a reverse Polish notation (RPN) output. It checks the precedence of operators and rearranges them accordingly to follow the proper rules of RPN.