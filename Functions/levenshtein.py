
def levenshtein(source, target):
    if source == '' or target == '':
        return len(source) or len(target)

    elif source[0] == target[0]:
        return levenshtein(source[1:], target[1:])

    else:
        return 1 + min(
            levenshtein(source,     target[1:]),
            levenshtein(source[1:], target[1:]),
            levenshtein(source[1:], target)
        )


# The levenshtein function takes in two strings as parameters and calculates the minimum number of single-character edits (insertions, deletions, or substitutions) needed to change the first string into the second string. The function uses a recursive approach, continually comparing the first character of each string and deciding whether to perform an edit or move to the next character until the entire string has been processed.