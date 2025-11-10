
def pascal(n):
    rows = [[1]]
    for r in range(1, n):
        row = []
        for c in range(0, r + 1):
            upleft = rows[r - 1][c - 1] if c > 0 else 0
            upright = rows[r - 1][c] if c < r else 0
            row.append(upleft + upright)
        rows.append(row)

    return rows


# This function takes an integer n as input and creates a Pascal Triangle with n rows, where each row is a list within a larger list called 'rows'. The values in each row are calculated by adding the values from the row above and to the left and right of the current position. The function ultimately returns the complete Pascal Triangle in the form of the 'rows' list.