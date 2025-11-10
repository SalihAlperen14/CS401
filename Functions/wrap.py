
def wrap(text, cols):
    lines = []
    while len(text) > cols:
        end = text.rfind(' ', 0, cols + 1)
        if end == -1:
            end = cols
        line, text = text[:end], text[end:]
        lines.append(line)

    lines.append(text)
    return lines


# This function takes in the parameters 'text' and 'cols' and creates new lines in the text to ensure that each line is no longer than the given number of columns. It does this by finding the last space within the given number of columns and then adding that portion of text as a new line to a list. The function continues this process until all of the text has been split into appropriately sized lines.