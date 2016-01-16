import sys
import numpy as np
import doctest

INFINITY = sys.maxint
np.set_printoptions(precision=False)

text = ""
cost = 0


def print_neatly(words, M):
    """
    >>> print_neatly(["aaa", "bb", "cc", "ddddd"], 6)
    (28, 'aaa\\nbb cc\\nddddd')

    >>> print_neatly(["aa", "bbb", "cc", "ddddd", "eeeee"], 10)
    (72, 'aa bbb\\ncc ddddd\\neeeee')

    >>> print_neatly(["Hey", "how", "are", "you", "?"], 9)
    (8, 'Hey how\\nare you ?')

    Parameters
    ----------
    words: list of str
        Each string in the list is a word from the file.
    M: int
        The max number of characters per line including spaces

    Returns
    -------
    cost: number
        The optimal value as described in the textbook.
    text: str
        The entire text as one string with newline characters.
        It should not end with a blank line.

    Details
    -------
    Look at print_neatly_test for some code to test the solution.
    """

    global text
    global cost
    text = ""
    cost = 0
    n = len(words)
    cost_matrix = np.zeros(shape=(n + 1, n + 1))

    for i in range(1, n + 1):
        curr_length = len(words[i - 1])
        for j in range(i, n + 1):
            if (M - curr_length) < 0:
                cost_matrix[i][j] = INFINITY
            elif j == n and (M - curr_length) >= 0:
                cost_matrix[i][j] = 0
            else:
                cost_matrix[i][j] = (M - curr_length) ** 3

            if j < len(words):
                curr_length = curr_length + len(words[j]) + 1

    c = np.zeros(shape=(n + 1))
    print_lines = np.zeros(shape=(n + 1))
    c[0] = 0

    for i in range(1, n + 1):
        c[i] = INFINITY
        for j in range(1, i + 1):
            if c[j - 1] != INFINITY and cost_matrix[j][i] != INFINITY:
                if c[j - 1] + cost_matrix[j][i] < c[i]:
                    c[i] = c[j - 1] + cost_matrix[j][i]
                    print_lines[i] = j

    print_lines = print_lines.astype(np.int64)
    build_text(print_lines, n, words, M)

    return cost, text


def build_text(print_lines, n, words, M):
    global text
    global cost
    l = 1
    if print_lines[n] != 1:
        l = build_text(print_lines, print_lines[n] - 1, words, M) + 1

    curr_text = ""
    for i in range(print_lines[n] - 1, n):
        curr_text += words[i]
        if i != n - 1:
            curr_text += " "

    if curr_text not in text:
        text += curr_text
        if len(text.split()) != len(words):
            cost += (M - len(curr_text)) ** 3
            text += "\n"

    return l


# run doctests
doctest.testmod()
