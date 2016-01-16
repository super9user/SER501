# -*- coding: utf-8 -*-
import numpy as numpy  # analysis:ignore
import doctest


# STOCK_PRICES  = [100,113,110,85,105,102,86,63,81,101,94,106,101,79,94,90,97]
SPC = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]


def find_maximum_subarray_brute(A, low=0, high=-1):
    """
    Return a tuple (i,j) where A[i:j] is the maximum subarray.
    Implement the brute force method from chapter 4
    time complexity = O(n^2)

    >>> find_maximum_subarray_brute(SPC, 0, len(SPC) - 1)
    (7, 10)

    >>> find_maximum_subarray_brute([-2, -5, 6, -2, -3, 1, 5, -6], 0, len([-2, -5, 6, -2, -3, 1, 5, -6]) - 1)
    (2, 6)

    >>> find_maximum_subarray_brute([-1, -2, 5, -1, 3, -2, 1], 0, len([-1, -2, 5, -1, 3, -2, 1]) - 1)
    (2, 4)

    >>> find_maximum_subarray_brute([-2, 11, -4, 13, -5, 2], 0, len([-2, 11, -4, 13, -5, 2]) - 1)
    (1, 3)

    """
    i, j = 0, 0  # Indexes for maximum sub-array
    total_sum = -float("inf")  # Total sum for maximum sub-array initialized with -infinity

    for x in range(low, high + 1):
        sum = A[x]  # initializing the variable sum with 1st element of iteration
        for y in range(x + 1, high + 1):
            sum = sum + A[y]
            if sum > total_sum:
                # Found a larger sub-array
                total_sum = sum
                i, j = x, y

    return i, j


def find_maximum_crossing_subarray(A, low, mid, high):
    """
    Find the maximum subarray that crosses mid
    Return a tuple (i,j) where A[i:j] is the maximum subarray.
    """
    left_sum = -float("inf")
    sum = 0
    max_left, max_right = 0, 0
    i = mid
    # Analyzing Left side of array
    while i >= low:
        sum = sum + A[i]
        if sum > left_sum:
            left_sum = sum
            max_left = i
        i = i - 1

    # Analyzing Right side of array
    right_sum = -float("inf")
    sum = 0
    j = mid + 1
    while j <= high:
        sum = sum + A[j]
        if sum > right_sum:
            right_sum = sum
            max_right = j
        j = j + 1

    if right_sum == -float("inf"):
        right_sum = 0
    if left_sum == -float("inf"):
        left_sum = 0

    return max_left, max_right, left_sum + right_sum


def find_maximum_subarray_recursive(A, low=0, high=-1):
    """
    Return a tuple (i,j) where A[i:j] is the maximum subarray.
    Recursive method from chapter 4

    >>> find_maximum_subarray_recursive(SPC, 0, len(SPC) - 1)
    (7, 10, 43)

    >>> find_maximum_subarray_recursive([-2, -5, 6, -2, -3, 1, 5, -6], 0, len([-2, -5, 6, -2, -3, 1, 5, -6]) - 1)
    (2, 6, 7)

    >>> find_maximum_subarray_recursive([-1, -2, 5, -1, 3, -2, 1], 0, len([-1, -2, 5, -1, 3, -2, 1]) - 1)
    (2, 4, 7)

    >>> find_maximum_subarray_recursive([-2, 11, -4, 13, -5, 2], 0, len([-2, 11, -4, 13, -5, 2]) - 1)
    (1, 3, 20)

    """
    if low == high:
        return low, high, A[low]
    else:
        mid = (low + high) / 2
        left_low, left_high, left_sum = find_maximum_subarray_recursive(A, low, mid)
        right_low, right_high, right_sum = find_maximum_subarray_recursive(A, mid + 1, high)
        cross_low, cross_high, cross_sum = find_maximum_crossing_subarray(A, low, mid, high)
        if left_sum >= right_sum and left_sum >= cross_sum:
            # Max-subarray present in left part of array
            return left_low, left_high, left_sum
        elif right_sum >= left_sum and right_sum >= cross_sum:
            # Max-subarray present in right part of array
            return right_low, right_high, right_sum
        else:
            # Max-subarray crosses mid part of array
            return cross_low, cross_high, cross_sum


def find_maximum_subarray_iterative(A, low=0, high=-1):
    """
    Return a tuple (i,j) where A[i:j] is the maximum subarray.
    Do problem 4.1-5 from the book.

    >>> find_maximum_subarray_iterative(SPC, 0, len(SPC) - 1)
    (7, 10)

    >>> find_maximum_subarray_iterative([-2, -5, 6, -2, -3, 1, 5, -6], 0, len([-2, -5, 6, -2, -3, 1, 5, -6]) - 1)
    (2, 6)

    >>> find_maximum_subarray_iterative([-1, -2, 5, -1, 3, -2, 1], 0, len([-1, -2, 5, -1, 3, -2, 1]) - 1)
    (2, 4)

    >>> find_maximum_subarray_iterative([-2, 11, -4, 13, -5, 2], 0, len([-2, 11, -4, 13, -5, 2]) - 1)
    (1, 3)

    """
    sum_now, sum_so_far = 0, 0  # pointer to current sum, pointer largest sum found till now
    start_index_so_far = -float("inf")  # pointer to start index of largest sum found till now
    stop_index_so_far = -float("inf")  # pointer to stop index of largest sum found till now
    start_index_now = -float("inf")  # pointer to start index of current sum
    i = 0
    while i <= high:
        value = sum_now + A[i]
        if value > 0:
            if sum_now == 0:
                start_index_now = i
            sum_now = value
        else:
            sum_now = 0

        if sum_now > sum_so_far:
            # Larger sum found
            sum_so_far = sum_now
            stop_index_so_far = i
            start_index_so_far = start_index_now
        i = i + 1

    return (start_index_so_far, stop_index_so_far)


def square_matrix_multiply(A, B):
    """
    Return the product AB of matrix multiplication.

    >>> square_matrix_multiply([[1, 2], [1, 2]], [[2, 3], [2, 3]])
    array([[6, 9],
           [6, 9]])

    >>> square_matrix_multiply([[9, -2], [11, 7]], [[3, 1], [-11, 1]])
    array([[ 49,   7],
           [-44,  18]])

    >>> square_matrix_multiply([[0, 1, 3], [-11, 32, 1], [8, -2, 0]], [[13, 12, -9], [-11, 1, 32], [-1, -1, -1]])
    array([[ -14,   -2,   29],
           [-496, -101, 1122],
           [ 126,   94, -136]])

    >>> square_matrix_multiply([[0, 2, 4], [4, 8, 0], [82, 3, 5]], [[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    array([[-4, -4, -4],
           [ 4,  4,  4],
           [77, 77, 77]])
    """
    A = numpy.asarray(A)
    B = numpy.asarray(B)

    assert A.shape == B.shape
    assert A.shape == A.T.shape

    height_of_matrix = len(A)

    # Initializing the result Matrix with zeros
    C = numpy.zeros(A.shape, int)

    for x in range(0, height_of_matrix):
        for y in range(0, height_of_matrix):
            sum = 0
            for z in range(0, height_of_matrix):
                sum = sum + A[x][z] * B[z][y]
            C[x][y] = sum

    return C


def square_matrix_multiply_strassens(A, B):
    """
    Return the product AB of matrix multiplication.
    Assume len(A) is a power of 2

    >>> square_matrix_multiply_strassens([[11, 2], [-1, 2]], [[2, 3], [2, 3]])
    array([[26, 39],
           [ 2,  3]])

    >>> square_matrix_multiply_strassens([[1, 1], [1, 1]], [[1, 1], [1, 1]])
    array([[2, 2],
           [2, 2]])

    >>> square_matrix_multiply_strassens([[1,2,3,4],[5,6,7,8],[9,0,1,2],[3,4,5,6]],[[1,2,3,4],[5,6,7,8],[9,0,1,2],[3,4,5,6]])
    array([[ 56,  40,  34,  40],
           [112,  72, 114, 136],
           [ 30,  16,  32,  60],
           [126,  48,  32,  96]])
    """
    A = numpy.asarray(A)
    B = numpy.asarray(B)
    assert A.shape == B.shape
    assert A.shape == A.T.shape
    assert (len(A) & (len(A) - 1)) == 0, "A is not a power of 2"

    size = len(A)
    if size == 1:
        return numpy.asarray([[A[0][0] * B[0][0]]])
    else:

        # Dividing A & B into 4 sub-matrices
        A11 = A[0:size/2, 0:size/2]
        A12 = A[0:size/2, size/2:size]
        A21 = A[size/2:size, 0:size/2]
        A22 = A[size/2:size, size/2:size]

        B11 = B[0:size/2, 0:size/2]
        B12 = B[0:size/2, size/2:size]
        B21 = B[size/2:size, 0:size/2]
        B22 = B[size/2:size, size/2:size]

        # Calculating P1 - P7
        P1 = square_matrix_multiply_strassens(A11, sub(B12, B22))
        P2 = square_matrix_multiply_strassens(B22, add(A11, A12))
        P3 = square_matrix_multiply_strassens(B11, add(A21, A22))
        P4 = square_matrix_multiply_strassens(A22, sub(B21, B11))
        P5 = square_matrix_multiply_strassens(add(A11, A22), add(B11, B22))
        P6 = square_matrix_multiply_strassens(sub(A12, A22), add(B21, B22))
        P7 = square_matrix_multiply_strassens(sub(A11, A21), add(B11, B12))

        # Calculating the elements of result matrix
        # C11 = P5 + P4 - P2 + P6
        C11 = add(add(sub(P4, P2), P6), P5)
        # C12 = P1 + P2
        C12 = add(P1, P2)
        # C21 = P3+ P4
        C21 = add(P3, P4)
        # C22 = P1 + P5 - P3 - P7
        C22 = add(sub(sub(P5, P3), P7), P1)

        # Building the final output
        C = numpy.bmat([[C11, C12], [C21, C22]])
        return numpy.asarray(C)


# this method is used internally for square_matrix_multiply_strassens
def sub(A, B):
    rows = len(A)
    C = numpy.zeros(numpy.asarray(A).shape, int)
    for i in range(0, rows):
        for j in range(0, rows):
            C[i][j] = A[i][j] - B[i][j]
    return numpy.asarray(C)


# this method is used internally for square_matrix_multiply_strassens
def add(A, B):
    rows = len(A)
    C = numpy.zeros(numpy.asarray(A).shape, int)
    for i in range(0, rows):
        for j in range(0, rows):
            C[i][j] = A[i][j] + B[i][j]
    return numpy.asarray(C)


def test():
    print find_maximum_subarray_brute(SPC, 0, len(SPC) - 1)
    print find_maximum_subarray_recursive(SPC, 0, len(SPC) - 1)
    print find_maximum_subarray_iterative(SPC, 0, len(SPC) - 1)
    print square_matrix_multiply(numpy.ones((8, 8), int), numpy.ones((8, 8), int))
    print square_matrix_multiply_strassens(numpy.ones((8, 8)), numpy.ones((8, 8)))
    doctest.testmod()


if __name__ == '__main__':
    test()
