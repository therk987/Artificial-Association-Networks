#! python3
# Implementation of the Shell sort algorithm (1959).

from math import log2

# This algorithm iterates through the gap sizes from largest to smallest and sorts
# all slices of elements in an array which are the current gap size apart. This
# improves the insertion sort algorithm speed because elements are quickly moved
# large distances instead of having to move through each slot in the list.

def shell_sort(A):
    n = len(A)
    # We generate a list of gaps from largest to smallest. Alternatively, it has been
    # determined experimentally by Marcin Ciura (2001) that the following list of gaps
    # is fastest:
    # ciura_gaps = [701, 301, 132, 57, 23, 10, 4, 1]
    gaps = [2 ** k - 1 for k in range(int(log2(n)), 0, -1)]
    for gap in gaps: # For each gap size...
        for j in range(0, gap): # ...slice the array into gap pieces.
            for i in range(gap + j, n, gap): # Then run an insertion sort on each slice.
                temp = A[i]
                l = i - gap
                while l >= 0 and A[l] > temp:
                    A[l + gap] = A[l]
                    l = l - gap
                A[l + gap] = temp
    return A
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shell_sort(INPUT_VALUE))