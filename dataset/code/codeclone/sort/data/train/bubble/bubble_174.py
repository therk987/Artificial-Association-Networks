#########################
# !!! SOLUTION CODE !!! #
#########################


#!/bin/python3

import sys


# Write Your Code Here

def bubble_sort(a):
    """
    Displays how many swaps it takes; in addition, the first and
    last element for reference.

    Parameters
    ----------
    a : list
        List of numbers, n is not needed.
    Returns
    -------
    _ : None
    It displays the content so it does not return anything.
    """
    count = 0
    threshold = len(a) - 1
    for _ in range(threshold):
        for i in range(threshold):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                count += 1

    print(f'Array is sorted in {count} swaps.')
    print(f'First Element: {a[0]}')
    print(f'Last Element: {a[-1]}')
    return a


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort(INPUT_VALUE))