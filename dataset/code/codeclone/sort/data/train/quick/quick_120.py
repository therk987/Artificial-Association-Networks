
from random import randint, shuffle, choice, randrange, random
def quicksort_val(array):
    """ Stable out-of-place Quicksort.

    This is written by yours truly in pythonic style.

    Parameters
    ----------
    array : list
        a possibly unordered list

    Returns
    -------
    sorted : list
        a sorted list
    """
    if len(array) <= 1:
        return array
    lower, upper, center = [], [], []
    part = choice(array)
    for i in array:
        if i < part:
            lower.append(i)
        elif i > part:
            upper.append(i)
        else:
            center.append(i)
    return quicksort_val(lower) + center + quicksort_val(upper)

def swap(array, i, j):
    """ Helper function for quicksort_ip. """
    array[i], array[j] = array[j], array[i]

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quicksort_val(INPUT_VALUE))