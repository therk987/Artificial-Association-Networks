def mergesort(array):
    """ Stable Mergesort (first attempt).

    Written by yours truly but inspired by pseudocode from:
        http://en.wikipedia.org/wiki/Merge_sort

    Has the downside of popping items off of a list from the front, causing
    resizing to happen all the time.

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
    part = len(array)//2
    return merge(mergesort(array[:part]), mergesort(array[part:]))

def merge(array1, array2):
    """ Helper for mergesort."""
    result = []
    while len(array1) or len(array2):
        if len(array1) and len(array2):
            if array1[0] <= array2[0]:
                result.append(array1.pop(0))
            else:
                result.append(array2.pop(0))
        elif len(array1):
            result.append(array1.pop(0))
        else:
            result.append(array2.pop(0))
    return result

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(mergesort(INPUT_VALUE))