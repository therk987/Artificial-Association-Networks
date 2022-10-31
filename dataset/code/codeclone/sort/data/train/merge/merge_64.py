def mergeSort(a):
    """
    Sorts and array of numbers into ascending order.
    """

    if len(a) == 1:
        return a
    else:
        nArrayMid = int(len(a) / 2)
        a1 = a[0:nArrayMid]
        a2 = a[nArrayMid: len(a)]
        return merge(mergeSort(a1), mergeSort(a2));


def merge(a1, a2):
    """
    Merges two arrays of numbers (both of which are expected to be pre-sorted into ascending order),
    into a new array, sorted in ascending order.
    """

    nTotalLength = len(a1) + len(a2)
    aMerged = []
    i = 0
    j = 0

    while len(aMerged) < nTotalLength:
        if i == len(a1):
            aMerged.append(a2[j])
            j = j + 1
        elif j == len(a2):
            aMerged.append(a1[i])
            i = i + 1
        elif a1[i] <= a2[j]:
            aMerged.append(a1[i])
            i = i + 1
        else:
            aMerged.append(a2[j])
            j = j + 1

    return aMerged

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(mergeSort(INPUT_VALUE))