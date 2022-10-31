import random

def merge_sort(array):
    ret = list()
    pivot = len(array)//2
    i, j = 0, 0
    a1 = array[:pivot]
    a2 = array[pivot:]

    if len(a1) != 1:
        a1 = merge_sort(a1)

    if len(a2) != 1:
        a2 = merge_sort(a2)

    for c in range(0, len(array)):
        if i == len(a1):
            ret.append(a2[j])
            j += 1
        elif j == len(a2):
            ret.append(a1[i])
            i += 1
        elif a1[i] < a2[j]:
            ret.append(a1[i])
            i += 1
        else:
            ret.append(a2[j])
            j += 1

    return ret

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))