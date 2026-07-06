"""
    Implementation of MergeSort
"""

def divide(array):
    if(len(array) == 1):
        return array
    a1 = divide(array[0:len(array)//2])
    a2 = divide(array[len(array)//2:])
    return merge(a1,a2)


def merge(a1,a2):
    pointer1 = 0
    pointer2 = 0
    mergedArray = []
    while(pointer1 < len(a1)):
        if(pointer2 == len(a2)):
            break;
        if(a1[pointer1] > a2[pointer2]):
            mergedArray.append(a2[pointer2])
            pointer2 = pointer2 + 1
        else:
            mergedArray.append(a1[pointer1])
            pointer1 = pointer1 + 1
    if(pointer1 < len(a1)):
        for i in range(pointer1,len(a1)):
            mergedArray.append(a1[i])

    if(pointer2 < len(a2)):
        for i in range(pointer2,len(a2)):
            mergedArray.append(a2[i])


    return mergedArray

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(divide(INPUT_VALUE))