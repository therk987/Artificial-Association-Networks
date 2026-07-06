
def mergesort(array):
    n = len(array)
    newArray = []

    y = 1
    while ( y < n):
        i = 0
        newArray = []
        while ( i < n):
            merge(array, i, min(i + y, n), min(i + 2*y, n), newArray) 
            i = i + 2 * y
        y =  y * 2

        array = newArray
    return array
def merge(array, leftIndex, rightIndex, endPoint, newArray):

    i = leftIndex
    j = rightIndex

    k = leftIndex
    while (k < endPoint):
        # Declare working variables to perform the switch if the values are different
        if i < rightIndex and ( j >= endPoint or array[i] <= array[j] ):
            newArray.append(array[i])
            i = i + 1
        else: 
            newArray.append(array[j])
            j = j + 1

        k = k + 1

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(mergesort(INPUT_VALUE))