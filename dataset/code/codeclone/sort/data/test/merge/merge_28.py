



def mergeSort(numList):
    left = []
    right = []

    if len(numList) <= 1:
        return numList

    for x in range(len(numList)):
        if x < (len(numList) / 2):
            left.append(numList[x])
        else:
            right.append(numList[x])


    left = mergeSort(left)
    right = mergeSort(right)

    return merge(left, right)


def merge(left, right):
    result = [] #

    while left and right:
        if left[0] <= right[0]: #
            result.append(left[0])
            left = left[1:]
        else:
            result.append(right[0])
            right = right[1:]

    while left:
        result.append(left[0]) #
        left = left[1:]
    while right:
        result.append(right[0]) #
        right = right[1:]

    return result #


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(mergeSort(INPUT_VALUE))