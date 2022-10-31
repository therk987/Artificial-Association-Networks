import random
import datetime


def bubbleSort(arr):
    before = datetime.datetime.now()
    count = len(arr) - 1
    while count > 0:
        for i in range(0, len(arr) - 1):
            if arr[i] > arr[i+1]:
                temp = arr[i]
                arr[i] = arr[i+1]
                arr[i+1] = temp
        count -=1

    after = datetime.datetime.now()
    duration = after - before
    print(arr)
    return arr


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubbleSort(INPUT_VALUE))