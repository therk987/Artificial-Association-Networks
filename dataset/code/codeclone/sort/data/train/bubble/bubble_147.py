#!/usr/bin/env python3

#count number of swaps needed for bubble sort by comparing adjacent items and exchanging those that are out of order


def countSwaps(a):
    numSwaps = 0
    while True:
        swapsFlag = False
        for i in range(len(a)-1):
            if a[i] > a[i+1]:
                a[i], a[i+1] = a[i+1], a[i]
                numSwaps += 1
                swapsFlag = True
        if not swapsFlag:
            break
    print("Array is sorted in {} swaps.".format(numSwaps))
    print("First Element: {}".format(a[0]))
    print("Last Element: {}".format(a[-1]))
    return a
    


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(countSwaps(INPUT_VALUE))