#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

def selectionSort(alist):
    for index in range(0, len(alist)):
        ismall = index
        for i in range(index,len(alist)):
            if alist[ismall] > alist[i]:
                ismall = i
        alist[index], alist[ismall] = alist[ismall], alist[index]

    return alist
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionSort(INPUT_VALUE))