#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:48:18 2016

@author: karl
"""

import numpy as np
import timeit


##Quik sort using first element of array as pivot
def quickSort(listToSort):
    sizeList = len(listToSort)
    nbComparaison = 0
    if sizeList == 1:
        return listToSort, 0
    elif sizeList > 0:
        pivot = listToSort[0]
        i = 0
        j = 1
        while j < sizeList:
            if listToSort[j] < pivot:
                i += 1
                tmp = listToSort[i]
                listToSort[i] = listToSort[j]
                listToSort[j] = tmp
            j += 1 
        tmp = listToSort[i]
        listToSort[i] = pivot
        listToSort[0] = tmp
        if len(listToSort[:i]) > 0:
            listToSort[:i], nbComparaisonSs1 = quickSort(listToSort[:i])
            nbComparaison += len(listToSort[:i]) + nbComparaisonSs1
        if len(listToSort[i+1:]) > 0:
            listToSort[i+1:], nbComparaisonSs2 = quickSort(listToSort[i+1:])
            nbComparaison += len(listToSort[i+1:]) + nbComparaisonSs2
        return listToSort, nbComparaison

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quickSort(INPUT_VALUE))