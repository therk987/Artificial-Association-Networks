#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:48:18 2016

@author: karl
"""

import numpy as np
import timeit

##Quik sort using median element of array as pivot
def quickSort3(listToSort):
    sizeList = len(listToSort)
    nbComparaison = 0
    if sizeList == 1:
        return listToSort, 0
    elif sizeList > 0:
        listCandidat = []

        #odd case
        if (sizeList % 2 == 0):
            mid = (sizeList//2)-1
        #even case
        else: 
            mid = sizeList//2
        #low
        listCandidat.append(listToSort[0])
        #mid
        listCandidat.append(listToSort[mid])
        #high
        listCandidat.append(listToSort[-1])
        iMedian = np.argsort(listCandidat)[len(listCandidat)//2]
        if (iMedian == 0):
            tmp = listToSort[0]
            listToSort[0] = listToSort[0]
            listToSort[0] = tmp
        elif (iMedian == 1):
            tmp = listToSort[mid]
            listToSort[mid] = listToSort[0]
            listToSort[0] = tmp
        elif (iMedian == 2):
            tmp = listToSort[-1]
            listToSort[-1] = listToSort[0]
            listToSort[0] = tmp
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
            listToSort[:i], nbComparaisonSs1 = quickSort3(listToSort[:i])
            nbComparaison = nbComparaison + (len(listToSort[:i]) + nbComparaisonSs1)
        if len(listToSort[i+1:]) > 0:
            listToSort[i+1:], nbComparaisonSs2 = quickSort3(listToSort[i+1:])
            nbComparaison = nbComparaison + (len(listToSort[i+1:]) + nbComparaisonSs2)
        return listToSort, nbComparaison
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quickSort3(INPUT_VALUE))