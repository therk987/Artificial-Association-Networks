# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:47:07 2016

@author: karl
"""


def merge(array1, array2):
    
    mergeArray = []
    countInversion = 0
    i = 0 # count for array 1
    j = 0 # count for array 2

    lenArray1 = len(array1)
    lenArray2 = len(array2)
    
    while ((i < lenArray1) | (j < lenArray2)):
        if (i == lenArray1):
            mergeArray.append(array2[j])
            j += 1
        elif (j == lenArray2):
            mergeArray.append(array1[i])
            i += 1
        else:
            if (array2[j] < array1[i]):
                mergeArray.append(array2[j])
                countInversion += lenArray1-i
                j += 1
            else:
                mergeArray.append(array1[i])
                i+= 1
    return mergeArray, countInversion

def mergeSortCount(array):
    countInversion = 0
    if len(array) == 1:
        return array, 0
    else:
        array1, countInversion1 = mergeSortCount(array[:len(array)//2])
        array2, countInversion2 = mergeSortCount(array[len(array)//2:])
        arraySorted, countInversionSorted = merge(array1, array2)
        countInversion = countInversion1 + countInversion2 + countInversionSorted
        return arraySorted, countInversion
    
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(mergeSortCount(INPUT_VALUE))