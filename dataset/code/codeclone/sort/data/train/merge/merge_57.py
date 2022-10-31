#! /usr/bin/env python
# -*- coding: utf-8 -*-


def mergeSort(vector):

    if len(vector)>1:
        mid = len(vector)//2
        lefthalf = vector[:mid]
        righthalf = vector[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                vector[k] = lefthalf[i]
                i += 1
            else:
                vector[k]=righthalf[j]
                j += 1
            k += 1

        while i < len(lefthalf):
            vector[k] = lefthalf[i]
            i += 1
            k += 1

        while j < len(righthalf):
            vector[k]=righthalf[j]
            j += 1
            k += 1

    return vector
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(mergeSort(INPUT_VALUE))