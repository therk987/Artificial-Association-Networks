#! /usr/bin/python3

from random import randint
def quick(list):
    return quickSort(list,0,len(list)-1)

def quickSort(A, p, r):
    if (p < r):
        '''q = partition(A, p, r)
        quickSort(A, p, q-1)
        quickSort(A, q+1, r)'''

        A, q1, q2 = partitionModified(A, p, r)
        A = quickSort(A, p, q1-1)
        A = quickSort(A, q2+1, r)
    return A

def partition(A, p, r):
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if (A[j] <= x):
            i += 1
            temp1 = A[i]
            A[i] = A[j]
            A[j] = temp1
    temp2 = A[i+1]
    A[i+1] = A[r]
    A[r] = temp2
    return i+1

def partitionModified(A, p, r):
    x = A[r]
    i = p - 1
    k = 0
    j = p
    while (j < r-k):
        if (A[j] < x):
            i += 1
            temp1 = A[i]
            A[i] = A[j]
            A[j] = temp1
        if (A[j] == x):
            k += 1
            temp1 = A[j]
            A[j] = A[r-k]
            A[r-k] = temp1
            j -= 1
        j += 1
            
    for l in range(0, k+1):
        temp2 = A[i+1+l]
        A[i+1+l] = A[r-k+l]
        A[r-k+l] = temp2
        
    return A, i+1, i+k+1


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick(INPUT_VALUE))