# selection sort:

def selectionSort(A):
    for i in range(len(A), 1, -1):
        biggest = 0
        for j in range(1,i):
            if A[biggest] < A[j]:
                biggest = j
        A[biggest], A[i-1] = A[i-1], A[biggest]
    return A

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionSort(INPUT_VALUE))