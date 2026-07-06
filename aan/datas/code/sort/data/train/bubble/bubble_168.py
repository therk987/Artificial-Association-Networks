# bubble sort:

def bubblesort(A):
    for j in range(len(A)-1): 
        for i in range(len(A)-1-j):
            if A[i] > A[i+1]:
                A[i], A[i+1] = A[i+1], A[i]
    return A

INPUT = [1,5,4,3,2,0]
bubblesort(INPUT)
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubblesort(INPUT_VALUE))