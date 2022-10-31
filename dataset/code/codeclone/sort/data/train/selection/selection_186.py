# Selection sort taking a Python3 list as input
def selectionSort(A):
    for i in range(len(A)):
        # Set minimum index to i value
        min_idx = i

        # Find index with minimum value from i+1 to end of array
        for j in range(i+1, len(A)):
            if A[j] < A[min_idx]:
                min_idx = j

        # Swap value at index i with value at minimum index
        holder = A[i]
        A[i] = A[min_idx]
        A[min_idx] = holder
    return A


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionSort(INPUT_VALUE))