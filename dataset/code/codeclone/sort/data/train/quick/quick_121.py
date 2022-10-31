def partition(A, p, r):
    # pivot is the first element
    x = A[p]
    k = p
    j = r
    while True:
        while A[k] <= x and k < j:
            k += 1

        while A[j] >= x and j >= k:
            j -= 1

        if k < j:
            # swap A[k] and A[j]
            temp = A[k]
            A[k] = A[j]
            A[j] = temp

        else:
            break

    # swap A[k] and A[j]
    temp = A[p]
    A[p] = A[j]
    A[j] = temp
    return j


# recursive sorting functions
# input: List, first index, last index
def quicksort(A, p, r):
    if p < r:
        q = partition(A, p, r)
        quicksort(A, p, q - 1)
        quicksort(A, q + 1, r)
    return A

def quick(A):
    return quicksort(A,0,len(A)-1)

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick(INPUT_VALUE))