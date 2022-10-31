# QuickSort in Python using O(n) space

# tester array
arr = [7,6,5,4,3,2,1,0]

# partition (pivot) procedure
def partition(arr, start, end): 
    pivot = arr[end]
    partitionIndex = start
    i = start
    while i < end:
        if (arr[i] <= pivot):
            arr[i],arr[partitionIndex] = arr[partitionIndex],arr[i]
            partitionIndex += 1
            i += 1
        else:
            i += 1
            arr[end],arr[partitionIndex]  = arr[partitionIndex],arr[end]
    return partitionIndex

# parent Quicksort algorithm
def quickSort(arr, start, end):
    if (start < end):
        partitionIndex = partition(arr, start, end)
        quickSort(arr, start, partitionIndex-1)
        quickSort(arr, partitionIndex+1, end)
    return arr

def quick(arr):
    n = len(arr)
    return quickSort(arr, 0, n-1)
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick(INPUT_VALUE))