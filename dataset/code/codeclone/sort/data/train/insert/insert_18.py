
def insertionSort(arr):
    n = len(arr)
    for i in range(1,n):
        for j in range(i,0,-1):
            if arr[j] < arr[j-1]:
                arr[j], arr[j-1] = arr[j-1], arr[j]
    return arr

arr = [5,4,3,5,7,9,20,12,34,57]
print (arr)
arr = insertionSort(arr)
print (arr)

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertionSort(INPUT_VALUE))