def insertionSort(arr):
    for i in range(len(arr)):
        j=i 
        while(j>0 and arr[j] < arr[j-1]):
            arr[j], arr[j-1] = arr[j-1], arr[j]
            j -= 1
    print(arr)
    return arr


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertionSort(INPUT_VALUE))