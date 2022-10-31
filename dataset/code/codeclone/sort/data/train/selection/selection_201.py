

def selectionSort(arr):
    n =  len(arr)
    for i in range(0,n):
        smallestNumber = i
        for j in range(i+1,n):
            if arr[j] < arr[smallestNumber]:
                smallestNumber = j
        arr[i],arr[smallestNumber] = arr[smallestNumber], arr[i]
    return arr
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionSort(INPUT_VALUE))