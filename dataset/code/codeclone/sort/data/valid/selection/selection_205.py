def selectionSort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[min_idx] > arr[j]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionSort(INPUT_VALUE))