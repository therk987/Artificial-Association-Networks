def selection_sort(arr):
    for i in range(len(arr)):
        min = arr[i]
        temp  = 0
        for j in range(i + 1, len(arr)):
            if arr[j] < min:
                min = arr[j]
                temp = j 
                arr[i], arr[temp] = arr[temp], arr[i]
    return arr

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selection_sort(INPUT_VALUE))