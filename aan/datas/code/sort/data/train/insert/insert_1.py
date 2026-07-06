
def insertion_sort(arr):
    for index in range(1, len(arr)):
        if (arr[index] < arr[index - 1]):
            remove = arr[index]
            lesser_index = index - 1
            while (remove < arr[lesser_index] and lesser_index >= 0):
                arr[lesser_index + 1] = arr[lesser_index]
                lesser_index = lesser_index - 1
            arr[lesser_index + 1] = remove

    return arr
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertion_sort(INPUT_VALUE))