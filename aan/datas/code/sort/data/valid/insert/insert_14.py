
def InsertionSort(array):
    for i in range(1, len(array)):
        j = i
        while j > 0 and array[j - 1] > array[j]:
            swap(array, j - 1, j)
            j -= 1

    return array

def swap(array, i, j):
    temp = array[i]
    array[i] = array[j]
    array[j] = temp

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(InsertionSort(INPUT_VALUE))