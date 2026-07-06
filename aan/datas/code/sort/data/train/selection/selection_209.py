
def sort(array):
    for i in range(len(array)):
        for j in range(i+1, len(array)):
            if j >= len(array):
                break

            if array[i] > array[j]:
                temp = array[i]
                array[i] = array[j]
                array[j] = temp
    return array
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(sort(INPUT_VALUE))