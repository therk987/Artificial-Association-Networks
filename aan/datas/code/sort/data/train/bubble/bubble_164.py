def bubble_sort(data):
    length = len(data)
    for i in range(0, length):
        for j in range(i+1, length):
            if (data[j] < data[i]):
                tmp = data[j]
                data[j] = data[i]
                data[i] = tmp
    return data

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort(INPUT_VALUE))