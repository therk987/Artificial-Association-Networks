
def partition(data, p, r):
    x = data[r]
    i = p - 1
    for j in range(p, r):
        if data[j] <= x:
            i = i + 1
            data[i], data[j] = data[j], data[i]
    data[i + 1], data[r] = data[r], data[i + 1]
    return i + 1


def quick_sort(data, p, r):
    if p < r:
        q = partition(data, p, r)
        quick_sort(data, p, q - 1)
        quick_sort(data, q + 1, r)
    return data
def quick(data):
    return quick_sort(data,0,len(data)-1)
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick(INPUT_VALUE))