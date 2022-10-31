def insertionsort(list):
    for i in range(0, len(list)):
        j = i
        val = list[i]
        while(val < list[j-1] and j > 0):
            list[j] = list[j-1]
            j -= 1
        list[j] = val
    return list
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertionsort(INPUT_VALUE))