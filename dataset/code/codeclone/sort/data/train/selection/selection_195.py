def selectionsort(list):
    for i in range(0, len(list)):
        minind = i
        for j in range(i+1, len(list)):
            if list[j] < list[minind]:
                minind = j
        (list[minind], list[i]) = (list[i], list[minind])
    return list
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionsort(INPUT_VALUE))