def selectionsort(list):
    for i in range(0, len(list)//2):
        maxind = i
        minind = i
        for j in range(i+1, len(list)-i):
            if list[j] < list[minind]:
                minind = j
            if list[j] > list[maxind]:
                maxind = j
        if(maxind == i and minind == len(list)-1-i):
            (list[minind], list[maxind]) = (list[maxind], list[minind])
        elif(minind == len(list)-1-i):
            (list[minind], list[i]) = (list[i], list[minind])
            (list[maxind], list[len(list)-1-i]) = (list[len(list)-1-i], list[maxind])
        else:
            (list[maxind], list[len(list)-1-i]) = (list[len(list)-1-i], list[maxind])
            (list[minind], list[i]) = (list[i], list[minind])
    return list
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionsort(INPUT_VALUE))