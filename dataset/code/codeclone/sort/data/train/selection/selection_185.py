



def selectionSort(alist):
    for fillslot in range(len(alist)-1,0,-1):
        positionOfMax=0
        for location in range(1,fillslot+1):
            if alist[location]>alist[positionOfMax]:
                positionOfMax = location

        temp = alist[fillslot]
        alist[fillslot] = alist[positionOfMax]
        alist[positionOfMax] = temp

    return alist
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionSort(INPUT_VALUE))