def quickSort(array, left, right):
    if left >= right:
        return ;

    pivot = checkArray(array, left, right)
    print(' end pivot %s ' % pivot)
    quickSort(array, left, pivot - 1)
    quickSort(array, pivot + 1, right)
    return array

def swap(array, i, j):
    temp = array[i]
    array[i] = array[j]
    array[j] = temp

def checkArray(array, low, high):
    middle = (low + high) // 2 # get middle index

    # ???????? ???????? ??? ????????? ?? ???????? ???????
    # ??? ???????? ?????? ? ????? ????? ??????? ? ???????? ??????????
    # ? ??????? ??????, ???????? ??????? pivot ? ??????? pivot
    swap(array, middle, high) # set the pivot at the end of array

    wall = low # ????? ?? ????? ???????? ??????, ?????? - ??????

    for i in range(low, high): # ?????????? ?? ????? ???????
        print('w - %s | i - %s | middle - %s ' % (wall, i, array[high]))
        if array[i] <= array[high]:
            swap(array, i, wall) # ???? ???????? ?????? pivot [high]
            # ?? ?????? ??????? wall ? I ? ???????? wall(?????) + 1
            wall += 1 # ????? ???????? ???????? ?????? ??? pivot [high]

    swap(array, wall, high) # ?????? pivot ???????, ????? ????? ?????

    return wall # ?????????? ?????? pivot (????? ?????)


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quickSort(INPUT_VALUE,0,len(INPUT_VALUE)-1))