def bubbleSort(list):
        print("Unsorted list: ",list)
        for x in range(len(list)-1): #range 0 to length of array (10)-1 = range(9)
                for y in range(len(list)-x-1): #Length of array=10, initial value of x = 0. 10-0-9 = 1. Traverse 0 to 1
                        if (list[y] > list[y+1]):
                                list[y], list[y+1] = list[y+1], list[y]
        return list



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubbleSort(INPUT_VALUE))