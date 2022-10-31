#Bubble Sort

def bubbleSort(list):
    changed = True
    passes = 0

    print("Before: ",list)

    while changed == True:

        changed = False
        passes += 1

        for i in range( len(list) - 1 ):
            if list[i] > list[i+1]:
                changed = True
                list[i], list[i+1] = list[i+1], list[i]

    print("After: ",list,"\ntook",passes,"passes")
    return list


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubbleSort(INPUT_VALUE))