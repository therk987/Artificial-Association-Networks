import random as rn


def bubbleSort(mylist):
    #bubble Sort

    print('Unsorted List:',mylist)
    for k in range(len(mylist)-1,0,-1):
        for i in range(k):
            if mylist[i]>mylist[i+1]:
                temp=mylist[i]
                mylist[i]=mylist[i+1]
                mylist[i+1]=temp

    print('Sorted List:',mylist)
    return mylist
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubbleSort(INPUT_VALUE))