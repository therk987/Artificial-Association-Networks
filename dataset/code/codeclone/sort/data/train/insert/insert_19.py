def insertionSort(alist):
    for index in range(1,len(alist)):
        currentvalue = alist[index]
        position = index
        while position>0 and alist[position-1]>currentvalue:
            alist[position]=alist[position-1]
            position = position-1
        alist[position]=currentvalue
    return alist


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertionSort(INPUT_VALUE))