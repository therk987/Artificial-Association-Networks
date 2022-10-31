def inserctionSort(alist):
    for i in range(1,len(alist)):
        p=alist[i]
        j = i
        while j>0 and alist[j-1]>p:
            alist[j]=alist[j-1]
            j=j-1
        alist[j]=p
    return alist

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(inserctionSort(INPUT_VALUE))