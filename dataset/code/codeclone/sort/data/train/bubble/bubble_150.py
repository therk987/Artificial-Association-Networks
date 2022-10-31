#Bubble Sort
def bubble_sort(alist):
    for j in range(len(alist)):
        for i in range (0,len(alist)-1):
            if alist[i]>alist[i+1]:
                alist[i],alist[i+1]=alist[i+1],alist[i]
                
                i=i+1
    return alist


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort(INPUT_VALUE))