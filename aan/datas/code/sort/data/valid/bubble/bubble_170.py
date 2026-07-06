def bubble_sort(l):
    for i in range(1,len(l)):
        for j in range(0,len(l)-i):
            if(l[j]>l[j+1]):
                t=l[j]
                l[j]=l[j+1]
                l[j+1]=t

    return l

    
        


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort(INPUT_VALUE))