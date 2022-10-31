def bubble_sort(marks):
    top=len(marks)
    i=0
    for i in range (0,top):
        for j in range (0,top-1):
            if marks[j]>marks[j+1]:
                temp=marks[j+1]
                marks[j+1]=marks[j]
                marks[j]=temp
            j=j+1
        i=i+1
    return marks


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort(INPUT_VALUE))