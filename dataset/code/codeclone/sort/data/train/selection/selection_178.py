def selection_sort(marks):
    top=len(marks)


    for i in range(0,top):
        min=i
        for j in range(i+1,top):
            if marks[j] < marks[min]:
                    min=j
        marks[i], marks[min]= marks [min], marks[i]
    return marks

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selection_sort(INPUT_VALUE))