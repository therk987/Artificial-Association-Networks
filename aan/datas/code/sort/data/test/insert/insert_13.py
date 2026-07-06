def insertion_sort(list):
    print("Array values: ", list)
    for i in range(1,len(list)):
        ix = i
        while(ix!=0 and list[ix] < list[ix-1]):
            list[ix-1], list[ix] = list[ix], list[ix-1]
            ix-=1
    print("sorted list insertion: ", list)
    return list

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertion_sort(INPUT_VALUE))