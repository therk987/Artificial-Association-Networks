# selection-sort-in-python

#First declare a function selection_Sort for a list numbers.
def selection_Sort(numbers):
    for i in range(5):
        minpos=i
        for j in range(i,6):
            if numbers[j]<numbers[minpos]:
                minpos=j
        temp=numbers[i]
        numbers[i]=numbers[minpos]
        numbers[minpos]=temp
        print(numbers)
    return numbers
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selection_Sort(INPUT_VALUE))