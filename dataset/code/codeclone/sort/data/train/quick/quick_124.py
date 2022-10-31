def quicksort(ls):
    if len(ls)<=1:
        return ls
    left = []
    right = []
    pivot = ls.pop(len(ls)-1)
    for i in ls:
        if i < pivot:
            left += [i]
        elif i >= pivot:
            right += [i]
    return quicksort(left)+[pivot]+quicksort(right)

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quicksort(INPUT_VALUE))