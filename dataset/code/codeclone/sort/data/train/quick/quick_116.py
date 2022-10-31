def quickSort(arr):
    lessThanPivot = []
    equalPivot = []
    greaterThanPivot = []

    if len(arr)>1:
        pivot = arr[len(arr)-1]
        for number in arr:
            if number < pivot:
                lessThanPivot.append(number)
            elif number == pivot:
                equalPivot.append(number)
            else:
                greaterThanPivot.append(number)
        return quickSort(lessThanPivot)+equalPivot+quickSort(greaterThanPivot)
    else:
        return arr
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quickSort(INPUT_VALUE))