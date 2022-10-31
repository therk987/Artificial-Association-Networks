
def _partition(numbers, start, end):
    """
    _partition(numbers, start, end)
    Inputs:  A list of numbers, the start and end index for that list
    Outputs:  The pivot index is returned, the list is reordered in place.
    """
    #classic pivot at end implementation
    pivot = numbers[end]
    i = start
    j = start
    while j <= (end-1):
        if numbers[j] <= pivot:
            #swap n[i] and n[j] and increment i
            temp = numbers[i]
            numbers[i] = numbers[j]
            numbers[j] = temp
            i += 1
        j += 1
    #At the end, n[i] will hold a value larger than pivot
    temp = numbers[i]
    numbers[i] = pivot
    numbers[end] = temp
    return i

def quicksort(numbers, start, end):
    """
    quicksort(numbers, start, end)
    Inputs:  A list of numbers, the start and end index for that list
    Outputs:  The list is sorted in place.
    """
    if start < end:
        pivot = _partition(numbers, start, end)
        quicksort(numbers,start,pivot-1)
        quicksort(numbers,pivot+1,end)
    return numbers

def quick(numbers):
    return quicksort(numbers,0,len(numbers)-1)
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick(INPUT_VALUE))