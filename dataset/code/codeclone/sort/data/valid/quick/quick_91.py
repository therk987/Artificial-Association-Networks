

def quicksort(array):
    return _quicksort(array, 0, len(array) - 1)

def _quicksort(array, start, stop):
    if stop - start > 0:
        pivot, left, right = array[int((start + stop) / 2)], start, stop
        while left <= right:
            while array[left] < pivot:
                left += 1
            while array[right] > pivot:
                right -= 1
            if left <= right:
                array[left], array[right] = array[right], array[left]
                left += 1
                right -= 1
        _quicksort(array, start, right)
        _quicksort(array, left, stop)
    return array


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quicksort(INPUT_VALUE))