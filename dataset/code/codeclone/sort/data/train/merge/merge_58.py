
def merge_sort(integers):
    """Divide and conquer approach to sorting a list. Check left index
    id > right index and return, otherwise, grab new middle variable and call
    method recursively until sorted, then merge the half together
    """
    if len(integers) > 1:
        mid = len(integers) // 2
        left = integers[:mid]
        right = integers[mid:]

        merge_sort(left)
        merge_sort(right)

        # Splitting here.
        i, j, k = 0, 0, 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                integers[k] = left[i]
                i += 1
            else:
                integers[k] = right[j]
                j += 1

            k += 1

        # Merging here.
        while i < len(left):
            integers[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            integers[k] = right[j]
            j += 1
            k += 1

    return integers

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))


