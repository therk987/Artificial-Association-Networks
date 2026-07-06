
def quick_sort(integers):
    """Perform a quick sort on a list of integers, selecting a pivot
    point, partition all elements into a first and second part while
    looping so all elements < pivot are in first part, any elements
    > then pivot are in seconds part, recursively sort both half's
    and combine.
    """
    integers_clone = list(integers)

    def helper(arr, first, last):
        """Quick sort helper method for finding pivot/split points in list."""
        if first < last:
            split = partition(arr, first, last)

            helper(arr, first, split - 1)
            helper(arr, split + 1, last)

    def partition(arr, first, last):
        """Generate a partition point for the given array."""
        pivot_value = arr[first]

        left = first + 1
        right = last

        done = False
        while not done:
            while left <= right and arr[left] <= pivot_value:
                left += 1

            while arr[right] >= pivot_value and right >= left:
                right -= 1

            if right < left:
                done = True
            else:
                temp = arr[left]
                arr[left] = arr[right]
                arr[right] = temp

        temp = arr[first]
        arr[first] = arr[right]
        arr[right] = temp

        return right

    helper(integers_clone, 0, len(integers_clone) - 1)
    return integers_clone


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick_sort(INPUT_VALUE))