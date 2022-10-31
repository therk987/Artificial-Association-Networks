def quick_sort(data):
    # recursive case: array size greater than 1
    if(len(data) > 1):
        # calculate the pivot by partitioning
        pivot = partition(data)
        
        # omit the pivot and recursively call quick_sort()

        left = quick_sort(data[:pivot])
        right = quick_sort(data[pivot+1:])
        
        # combine the left and right partitions along with the pivot
        left.append(data[pivot])
        left.extend(right)
        
        return left
    # base case: return array of size 1
    else:
        return data

'''
    This is the partition step of Quick Sort.

    Parameters
    ----------
    data: list
        List of values for which we want to determine the pivot value.
        
    Returns
    -------
    int
'''
def partition(data):
    # set the pivot to be the element at the start index
    pivot = data[0]
    h = 0
    
    # start k at 1 and switch values that are less than the pivot
    for k in range(1, len(data)):
        if(data[k] < pivot):
            h += 1
            swap(data, h, k)

    # put the pivot in its new position
    swap(data, 0, h)
    
    # return the new position of the pivot
    return h

'''
    This is the swap step of Quick Sort.
    
    Parameters
    ----------
    data: list
        The list we're sorting.
    i: int
        The index of the first value.
    j: int
        The index of the second value.
'''
def swap(data, i, j):
    temp = data[i]
    data[i] = data[j]
    data[j] = temp
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick_sort(INPUT_VALUE))