

def selection_sort(data):

    """
    Applies the selection sort algorithm to a list of integers,
    also printing the data on each iteration to show progress.
    """


    print("Selection Sorting...")

    sorted_to = 0;

    while(sorted_to < len(data) - 1):

        index_of_lowest = find_lowest_index(data, sorted_to)

        swap(data, sorted_to, index_of_lowest)

        sorted_to += 1

    return data


def swap(data, i1, i2):

    """
    A neat little trick to swap integer values without using a third variable
    """

    if i1 != i2:
        data[i1] = data[i1] ^ data[i2]
        data[i2] = data[i1] ^ data[i2]
        data[i1] = data[i1] ^ data[i2]

def find_lowest_index(data, start):

    """
    Finds the index of the lowest item in the unsorted part of the data.
    """

    lowest_index = start

    for i in range(start, len(data)):
        if data[i] < data[lowest_index]:
            lowest_index = i

    return lowest_index
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selection_sort(INPUT_VALUE))