"""
@Author: Chandan Sharma.
@GitHub: https://github.com/devchandansh/
"""

"""
Here, bubble sort algorith is implemented by taking user inputs.
It sorted in Ascending Order of the given Input. 
"""

def bubble_sort(unsorted_data):

    # For Sorting as Ascending Order

    for i in range(len(unsorted_data)-1):
        for j in range(len(unsorted_data)-1):
            # Sorted Data for ascending order 
            if unsorted_data[j] > unsorted_data[j+1]:
                unsorted_data[j+1], unsorted_data[j] = unsorted_data[j], unsorted_data[j+1]


    print("New Sorted Data (ascending order): ", unsorted_data)
    print("--------------------------")
    return unsorted_data
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort(INPUT_VALUE))