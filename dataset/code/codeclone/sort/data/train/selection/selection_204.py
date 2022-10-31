#  File Name: selection_sort.py
#
#  Language: Python
#
#  Selection Sort
#
#  Date Created: 8/23/19
#
#  Created By: Justin Snider


def selection_sort (in_arr):
    for i in range(0, len(in_arr)-1):
        min = i
        for j in range(i+1, len(in_arr)):
            if in_arr[min] > in_arr[j]:
                min = j
        temp = in_arr[min]
        in_arr[min] = in_arr[i]
        in_arr[i] = temp

    return in_arr
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selection_sort(INPUT_VALUE))