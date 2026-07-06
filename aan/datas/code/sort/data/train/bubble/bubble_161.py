#  File Name: bubble_sort.py
#
#  Language: Python
#
#  Bubble Sort
#
#  Date Created: 8/22/19
#
#  Created By: Justin Snider


def bubble_sort(in_arr):
    for i in range(0, (len(in_arr) - 1)):
        swap = False
        for j in range(0, (len(in_arr) - i - 1)):
            if in_arr[j] > in_arr[j + 1]:
                temp = in_arr[j]
                in_arr[j] = in_arr[j + 1]
                in_arr[j + 1] = temp
                swap = True
        if swap is False:
            break
    return in_arr


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort(INPUT_VALUE))