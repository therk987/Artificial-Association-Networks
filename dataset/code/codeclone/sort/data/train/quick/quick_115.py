'''
Created on Oct 10, 2013

@author: av
'''



def _find_a_place_for_first_element(list_to_sort, start, end):
    lower_pointer = start
    upper_pointer = end
    while lower_pointer < upper_pointer:
        while lower_pointer < upper_pointer:
            upper_pointer += -1
            if list_to_sort[upper_pointer] < list_to_sort[lower_pointer]:
                list_to_sort[upper_pointer], list_to_sort[lower_pointer] = list_to_sort[lower_pointer], list_to_sort[upper_pointer]
                break
        while lower_pointer < upper_pointer:
            lower_pointer += +1
            if list_to_sort[lower_pointer] > list_to_sort[upper_pointer]:
                list_to_sort[upper_pointer], list_to_sort[lower_pointer] = list_to_sort[lower_pointer], list_to_sort[upper_pointer]
                break
    return list_to_sort, lower_pointer  
                


def my_quick_sort(list_to_sort, start=None, end=None):
    if start == None:
        start = 0
    if end == None:
        end = len(list_to_sort)
    if end-start < 2:
        return list_to_sort
    list_to_sort, first_element_new_position = _find_a_place_for_first_element( list_to_sort, start, end )
    list_to_sort = my_quick_sort(list_to_sort, start, first_element_new_position)
    list_to_sort = my_quick_sort(list_to_sort, first_element_new_position+1, end)
    return list_to_sort

def is_sorted_asc(vector):
    if len(vector) < 2:
        return True
    return vector[0] <= vector[1] and is_sorted_asc(vector[1:])

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(my_quick_sort(INPUT_VALUE))