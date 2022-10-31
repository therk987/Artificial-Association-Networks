
def merge_sort(unsorted_list):
    if len(unsorted_list) <= 1:
        return unsorted_list
 
    middle = len(unsorted_list) // 2
    left = unsorted_list[:middle]
    right = unsorted_list[middle:]
 
    left = merge_sort(left)
    right = merge_sort(right)
    
    return list(merge(left, right))


#main sort function
def merge(left, right):
    sorted_list = []
    left_index, right_index = 0, 0
    
    while left_index < len(left) and right_index < len(right):
        #make comparisons between left and right arrays and add smallest value to sorted array
        if left[left_index] <= right[right_index]:
            sorted_list.append(left[left_index])
            left_index += 1
        else:
            sorted_list.append(right[right_index])
            right_index += 1
 
    if left:
        sorted_list.extend(left[left_index:])
    if right:
        sorted_list.extend(right[right_index:])

    return sorted_list

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))