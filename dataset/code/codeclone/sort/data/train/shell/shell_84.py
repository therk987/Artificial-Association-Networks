
def shell_sort(integers):
    """The idea of shellSort is to allow exchange of far items. 
    An h-sorted array is made for a large value of h. 
    We keep reducing the value of h until it becomes 1.
    """
    integers_clone = list(integers)
    
    # length of the list
    list_len = len(integers_clone)

    # Select a big initial value of gap
    jumps = list_len // 2

    while jumps >= 1:
        for i in range(jumps, list_len):
            
            temp = integers_clone[i]
            j = i

            while j >= jumps and integers_clone[j-jumps] > temp:
                integers_clone[j] = integers_clone[j-jumps]
                j -= jumps
            integers_clone[j] = temp
        jumps //= 2
    
    return integers_clone
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shell_sort(INPUT_VALUE))