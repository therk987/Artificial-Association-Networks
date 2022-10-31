
def sorting_method(nums_list): #11
    for j in range(1, len(nums_list)): #9
        key = nums_list[j] #7
        i = j-1 #5
        while i>=0 and nums_list[i] > key: #4
            nums_list[i+1] = nums_list[i] #3
            i-=1 #10
        nums_list[i+1] = key #8
    return nums_list #6


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2] #2
print(sorting_method(INPUT_VALUE)) #1