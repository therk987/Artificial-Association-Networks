
def shellSort(num_list):
    gap = len(num_list) // 2 #Calculates the gap (i)
    while gap > 0: #Ensures the gap is above 0, otherwise program sort would fail
        for i in range(gap, len(num_list)):
            var = num_list[i] #Define var
            b = i #Set to new variable
            while b >= gap and num_list[b - gap] > var: #Identifies if integer needs to be moved
                num_list[b] = num_list[b - gap] #Sets intger to new location in list
                b -= gap #Decreases gap size
            num_list[b] = var #Resets for next loop
        gap //= 2 #Decreases gap and discards remainder
    return num_list #Sends back sorted list to main function



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shellSort(INPUT_VALUE))