def shellsort(marks):
    "Shell sort using Shell's (original) gap sequence: n/2, n/4, ..., 1."
    gap = len(marks) // 2
    # loop over the gaps
    while(gap > 0):
    # do the insertion sort
        for i in range(gap, len(marks)):
            val = marks[i]
            j = i
            while j >= gap and marks[j - gap] > val:
                marks[j] = marks[j - gap]
                j -= gap
            marks[j] = val
        gap //= 2
    return marks
    
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shellsort(INPUT_VALUE))