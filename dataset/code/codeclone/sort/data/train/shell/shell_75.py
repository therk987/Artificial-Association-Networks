def shellSort(arr):
    arrLen = len(arr)
    gap = int(arrLen / 2)

    while (gap > 0):
        i = gap
        for i in range(gap, arrLen):
            val = arr[i]
            j = i
            while (j - gap >= 0 and val < arr[j - gap]):
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = val
            i += 1
        gap = int(gap/2)
        print(arr)
    print(arr)
    return arr
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shellSort(INPUT_VALUE))