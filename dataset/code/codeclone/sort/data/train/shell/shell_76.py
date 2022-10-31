

def shellSort(nums):
    h = 1
    n = len(nums)
    while h > 0:
            for i in range(h, n):
                c = nums[i]
                j = i
                while j >= h and c < nums[j - h]:
                    nums[j] = nums[j - h]
                    j = j - h
                    nums[j] = c
            h = int(h / 2.2)
    return nums
    

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shellSort(INPUT_VALUE))