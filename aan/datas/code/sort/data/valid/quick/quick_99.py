def partion(nums,left,right):
    key = nums[left]
    while left < right:
        while left < right and nums[right] >= key:
            right -= 1
        while left < right and nums[left] < key:
            left += 1
        if left < right:
            nums[right],nums[left] = nums[left],nums[right]

    return left


def quick_sort_standard(nums,left,right):
    if left < right:
        key_index = partion(nums,left,right)
        quick_sort_standard(nums,left,key_index)
        quick_sort_standard(nums,key_index+1,right)
    return nums

def quick_sort(nums):
    return quick_sort_standard(nums,0,len(nums)-1)


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick_sort(INPUT_VALUE))