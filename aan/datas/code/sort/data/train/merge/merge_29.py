def merge(left, right):
    merged = []
    leftInd = 0
    rightInd = 0
    while leftInd < len(left) and rightInd < len(right):
        if left[leftInd] <= right[rightInd]:
            merged.append(left[leftInd])
            leftInd+=1
        else:
            merged.append(right[rightInd])
            rightInd+=1

    merged+=left[leftInd:]
    merged+=right[rightInd:]
    return merged
    
def sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr)//2
    left = sort(arr[:mid])
    right = sort(arr[mid:])

    return merge(left,right)

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(sort(INPUT_VALUE))