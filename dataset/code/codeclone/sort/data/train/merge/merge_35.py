
# merge sort with recurrise
def merge_sort(List):
    if len(List) <= 1:
        return List
    Left = List[:round(len(List)/2)]
    Right = List[round(len(List)/2):]
    Left_sorted = merge_sort(Left)
    Right_sorted = merge_sort(Right)
    List_sorted =[]
    i = 0
    j = 0
    while i<len(Left_sorted) and j<len(Right_sorted):
        if Left_sorted[i] <= Right_sorted[j]:
            List_sorted.append(Left_sorted[i])
            i +=1
        else:
            List_sorted.append(Right_sorted[j])
            j +=1
    List_sorted.extend(Left_sorted[i:])
    List_sorted.extend(Right_sorted[j:])
    return List_sorted

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))



