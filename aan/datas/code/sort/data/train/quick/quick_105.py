# several sort function 1.fast_sort 2.merge_sort 3...

# fast-sort
def fast_sort(List):
    if len(List) <= 1:
        return List
    label = List[0]
    i = 1
    j = len(List)-1
    while i <= j:
        if List[i]<=label:
            i += 1
        if List[j] >label:
            j -= 1
        if i<j:
            temp = List[i]
            List[i] = List[j]
            List[j] = temp
    A = List[1:min(i,j) + 1]
    B = List[max(j,i):]
    A_sorted = fast_sort(A)
    B_sorted = fast_sort(B)
    List_sorted = []
    List_sorted.extend(A_sorted)
    List_sorted.append(label)
    List_sorted.extend(B_sorted)
    return List_sorted


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(fast_sort(INPUT_VALUE))