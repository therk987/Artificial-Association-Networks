def merge(left,right):
    n=len(left)+len(right)
    left.append(float('inf'))
    right.append(float('inf'))
    s=[]
    i=0
    j=0
    for k in range(n):
        if left[j]<right[i]:
            s.append(left[j])
            j+=1
        else:
            s.append(right[i])
            i+=1
    return s
def merge_sort(A):
    if len(A) >= 2:
        p = len(A) // 2
        left=A[:p].copy()
        right = A[p:].copy()
        left = merge_sort(left)
        right = merge_sort(right)
        return merge(left.copy(),right.copy())
    else:
        return A

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))