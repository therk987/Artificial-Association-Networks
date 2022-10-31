def mergeSort(L):
    if len(L) <= 1:
        return
    else:
        mid = len(L)//2
        a = L[:mid]
        b = L[mid:]
        for i in range(len(a)):
            a[i] = L[i]
        for i in range(len(b)):
            b[i] = L[len(a) + i]
        mergeSort(a)
        mergeSort(b)
        return merge(a,b,L)

def merge(a, b, L):
    a_count = 0
    b_count = 0
    L_count = 0
    while a_count < len(a) and b_count < len(b):
        if a[a_count] < b[b_count]:
            L[L_count] = a[a_count]
            a_count = a_count + 1
        else:
            L[L_count] = b[b_count]
            b_count = b_count + 1
        L_count = L_count + 1
    while a_count < len(a):
        L[L_count] = a[a_count]
        a_count = a_count + 1
        L_count = L_count + 1
    while b_count < len(b):
        L[L_count] = b[b_count]
        b_count = b_count + 1
        L_count = L_count + 1
    
    return L

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(mergeSort(INPUT_VALUE))