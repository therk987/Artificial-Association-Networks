def merge_sort(A):
    divide(A,0,len(A)-1)
    return A
def merge(arr,initial,middle,final):
    n1 = int(middle-initial + 1)
    n2 = int(final-middle)
    left = [0] * (n1)
    right = [0] * (n2)
    for i in range(0 , n1):
    	left[i] = arr[initial + i]

    for j in range(0 , n2):
    	right[j] = arr[middle +1 + j]
    i = 0     
    j = 0
    k = initial
    while i < n1 and j < n2 :
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    while i < n1:
        arr[k] = left[i]
        i += 1
        k += 1
    while j < n2:
        arr[k] = right[j]
        j += 1
        k += 1
def divide(d,i,f):
    if(i<f):
        m=(i+f)//2
        divide(d,i,m)
        divide(d,m+1,f)
        merge(d,i,m,f)
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))