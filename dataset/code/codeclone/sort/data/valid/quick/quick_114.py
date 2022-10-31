

def swap(arr,a,b):
    tmp = arr[a]
    arr[a] = arr[b]
    arr[b] = tmp
    return
def partition(arr, front, end):
    pivot = arr[end]
    i = front - 1
    j = front
    while(j < end):
        if arr[j] < pivot:
            i += 1
            swap(arr, i, j)
        j += 1
    i += 1
    swap(arr, i, end)
    return i

def quick_sort(arr, front, end):
    if front < end:
        pivot = partition(arr, front, end)
        quick_sort(arr, front, pivot - 1)
        quick_sort(arr, pivot+1, end)
    return arr

def quick(arr):
    front = 0
    end = len(arr)-1
    return quick_sort(arr,front,end)
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick(INPUT_VALUE))