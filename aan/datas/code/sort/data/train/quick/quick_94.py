#Author: chriskbwong
def partition(arr,l,h):
#helper function for quickSort
    i = ( l - 1 )
    x = arr[h]
 
    for j in range(l , h):
        if   arr[j] <= x:
 
            # increment index of smaller element
            i = i+1
            arr[i],arr[j] = arr[j],arr[i]
 
    arr[i+1],arr[h] = arr[h],arr[i+1]
    return (i+1)
 

# arr[] --> Array to be sorted,
# l  --> Starting index,
# h  --> Ending index
def quickSort(arr,l,h):
 
    # Create a stack
    size = h - l + 1
    stack = [0] * (size)
 
    # initialize top of stack
    top = -1
 
    # push initial values of l and h to stack
    top = top + 1
    stack[top] = l
    top = top + 1
    stack[top] = h
 
    # Keep popping from stack while is not empty
    while top >= 0:
 
        # Pop h and l from the list
        h = stack[top]
        top = top - 1
        l = stack[top]
        top = top - 1
 
        # Set pivot element at its correct position in sorted array
        p = partition( arr, l, h )
 
        # If there are elements on left side of pivot, then push left side to stack
        if p-1 > l:
            top = top + 1
            stack[top] = l
            top = top + 1
            stack[top] = p - 1
 
        # does the same as above but for the right side instead
        if p+1 < h:
            top = top + 1
            stack[top] = p + 1
            top = top + 1
            stack[top] = h
    return arr
def quick(arr):
    return quickSort(arr,0,len(arr)-1)

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick(INPUT_VALUE))