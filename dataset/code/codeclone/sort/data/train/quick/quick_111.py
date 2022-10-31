def quicksort(array):
    
    if len(array) < 2:
        return array
    else:
        
        pivot = array[0]
        
        less = [i for i in array[1:] if i <= pivot]
        greater = [i for i in array[1:] if i > pivot]
        
        return quicksort(less) + [pivot] + quicksort(greater)

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quicksort(INPUT_VALUE))