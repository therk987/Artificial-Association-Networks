def quick_sort(l):
    
    if len(l) < 2:
        return l
    
    key = l[len(l)//2]
    left = [ item for item in l if item < key]
    middle = [ item for item in l if item == key]
    right = [ item for item in l if item > key]
    return quick_sort(left) + middle + quick_sort(right)
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick_sort(INPUT_VALUE))