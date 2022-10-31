def quick(a):
    if len(a) <= 1:
        return a
    pivot = a[len(a)//2]
    left = [x for x in a if x < pivot]
    middle = [x for x in a if x == pivot]
    right = [x for x in a if x > pivot]
    return quick(left) + middle + quick(right)

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick(INPUT_VALUE))