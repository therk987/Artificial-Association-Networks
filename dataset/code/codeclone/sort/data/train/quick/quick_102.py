#!/usr/bin/python3
def quicksort(l):
    if len(l) > 1:
        pivote = l[0]
        izq = [x for x in l if x < pivote]
        cen = [x for x in l if x == pivote]
        der = [x for x in l if x > pivote]
        return quicksort(izq)+cen+quicksort(der)
    else:
        return l

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quicksort(INPUT_VALUE))