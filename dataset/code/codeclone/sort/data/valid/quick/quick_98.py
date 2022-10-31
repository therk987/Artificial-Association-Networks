# Author: Krems

import random

# insertion sort is faster then qsort on extremely small sizes
def insertion_sort(a):
    if len(a) == 0:
        return a
    for i in range(1, len(a)):
        tmp = a[i]
        hole_pos = i
        while hole_pos > 0 and a[hole_pos - 1] > tmp:
            a[hole_pos] = a[hole_pos - 1]
            hole_pos -= 1
        a[hole_pos] = tmp
    return a

# iterative version with the stack on the heap
def qsort_iterative(a):
    stack = [[]]
    result = list()
    stack.append(a)
    while stack:
        cur = stack.pop()
        stack.append([])
        if len(cur) < 7:
            result += insertion_sort(cur) + stack.pop()
            continue
        pivot_pos = random.randint(0, len(cur) - 1)
        pivot = (cur[pivot_pos] + cur[0] + cur[len(cur) - 1]) / 3
        left = [x for x in cur if x < pivot]
        middle = [x for x in cur if x == pivot]
        right = [x for x in cur if x > pivot]
        stack.append(right)
        stack.append(middle)
        stack.append(left)
    return result

# lhs must be sorted for correct work
# tail recursion used
def qsort_tail(lhs, rhs):
    if len(rhs) < 7:
        return lhs + insertion_sort(rhs)
    pivot_pos = random.randint(0, len(rhs) - 1)
    pivot = (rhs[pivot_pos] + rhs[0] + rhs[len(rhs) - 1]) / 3
    left = [x for x in rhs if x < pivot]
    middle = [x for x in a if x == pivot]
    right = [x for x in rhs if x > pivot]
    return qsort_tail(lhs + qsort_iterative(left) + middle, right)

# quick sort with randomized pivot
def qsort(a):
    if len(a) < 7:
        return insertion_sort(a)
    pivot_pos = random.randint(0, len(a) - 1)
    pivot = (a[pivot_pos] + a[0] + a[len(a) - 1]) / 3
    left = [x for x in a if x < pivot]
    middle = [x for x in a if x == pivot]
    right = [x for x in a if x > pivot]
    return qsort_tail(qsort_iterative(left) + middle, right)


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(qsort(INPUT_VALUE))