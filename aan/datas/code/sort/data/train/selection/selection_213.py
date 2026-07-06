def selection_sort_(a):
    n=len(a)
    for i in range(n):
        mini = i
        for j in range(i, n):
            if a[j] < a[mini]:
                mini = j
        if i != mini:
            a[i], a[mini] = a[mini], a[i]
    return a

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selection_sort_(INPUT_VALUE))