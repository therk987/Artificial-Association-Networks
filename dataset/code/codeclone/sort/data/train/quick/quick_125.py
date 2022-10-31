def quicksort(lst):
    #count_op=0
    smaller=[]
    equal=[]
    bigger=[]
    if len(lst)==0 or len(lst)==1:
        return (lst)
    else:
        pivot=lst[0]
        for n in lst:
            if n<pivot:
                smaller.append(n)
            elif n==pivot:
                equal.append(n)
            else:
                bigger.append(n)
        return (quicksort(smaller)+equal+quicksort(bigger))
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quicksort(INPUT_VALUE))