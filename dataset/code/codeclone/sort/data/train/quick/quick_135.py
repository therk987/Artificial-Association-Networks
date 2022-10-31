def qsort(L):
	if L == []: return []
	return qsort([x for x in L[1:] if x< L[0]]) + L[0:1] + qsort([x for x in L[1:] if x>=L[0]])
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(qsort(INPUT_VALUE))