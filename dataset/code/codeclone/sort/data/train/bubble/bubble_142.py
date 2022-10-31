def bub_sort(a):
	for i in range(0, len(a) - 1):
		for j in range(0, len(a) - 1 - i):
			if a[j + 1] < a[j]:
				tmp = a[j]
				a[j] = a[j+1]
				a[j+1] = tmp
		
	return a

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bub_sort(INPUT_VALUE))




