
def selectionsort(a):
	for i in range(0, len(a)):
		min = i
		for x in range (i+1, len(a)):
			if a[x] < a[min]:
				min = x
		a[min], a[i] = a[i], a[min]
	return a
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionsort(INPUT_VALUE))