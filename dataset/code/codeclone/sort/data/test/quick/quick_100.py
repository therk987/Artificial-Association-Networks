def quick_sort(arr):
	less = []
	pivotList = []
	more = []
	if len(arr) <= 1:
		return arr
	else:
		pivot = arr[0]
		for i in arr:
			if i < pivot:
				less.append(i)
			elif i > pivot:
				more.append(i)
			else:
				pivotList.append(i)
		less = quick_sort(less)
		more = quick_sort(more)
		return less + pivotList + more


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick_sort(INPUT_VALUE))