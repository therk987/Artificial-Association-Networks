
def merge_sort(A):
	return merge_sort2(A, 0, len(A)-1)

def merge_sort2(A, first, last):
	threshold = 20
	if last-first < threshold and first < last:
		quick_selection(A, first, last)
	elif first < last:
		middle = (first + last)//2
		merge_sort2(A, first, middle)
		merge_sort2(A, middle+1, last)
		merge(A, first, middle, last)
	return A

def merge(A, first, middle, last):
	L = A[first:middle]
	R = A[middle:last+1]
	L.append(999999999)
	R.append(999999999)
	i = j = 0

	for k in range (first, last+1):
		if L[i] <= R[j]:
			A[k] = L[i]
			i += 1
		else:
			A[k] = R[j]
			j += 1
    


def quick_selection(x, first, last):
	for i in range (first, last):
		minIndex = i
		for j in range (i+1, last+1):
			if x[j] < x[minIndex]:
				minIndex = j
		if minIndex != i:
			x[i], x[minIndex] = x[minIndex], x[i]



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))