def merge_sort(unsorted_list):
	size = len(unsorted_list)
	if size <= 1:
		return unsorted_list #consider it sorted

	mid = int(size / 2)

	left = unsorted_list[:mid]
	right = unsorted_list[mid:]

	left = merge_sort(left)
	right = merge_sort(right)

	return merge(left,right)

def merge (left,right):
	result = []
	while len(left) > 0 or len(right) > 0: #mientras alguno de los dos tiene elementos
		if len(left) > 0 and len(right) > 0:
			if left[0] <= right[0]:
				result.append(left[0])
				left.remove(left[0])
			else:
				result.append(right[0])
				right.remove(right[0])
		elif len(left) > 0:
			result += left
			left = []
		elif len(right) > 0:
			result += right
			right = []

	return result
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))