
def divide(array):

	n = len(array)

	if n <= 1:
		return array, 0
	else:

		left = array[:n//2]
		right = array[n//2:]

		leftT = divide(left)
		rightT = divide(right)

		left = leftT[0]
		right = rightT[0]
		inv = leftT[1] + rightT[1]

		return merge(left, right, inv)

	

def merge(array1, array2, inv = 0):

	n = len(array1) + len(array2)

	maximun = 0
	if max(array1) >= max(array2):
		maximun = max(array1)
	else:
		maximun = max(array2)

	if len(array1) < 1:
		maximun = max(array2)

	if len(array2) < 1:
		maximun = max(array2)

	array1.append(maximun+1)
	array2.append(maximun+1)
 
	sorted_array = []

	i = 0
	j = 0

	inversions = 0

	for k in range(0, n):

		if array1[i] >= array2[j]:
			sorted_array.append(array2[j]) 
			if k != (j + n/2) and (array1[i] != array1[-1]):
				inversions += len(array1) - (i+1)
			j += 1

		elif array1[i] < array2[j]:
			sorted_array.append(array1[i])
			i += 1

	inversions += inv

	return sorted_array, inversions


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(divide(INPUT_VALUE))