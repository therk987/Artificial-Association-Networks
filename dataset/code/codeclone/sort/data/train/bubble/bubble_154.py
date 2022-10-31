# Define function that takes in array and outputs sorted array
def bubble_sort(array):

	# Store length of array
	arr_len = len(array)

	# Iterate through array
	for i in range(0, arr_len):

		# Compare each element to all elements succeeding it
		for j in range(1, arr_len):

			# Swap it with elements that are less than it
			if (array[j-1] > array[j]):

				temp = array[j-1]

				array[j-1] = array[j]

				array[j] = temp
	return array
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort(INPUT_VALUE))