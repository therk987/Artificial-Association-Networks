
def insertion_sort(array):

	# Loop over Array starting from index one
	for i in range(1, len(array)):

		# Current element
		current_elem = array[i]

		# Index before the current element
		indx_B4 = i - 1

		# Compare the current element to previous elements
		# until no lesser value is found 
		while (indx_B4 >= 0):

			if (current_elem < array[indx_B4] ):

				# Insert
				array[indx_B4 + 1] = array[indx_B4]

				array[indx_B4] = current_elem 

				indx_B4 -= 1

			else:
				break

	return array





INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertion_sort(INPUT_VALUE))