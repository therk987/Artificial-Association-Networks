
def merge_sort(array):
	if(len(array) < 2):
		return array

	center = int(len(array)/2)
	
	left = array[:center]

	right = array[center:]
 
	return merge( merge_sort(left), merge_sort(right))


def merge(left_array, right_array):
	result, left_indx, right_indx = [], 0, 0

	while(left_indx < len(left_array) and right_indx < len(right_array)):

		if(left_array[left_indx] < right_array[right_indx]):

			result.append(left_array[left_indx])

			left_indx += 1
		else:
			result.append(right_array[right_indx])

			right_indx += 1

	remainder = [*left_array[left_indx:], *right_array[right_indx:]]

	return [*result, *remainder]		



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))