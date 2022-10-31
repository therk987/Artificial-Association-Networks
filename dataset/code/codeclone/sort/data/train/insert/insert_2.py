def insetion_sort(my_list):
	"""
	insertion sort algorithm
	param : my_list(unordererd) -> List object

	return : my_list(reordered non-decresed) 
	"""
	for index in range(1,len(my_list)):
		current_element = my_list[index]
		position = index

		while position > 0 and my_list[position -1 ] > current_element :
			my_list[position] = my_list[position - 1]
			position -= 1

		my_list[position] = current_element

	return my_list

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insetion_sort(INPUT_VALUE))