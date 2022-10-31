def selectionSort(alist):
	a = alist
	for times in range(len(a)-1, 0, -1):
		largest_num_pos = 0
		for place in range(1, times + 1):
			
			if(a[place] > a[largest_num_pos]):
				largest_num_pos = place

		holder = a[times]
		a[times] = a[largest_num_pos]
		a[largest_num_pos] = holder
		
	return a

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionSort(INPUT_VALUE))